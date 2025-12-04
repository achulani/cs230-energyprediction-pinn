#!/usr/bin/env python

import os
import json
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg

from buildings_bench import load_pandas_dataset


def load_series(benchmark: str, building_id: int | None = None, target_col: str = "electricity") -> pd.Series:
    """
    Load a univariate time series y(t) from a BuildingsBench benchmark.

    Parameters
    ----------
    benchmark : str
        Name of the BuildingsBench benchmark (e.g. 'buildings-100').
    building_id : int or None
        If given, filter the dataset to a single building_id.
    target_col : str
        Name of the column to use as the target (default: 'electricity').

    Returns
    -------
    y : pd.Series
        Target time series sorted by timestamp and indexed by timestamp.
    """
    df = load_pandas_dataset(benchmark)

    # Optional: filter to a single building
    if building_id is not None and "building_id" in df.columns:
        df = df[df["building_id"] == building_id].copy()

    # Sort and ensure timestamp is datetime index
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in the dataset.")

    df = df.sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df.columns}.")

    y = df[target_col].astype(float)

    if len(y) < 500:
        print(f"[WARN] Very short series (len={len(y)}). Consider using a larger subset / different building.")

    return y


def train_val_split(y: pd.Series, val_frac: float = 0.2):
    """
    Simple chronological split: first (1 - val_frac) for train, last val_frac for validation.
    """
    n = len(y)
    val_size = int(n * val_frac)
    if val_size == 0:
        raise ValueError("Validation split is empty; increase dataset size or val_frac.")
    train = y.iloc[:-val_size]
    val = y.iloc[-val_size:]
    return train, val


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_grid_search(
    y: pd.Series,
    lag: int,
    param_grid: dict,
    results_path: Path,
    best_params_path: Path,
):
    """
    Brute-force grid search over param_grid for an LGBMRegressor wrapped in ForecasterAutoreg.

    Saves:
      - a CSV with all trials to results_path
      - best hyperparameters as JSON to best_params_path
    """
    y_train, y_val = train_val_split(y, val_frac=0.2)

    print(f"[INFO] Series length: {len(y)}  | train: {len(y_train)}  val: {len(y_val)}")
    print(f"[INFO] Using lag = {lag}")
    print(f"[INFO] Grid sizes: {[len(v) for v in param_grid.values()]} "
          f"(total combos = {np.prod([len(v) for v in param_grid.values()])})")

    keys = list(param_grid.keys())
    all_combos = list(product(*param_grid.values()))

    results = []

    for i, combo in enumerate(all_combos, start=1):
        params = dict(zip(keys, combo))
        print(f"\n[GRID] ({i}/{len(all_combos)}) Testing params: {params}")

        # Create model + forecaster
        reg = LGBMRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
        )
        forecaster = ForecasterAutoreg(regressor=reg, lags=lag)

        # Fit on train
        forecaster.fit(y=y_train)

        # Forecast exactly len(y_val) steps ahead
        y_pred = forecaster.predict(steps=len(y_val))

        score_rmse = rmse(y_val.values, y_pred.values)

        print(f"[RESULT] RMSE = {score_rmse:.4f}")

        row = {"rmse": score_rmse}
        row.update(params)
        results.append(row)

    # Save all results
    results_df = pd.DataFrame(results).sort_values("rmse")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] Saved all grid search results to: {results_path}")

    # Best params
    best_row = results_df.iloc[0]
    best_params = {k: best_row[k] for k in keys}
    best_rmse = float(best_row["rmse"])

    with best_params_path.open("w") as f:
        json.dump(
            {
                "best_rmse": best_rmse,
                "best_params": best_params,
            },
            f,
            indent=2,
        )

    print(f"[BEST] RMSE = {best_rmse:.4f} with params: {best_params}")
    print(f"[INFO] Saved best params to: {best_params_path}")


def main():
    parser = argparse.ArgumentParser(description="Grid search for LGBM + ForecasterAutoreg on BuildingsBench.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="buildings-100",
        help="BuildingsBench benchmark name (e.g. 'buildings-100', 'buildings-900k-train').",
    )
    parser.add_argument(
        "--building_id",
        type=int,
        default=None,
        help="Optional building_id to filter to a single building (if column exists).",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=168,
        help="Number of autoregressive lags (e.g. 168 for one week of hourly data).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="grid_search_outputs",
        help="Directory to save CSV and JSON with results.",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_path = results_dir / f"grid_results_{args.benchmark}.csv"
    best_params_path = results_dir / f"best_params_{args.benchmark}.json"

    print(f"[INFO] Loading data from benchmark='{args.benchmark}', building_id={args.building_id}")
    y = load_series(benchmark=args.benchmark, building_id=args.building_id)

    # ---- HYPERPARAMETER GRID ----
    # You should adjust this based on how heavy you want the search to be.
    param_grid = {
        "n_estimators": [300, 500],
        "learning_rate": [0.01, 0.05],
        "num_leaves": [31, 63],
        "max_depth": [-1, 10],
        "min_child_samples": [20, 100],
    }

    run_grid_search(
        y=y,
        lag=args.lag,
        param_grid=param_grid,
        results_path=results_path,
        best_params_path=best_params_path,
    )


if __name__ == "__main__":
    main()
