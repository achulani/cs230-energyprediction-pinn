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


def load_building_data(
    benchmark: str,
    building_id: str | None = None,
    use_temperature_input: bool = False,
    include_outliers: bool = False,
):
    """
    Load building data with engineered features matching transfer_learning setup.

    Parameters
    ----------
    benchmark : str
        Name of the BuildingsBench benchmark (e.g. 'buildings-100').
    building_id : str or None
        If given, filter to a specific building by name.
    use_temperature_input : bool
        Whether to include temperature as input feature.
    include_outliers : bool
        Whether to include outlier data points.

    Returns
    -------
    bldg_df : pd.DataFrame
        Building dataframe with 'power' target and engineered features.
    building_name : str
        Name of the building.
    """
    dataset_generator = load_pandas_dataset(
        benchmark,
        feature_set='engineered',
        include_outliers=include_outliers,
        weather_inputs=['temperature'] if use_temperature_input else None
    )

    # Get first building or specific building
    for bldg_name, bldg_df in dataset_generator:
        if building_id is None or bldg_name == building_id:
            print(f"[INFO] Loaded building: {bldg_name}")
            print(f"[INFO] Date range: {bldg_df.index[0]} to {bldg_df.index[-1]}")
            print(f"[INFO] Shape: {bldg_df.shape}")
            print(f"[INFO] Columns: {list(bldg_df.columns)}")
            return bldg_df, bldg_name

    raise ValueError(f"Building '{building_id}' not found in benchmark '{benchmark}'")


def train_val_split(
    bldg_df: pd.DataFrame,
    num_training_days: int = 90,
    num_val_days: int = 30,
):
    """
    Chronological split matching transfer learning approach.

    Parameters
    ----------
    bldg_df : pd.DataFrame
        Full building dataframe.
    num_training_days : int
        Number of days for training.
    num_val_days : int
        Number of days for validation.

    Returns
    -------
    train_df : pd.DataFrame
        Training data.
    val_df : pd.DataFrame
        Validation data.
    """
    start_timestamp = bldg_df.index[0]
    train_end = start_timestamp + pd.Timedelta(days=num_training_days)
    val_end = train_end + pd.Timedelta(days=num_val_days)

    train_date_range = pd.date_range(start=start_timestamp, end=train_end, freq='H')
    train_df = bldg_df.loc[bldg_df.index.isin(train_date_range)]

    val_start = train_date_range[-1] + pd.Timedelta(hours=1)
    val_date_range = pd.date_range(start=val_start, end=val_end, freq='H')
    val_df = bldg_df.loc[bldg_df.index.isin(val_date_range)]

    print(f"[INFO] Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} hours)")
    print(f"[INFO] Val: {val_df.index[0]} to {val_df.index[-1]} ({len(val_df)} hours)")

    return train_df, val_df


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_grid_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    lag: int,
    param_grid: dict,
    results_path: Path,
    best_params_path: Path,
):
    """
    Grid search over param_grid for LGBMRegressor with exogenous features.

    Saves:
      - CSV with all trials to results_path
      - Best hyperparameters as JSON to best_params_path
    """
    # Separate target and features
    exog_cols = [col for col in train_df.columns if col != 'power']

    print(f"[INFO] Train size: {len(train_df)}, Val size: {len(val_df)}")
    print(f"[INFO] Using lag = {lag}")
    print(f"[INFO] Exogenous features: {exog_cols}")
    print(f"[INFO] Grid sizes: {[len(v) for v in param_grid.values()]} "
          f"(total combos = {np.prod([len(v) for v in param_grid.values()])})")

    keys = list(param_grid.keys())
    all_combos = list(product(*param_grid.values()))

    results = []

    for i, combo in enumerate(all_combos, start=1):
        params = dict(zip(keys, combo))
        print(f"\n[GRID] ({i}/{len(all_combos)}) Testing params: {params}")

        # Create forecaster with exogenous variables
        reg = LGBMRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        forecaster = ForecasterAutoreg(regressor=reg, lags=lag)

        # Fit on train with exogenous features
        forecaster.fit(
            y=train_df['power'],
            exog=train_df[exog_cols] if len(exog_cols) > 0 else None
        )

        # Predict on validation set using rolling window approach
        # This matches the transfer_learning evaluation logic
        predictions_list = []
        ground_truth_list = []

        pred_days = (len(val_df) - lag - 24) // 24

        for day_idx in range(pred_days):
            seq_ptr = lag + 24 * day_idx

            # Get last window from validation set
            last_window = val_df.iloc[seq_ptr - lag : seq_ptr]
            ground_truth = val_df.iloc[seq_ptr : seq_ptr + 24]

            # Predict next 24 hours
            preds = forecaster.predict(
                steps=24,
                last_window=last_window['power'],
                exog=ground_truth[exog_cols] if len(exog_cols) > 0 else None
            )

            predictions_list.append(preds.values)
            ground_truth_list.append(ground_truth['power'].values)

        # Concatenate all predictions and ground truth
        all_preds = np.concatenate(predictions_list)
        all_truth = np.concatenate(ground_truth_list)

        score_rmse = rmse(all_truth, all_preds)

        print(f"[RESULT] RMSE = {score_rmse:.4f} (over {len(all_preds)} hours)")

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

    print(f"\n[BEST] RMSE = {best_rmse:.4f} with params: {best_params}")
    print(f"[INFO] Saved best params to: {best_params_path}")

    return best_params, best_rmse


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for LGBM + ForecasterAutoreg on BuildingsBench (matching transfer_learning setup)."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="sceaux",
        help="BuildingsBench benchmark name. Available: 'sceaux', 'electricity', 'bdg-2', 'borealis', 'smart', 'ideal', 'lcl', 'buildings-900k-test'.",
    )
    parser.add_argument(
        "--building_id",
        type=str,
        default=None,
        help="Optional building name to select a specific building.",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=168,
        help="Number of autoregressive lags (e.g. 168 for one week of hourly data).",
    )
    parser.add_argument(
        "--num_training_days",
        type=int,
        default=90,
        help="Number of days for training.",
    )
    parser.add_argument(
        "--num_val_days",
        type=int,
        default=30,
        help="Number of days for validation.",
    )
    parser.add_argument(
        "--use_temperature_input",
        action="store_true",
        help="Include temperature as an additional feature.",
    )
    parser.add_argument(
        "--include_outliers",
        action="store_true",
        help="Include outlier data points.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="grid_search_outputs",
        help="Directory to save CSV and JSON with results.",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    benchmark_name = args.benchmark.replace('-', '_')  # Convert dashes to underscores for filenames
    results_path = results_dir / f"grid_results_{benchmark_name}.csv"
    best_params_path = results_dir / f"best_params_{benchmark_name}.json"

    print(f"[INFO] Loading data from benchmark='{args.benchmark}'")
    bldg_df, building_name = load_building_data(
        benchmark=args.benchmark,
        building_id=args.building_id,
        use_temperature_input=args.use_temperature_input,
        include_outliers=args.include_outliers,
    )

    # Check if we have enough data
    min_required_days = args.num_training_days + args.num_val_days
    if len(bldg_df) < min_required_days * 24:
        raise ValueError(
            f"Not enough data: {len(bldg_df)} hours available, "
            f"need at least {min_required_days * 24} hours "
            f"({args.num_training_days} train + {args.num_val_days} val days)"
        )

    train_df, val_df = train_val_split(
        bldg_df,
        num_training_days=args.num_training_days,
        num_val_days=args.num_val_days,
    )

    # ---- HYPERPARAMETER GRID ----
    # Expanded grid to find better parameters than the hardcoded defaults
    param_grid = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 10, 20],
        "min_child_samples": [20, 50, 100],
    }

    run_grid_search(
        train_df=train_df,
        val_df=val_df,
        lag=args.lag,
        param_grid=param_grid,
        results_path=results_path,
        best_params_path=best_params_path,
    )


if __name__ == "__main__":
    main()
