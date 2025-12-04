from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

from buildings_bench import BuildingTypes
from buildings_bench import load_pandas_dataset, benchmark_registry
from buildings_bench.data.datasets import keep_buildings
from buildings_bench import utils
from buildings_bench.evaluation.managers import DatasetMetricsManager


SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


def transfer_learning(args, results_path: Path):
    global benchmark_registry
    lag = 168  # number of autoregressive lags

    # directory for loss curves
    loss_curves_dir = results_path / 'loss_curves'
    loss_curves_dir.mkdir(parents=True, exist_ok=True)

    # remove synthetic
    benchmark_registry = [b for b in benchmark_registry if b != 'buildings-900k-test']
    if args.benchmark[0] == 'all':
        args.benchmark = benchmark_registry

    metrics_manager = DatasetMetricsManager()

    # Subsample buildings using metadata lists unless explicitly disabled
    target_buildings = []
    if not args.dont_subsample_buildings:
        metadata_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata')
        with open(metadata_dir / 'transfer_learning_commercial_buildings.txt', 'r') as f:
            target_buildings += f.read().splitlines()
        with open(metadata_dir / 'transfer_learning_residential_buildings.txt', 'r') as f:
            target_buildings += f.read().splitlines()

    for dataset_name in args.benchmark:
        dataset_generator = load_pandas_dataset(
            dataset_name,
            feature_set='engineered',
            include_outliers=args.include_outliers,
            weather_inputs=['temperature'] if args.use_temperature_input else None
        )

        # Filter to target buildings
        if len(target_buildings) > 0:
            dataset_generator = keep_buildings(dataset_generator, target_buildings)

        # For metrics management
        # Handle missing building_type attribute
        try:
            building_type = dataset_generator.building_type
        except AttributeError:
            print(f"Warning: building_type not found for dataset {dataset_name}, "
                  f"defaulting to residential")
            building_type = BuildingTypes.RESIDENTIAL

        if building_type == BuildingTypes.COMMERCIAL:
            building_types_mask = (
                BuildingTypes.COMMERCIAL_INT * torch.ones([1, 24, 1])
            ).bool()
        else:
            building_types_mask = (
                BuildingTypes.RESIDENTIAL_INT * torch.ones([1, 24, 1])
            ).bool()

        for building_name, bldg_df in dataset_generator:

            # Require at least num_training_days for training + 30 days extra (as in your original)
            if len(bldg_df) < (args.num_training_days + 30) * 24:
                print(f'{dataset_name} {building_name} has too few days {len(bldg_df)}')
                continue

            print(f'dataset {dataset_name} building {building_name}')

            metrics_manager.add_building_to_dataset_if_missing(
                dataset_name, f'{building_name}',
            )

            # Split into fine-tuning (training) and evaluation set by date
            start_timestamp = bldg_df.index[0]
            end_timestamp = start_timestamp + pd.Timedelta(days=args.num_training_days)
            historical_date_range = pd.date_range(
                start=start_timestamp, end=end_timestamp, freq='H'
            )

            training_set = bldg_df.loc[historical_date_range]

            test_set = bldg_df.loc[~bldg_df.index.isin(historical_date_range)]
            test_start_timestamp = test_set.index[0]
            test_end_timestamp = test_start_timestamp + pd.Timedelta(days=180)
            test_set = test_set[test_set.index <= test_end_timestamp]

            print(
                f'fine-tune set date range: {training_set.index[0]} {training_set.index[-1]}, '
                f'test set date range: {test_set.index[0]} {test_set.index[-1]}'
            )

            # ------------------------------------------------------------------
            # 1) Build lagged autoregressive dataset for LightGBM
            # ------------------------------------------------------------------
            feature_cols = [c for c in training_set.columns if c != 'power']

            values = training_set['power'].values  # time-series target
            exog_vals = training_set[feature_cols].values  # exogenous features
            n = len(training_set)

            if n <= lag + 1:
                print(f'{dataset_name} {building_name} has too few training samples for lagged model')
                continue

            X_rows = []
            y_vals = []

            # For each time t, predict power[t] from power[t-lag:t] and exog at time t
            for t in range(lag, n):
                y_vals.append(values[t])
                lagged = values[t - lag:t]                    # shape (lag,)
                exog_t = exog_vals[t]                         # shape (num_exog,)
                X_rows.append(np.concatenate([lagged, exog_t]))

            X = np.vstack(X_rows)  # shape (N_samples, lag + num_exog)
            y = np.array(y_vals)   # shape (N_samples,)

            # ------------------------------------------------------------------
            # 2) Time-based split into training and validation for curves
            #    (use last up to 30 days of samples as validation)
            # ------------------------------------------------------------------
            num_hours_in_training = len(training_set)
            # number of usable AR samples is len(training_set) - lag
            # 30 days of samples -> 30*24 hours
            max_val_hours = 30 * 24
            # bound by number of AR samples
            max_val_samples = min(max_val_hours, len(y) // 5 if len(y) >= 5 else 1)
            # ensure at least 1 validation sample if possible
            val_size = max(1, max_val_samples)
            if val_size >= len(y):
                # fallback: keep at least one training sample
                val_size = max(1, len(y) - 1)

            split_idx = len(y) - val_size

            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]

            # ------------------------------------------------------------------
            # 3) Train LightGBM with eval_set to get train/val loss curves
            # ------------------------------------------------------------------
            model = LGBMRegressor(
                max_depth=-1,
                n_estimators=500,
                learning_rate=0.05,
                n_jobs=24,
                verbose=-1
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_names=['train', 'val'],
                eval_metric='l2'
            )

            evals_result = model.evals_result_
            # Keys are 'train' and 'val' because we set eval_names above
            train_loss = evals_result['train']['l2']
            val_loss = evals_result['val']['l2']

            # ------------------------------------------------------------------
            # 4) Save training + validation loss curves per building
            # ------------------------------------------------------------------
            base_name = f"{dataset_name}_{building_name}"
            base_name = str(base_name).replace('/', '_').replace('\\', '_').replace(' ', '_')

            plt.figure()
            plt.plot(train_loss, label='train')
            plt.plot(val_loss, label='val')
            plt.xlabel('Boosting iteration')
            plt.ylabel('L2 loss')
            plt.title(f'{dataset_name} - {building_name} - Training vs Validation Loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(loss_curves_dir / f'{base_name}_train_val_loss.png')
            plt.close()

            # ------------------------------------------------------------------
            # 5) Forecasting on the test set using autoregressive LightGBM
            #    (manual multi-step ahead similar to ForecasterAutoreg)
            # ------------------------------------------------------------------
            feature_cols_test = [c for c in test_set.columns if c != 'power']

            pred_days = (len(test_set) - lag - 24) // 24
            if pred_days <= 0:
                print(f'{dataset_name} {building_name} has too few test samples for prediction windows.')
                continue

            for i in range(pred_days):

                seq_ptr = lag + 24 * i

                # last_window: last 168 hours before forecast start (within test_set)
                last_window = test_set['power'].iloc[seq_ptr - lag: seq_ptr].values.astype(float).copy()
                ground_truth_df = test_set.iloc[seq_ptr: seq_ptr + 24]

                gt_vals = ground_truth_df['power'].values.astype(float)
                exog_block = ground_truth_df[feature_cols_test].values  # shape (24, num_exog)

                preds = []
                for step in range(24):
                    # Build feature vector for this step: current last_window + exog at this forecast hour
                    x_step = np.concatenate([last_window, exog_block[step]])
                    y_hat = model.predict(x_step.reshape(1, -1))[0]
                    preds.append(y_hat)
                    # update window: drop oldest, append new prediction
                    last_window = np.concatenate([last_window[1:], [y_hat]])

                preds = np.array(preds, dtype=float)

                # Feed into DatasetMetricsManager (same as your original)
                metrics_manager(
                    dataset_name,
                    f'{building_name}',
                    torch.from_numpy(gt_vals).float().view(1, 24, 1),
                    torch.from_numpy(preds).float().view(1, 24, 1),
                    building_types_mask
                )

    print('Generating summaries...')
    variant_name = f':{args.variant_name}' if args.variant_name != '' else ''
    metrics_file = results_path / f'TL_metrics_lightgbm{variant_name}.csv'

    metrics_df = metrics_manager.summary()
    if metrics_file.exists():
        metrics_df.to_csv(metrics_file, mode='a', index=False, header=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument(
        '--benchmark', nargs='+', type=str, default=['all'],
        help='Which datasets in the benchmark to run. Default is ["all."] '
             'See the dataset registry in buildings_bench.data.__init__.py for options.'
    )
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--variant_name', type=str, default='',
        help='Name of the variant. Optional. Used for results files.'
    )
    parser.add_argument('--include_outliers', action='store_true')

    # Transfer learning - data
    parser.add_argument(
        '--num_training_days', type=int, default=180,
        help='Number of days for fine-tuning (last 30 used for early stopping)'
    )
    parser.add_argument(
        '--dont_subsample_buildings', action='store_true', default=False,
        help='Evaluate on all instead of a subsample of 100 res/100 com buildings'
    )
    parser.add_argument(
        '--use_temperature_input', action='store_true',
        help='Include temperature as an additional feature in the model'
    )

    args = parser.parse_args()
    utils.set_seed(args.seed)

    results_path = Path(args.results_path)
    if args.include_outliers:
        results_path = results_path / 'buildingsbench_with_outliers'
    results_path.mkdir(parents=True, exist_ok=True)

    transfer_learning(args, results_path)
