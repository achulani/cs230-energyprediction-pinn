"""
LightGBM Baseline with Physics Evaluation

Trains per-building LightGBM models and evaluates with both:
1. Standard metrics (CVRMSE, MAE) 
2. Physics-based metrics (smoothness, gradient violations, negatives)

Outputs:
- Standard metrics CSV (from DatasetMetricsManager)
- Physics metrics JSON (aggregated across all windows)
- Per-building physics metrics CSV (one row per building)
- Loss curves (one per building)
"""

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

# Import evaluation metrics
from eval_metrics import (
    compute_physics_metrics, 
    aggregate_physics_metrics, 
    save_metrics_to_json,
    print_metrics_summary,
    create_per_building_metrics_df
)


SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


def baseline_with_physics(args, results_path: Path):
    """
    Train LightGBM baseline and evaluate with both standard and physics metrics.
    """
    global benchmark_registry
    lag = 168

    # Directory setup
    loss_curves_dir = results_path / 'loss_curves'
    loss_curves_dir.mkdir(parents=True, exist_ok=True)
    physics_metrics_dir = results_path / 'physics_metrics'
    physics_metrics_dir.mkdir(parents=True, exist_ok=True)

    # Remove synthetic
    benchmark_registry = [b for b in benchmark_registry if b != 'buildings-900k-test']
    if args.benchmark[0] == 'all':
        args.benchmark = benchmark_registry

    metrics_manager = DatasetMetricsManager()

    # Subsample buildings
    target_buildings = []
    if not args.dont_subsample_buildings:
        metadata_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata')
        with open(metadata_dir / 'transfer_learning_commercial_buildings.txt', 'r') as f:
            target_buildings += f.read().splitlines()
        with open(metadata_dir / 'transfer_learning_residential_buildings.txt', 'r') as f:
            target_buildings += f.read().splitlines()

    # Store physics metrics (list of all windows)
    all_physics_metrics = []
    total_buildings = 0

    print('\n' + '='*70)
    print('LIGHTGBM BASELINE WITH PHYSICS EVALUATION')
    print('='*70)

    for dataset_name in args.benchmark:
        print(f'\n{"="*70}')
        print(f'Processing dataset: {dataset_name}')
        print(f'{"="*70}')
        
        dataset_generator = load_pandas_dataset(
            dataset_name,
            feature_set='engineered',
            include_outliers=args.include_outliers,
            weather_inputs=['temperature'] if args.use_temperature_input else None
        )

        if len(target_buildings) > 0:
            dataset_generator = keep_buildings(dataset_generator, target_buildings)

        try:
            building_type = dataset_generator.building_type
        except AttributeError:
            building_type = BuildingTypes.RESIDENTIAL

        if building_type == BuildingTypes.COMMERCIAL:
            building_types_mask = (BuildingTypes.COMMERCIAL_INT * torch.ones([1, 24, 1])).bool()
        else:
            building_types_mask = (BuildingTypes.RESIDENTIAL_INT * torch.ones([1, 24, 1])).bool()

        building_count = 0

        for building_name, bldg_df in dataset_generator:

            if len(bldg_df) < (args.num_training_days + 30) * 24:
                continue

            total_buildings += 1
            building_count += 1
            print(f'\n[{total_buildings}] {dataset_name}/{building_name}')

            metrics_manager.add_building_to_dataset_if_missing(dataset_name, f'{building_name}')

            # Split data
            start_timestamp = bldg_df.index[0]
            end_timestamp = start_timestamp + pd.Timedelta(days=args.num_training_days)
            historical_date_range = pd.date_range(start=start_timestamp, end=end_timestamp, freq='H')

            training_set = bldg_df.loc[historical_date_range]
            test_set = bldg_df.loc[~bldg_df.index.isin(historical_date_range)]
            test_start_timestamp = test_set.index[0]
            test_end_timestamp = test_start_timestamp + pd.Timedelta(days=180)
            test_set = test_set[test_set.index <= test_end_timestamp]

            print(f'  Training: {len(training_set)//24} days, Test: {len(test_set)//24} days')

            # Build lagged autoregressive dataset
            feature_cols = [c for c in training_set.columns if c != 'power']
            values = training_set['power'].values
            exog_vals = training_set[feature_cols].values
            n = len(training_set)

            if n <= lag + 1:
                print(f'  SKIP: Insufficient training data')
                continue

            X_rows = []
            y_vals = []

            for t in range(lag, n):
                y_vals.append(values[t])
                lagged = values[t - lag:t]
                exog_t = exog_vals[t]
                X_rows.append(np.concatenate([lagged, exog_t]))

            X = np.vstack(X_rows)
            y = np.array(y_vals)

            # Train/val split
            max_val_hours = 30 * 24
            max_val_samples = min(max_val_hours, len(y) // 5 if len(y) >= 5 else 1)
            val_size = max(1, max_val_samples)
            if val_size >= len(y):
                val_size = max(1, len(y) - 1)

            split_idx = len(y) - val_size
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]

            # Train LightGBM
            print(f'  Training LightGBM...')
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
            train_loss = evals_result['train']['l2']
            val_loss = evals_result['val']['l2']

            # Save loss curves
            base_name = f"{dataset_name}_{building_name}"
            base_name = str(base_name).replace('/', '_').replace('\\', '_').replace(' ', '_')

            plt.figure(figsize=(10, 6))
            plt.plot(train_loss, label='Train')
            plt.plot(val_loss, label='Validation')
            plt.xlabel('Boosting Iteration')
            plt.ylabel('L2 Loss')
            plt.title(f'{dataset_name} - {building_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(loss_curves_dir / f'{base_name}_loss.png', dpi=150)
            plt.close()

            # Evaluate on test set
            print(f'  Evaluating on test set...')
            feature_cols_test = [c for c in test_set.columns if c != 'power']
            pred_days = (len(test_set) - lag - 24) // 24
            
            if pred_days <= 0:
                print(f'  SKIP: Insufficient test data')
                continue

            window_count = 0
            for i in range(pred_days):
                seq_ptr = lag + 24 * i

                last_window = test_set['power'].iloc[seq_ptr - lag: seq_ptr].values.astype(float).copy()
                ground_truth_df = test_set.iloc[seq_ptr: seq_ptr + 24]

                gt_vals = ground_truth_df['power'].values.astype(float)
                exog_block = ground_truth_df[feature_cols_test].values

                # Autoregressive predictions
                preds = []
                for step in range(24):
                    x_step = np.concatenate([last_window, exog_block[step]])
                    y_hat = model.predict(x_step.reshape(1, -1))[0]
                    preds.append(y_hat)
                    last_window = np.concatenate([last_window[1:], [y_hat]])

                preds = np.array(preds, dtype=float)

                # Compute physics metrics for this window
                physics_metrics = compute_physics_metrics(preds, gt_vals)
                physics_metrics['dataset'] = dataset_name
                physics_metrics['building'] = building_name
                physics_metrics['window_idx'] = i
                all_physics_metrics.append(physics_metrics)

                # Standard metrics (CVRMSE, etc.)
                metrics_manager(
                    dataset_name,
                    f'{building_name}',
                    torch.from_numpy(gt_vals).float().view(1, 24, 1),
                    torch.from_numpy(preds).float().view(1, 24, 1),
                    building_types_mask
                )
                
                window_count += 1

            print(f'  Evaluated {window_count} windows')

        print(f'\nDataset {dataset_name}: Processed {building_count} buildings')

    # ========================================
    # Save Results
    # ========================================
    print('\n' + '='*70)
    print('Saving Results...')
    print('='*70)
    
    variant_name = f'_{args.variant_name}' if args.variant_name != '' else ''
    
    # 1. Standard metrics CSV (CVRMSE, MAE, etc.)
    print('\n1. Standard Metrics (CVRMSE, MAE)...')
    metrics_file = results_path / f'metrics_standard{variant_name}.csv'
    metrics_df = metrics_manager.summary()
    metrics_df.to_csv(metrics_file, index=False)
    print(f'   ✅ Saved to {metrics_file}')
    print(f'   Summary:\n{metrics_df.describe()}')

    # 2. Aggregated physics metrics JSON (all windows combined)
    print('\n2. Aggregated Physics Metrics (all windows)...')
    if len(all_physics_metrics) > 0:
        physics_summary = aggregate_physics_metrics(all_physics_metrics)
        
        physics_file = physics_metrics_dir / f'physics_aggregated{variant_name}.json'
        save_metrics_to_json(physics_summary, physics_file)
        
        print_metrics_summary(physics_summary, "Physics Metrics - All Windows")

    # 3. Per-building physics metrics CSV
    print('\n3. Per-Building Physics Metrics...')
    if len(all_physics_metrics) > 0:
        per_building_df = create_per_building_metrics_df(all_physics_metrics)
        
        per_building_file = results_path / f'physics_per_building{variant_name}.csv'
        per_building_df.to_csv(per_building_file, index=False)
        print(f'   ✅ Saved to {per_building_file}')
        print(f'   Shape: {per_building_df.shape[0]} buildings x {per_building_df.shape[1]} columns')
        
        # Show sample
        print(f'\n   Sample (first 3 buildings):')
        display_cols = [
            'dataset', 'building', 
            'smoothness_violation_mean_mean',
            'max_gradient_mean',
            'negative_count_mean',
            'num_windows'
        ]
        available_cols = [c for c in display_cols if c in per_building_df.columns]
        print(per_building_df[available_cols].head(3).to_string(index=False))

    print('\n' + '='*70)
    print('COMPLETE!')
    print('='*70)
    print(f'\nTotal buildings processed: {total_buildings}')
    print(f'Total prediction windows: {len(all_physics_metrics)}')
    print(f'\nResults saved to: {results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LightGBM Baseline with Physics Metrics')

    parser.add_argument('--results_path', type=str, default='results/baseline/')
    parser.add_argument('--benchmark', nargs='+', type=str, default=['all'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--variant_name', type=str, default='')
    parser.add_argument('--include_outliers', action='store_true')
    parser.add_argument('--num_training_days', type=int, default=180)
    parser.add_argument('--dont_subsample_buildings', action='store_true', default=False)
    parser.add_argument('--use_temperature_input', action='store_true')

    args = parser.parse_args()
    utils.set_seed(args.seed)

    results_path = Path(args.results_path)
    if args.include_outliers:
        results_path = results_path / 'buildingsbench_with_outliers'
    results_path.mkdir(parents=True, exist_ok=True)

    baseline_with_physics(args, results_path)
