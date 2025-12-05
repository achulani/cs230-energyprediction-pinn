from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder

from buildings_bench import BuildingTypes
from buildings_bench import load_pandas_dataset, benchmark_registry
from buildings_bench.data.datasets import keep_buildings
from buildings_bench import utils
from buildings_bench.evaluation.managers import DatasetMetricsManager


SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


def global_lightgbm(args, results_path: Path):
    global benchmark_registry
    lag = 168  # number of autoregressive lags

    # directory for results
    results_path.mkdir(parents=True, exist_ok=True)

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

    # ------------------------------------------------------------------
    # Phase 1: Collect ALL training data from ALL buildings
    # ------------------------------------------------------------------
    print("Phase 1: Collecting training data from all buildings...")
    
    all_X_train = []
    all_y_train = []
    all_X_val = []
    all_y_val = []
    
    # Store building-specific info for testing later
    building_test_data = {}
    building_info = {}
    
    # Encode building IDs
    building_encoder = LabelEncoder()
    all_building_ids = []

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

        # Get building type for metrics
        try:
            building_type = dataset_generator.building_type
        except AttributeError:
            print(f"Warning: building_type not found for dataset {dataset_name}, "
                  f"defaulting to residential")
            building_type = BuildingTypes.RESIDENTIAL

        for building_name, bldg_df in dataset_generator:
            
            # Require minimum data
            if len(bldg_df) < (args.num_training_days + 30) * 24:
                print(f'{dataset_name} {building_name} has too few days {len(bldg_df)}')
                continue

            print(f'Collecting data from {dataset_name} {building_name}')
            
            building_id = f"{dataset_name}_{building_name}"
            all_building_ids.append(building_id)
            
            # Store building info for later
            building_info[building_id] = {
                'dataset_name': dataset_name,
                'building_name': building_name,
                'building_type': building_type
            }

            # Split into training and test by date
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

            # Build lagged autoregressive dataset
            feature_cols = [c for c in training_set.columns if c != 'power']
            values = training_set['power'].values
            exog_vals = training_set[feature_cols].values
            n = len(training_set)

            if n <= lag + 1:
                print(f'{dataset_name} {building_name} has too few training samples')
                continue

            X_rows = []
            y_vals = []

            # Create lagged features
            for t in range(lag, n):
                y_vals.append(values[t])
                lagged = values[t - lag:t]
                exog_t = exog_vals[t]
                X_rows.append(np.concatenate([lagged, exog_t]))

            X = np.vstack(X_rows)
            y = np.array(y_vals)

            # Add building ID as a feature (will be encoded later)
            building_id_col = np.full((len(X), 1), building_id, dtype=object)

            # Split into train/val (last 30 days for validation)
            max_val_hours = 30 * 24
            max_val_samples = min(max_val_hours, len(y) // 5 if len(y) >= 5 else 1)
            val_size = max(1, max_val_samples)
            if val_size >= len(y):
                val_size = max(1, len(y) - 1)

            split_idx = len(y) - val_size

            X_train_building = X[:split_idx]
            y_train_building = y[:split_idx]
            X_val_building = X[split_idx:]
            y_val_building = y[split_idx:]
            
            building_id_train = building_id_col[:split_idx]
            building_id_val = building_id_col[split_idx:]

            # Append to global training data
            all_X_train.append(np.hstack([X_train_building, building_id_train]))
            all_y_train.append(y_train_building)
            all_X_val.append(np.hstack([X_val_building, building_id_val]))
            all_y_val.append(y_val_building)

            # Store test data for later evaluation
            building_test_data[building_id] = {
                'test_set': test_set,
                'feature_cols': feature_cols
            }

            metrics_manager.add_building_to_dataset_if_missing(
                dataset_name, f'{building_name}'
            )

    # ------------------------------------------------------------------
    # Phase 2: Encode building IDs and prepare global training set
    # ------------------------------------------------------------------
    print("\nPhase 2: Preparing global training set...")
    
    # Fit encoder on all building IDs
    building_encoder.fit(all_building_ids)
    
    # Concatenate all training data
    X_train_global = np.vstack(all_X_train)
    y_train_global = np.concatenate(all_y_train)
    X_val_global = np.vstack(all_X_val)
    y_val_global = np.concatenate(all_y_val)
    
    # Encode building IDs (last column)
    X_train_global[:, -1] = building_encoder.transform(X_train_global[:, -1])
    X_val_global[:, -1] = building_encoder.transform(X_val_global[:, -1])
    
    # Convert to float
    X_train_global = X_train_global.astype(float)
    X_val_global = X_val_global.astype(float)
    
    print(f"Global training set: {X_train_global.shape[0]} samples")
    print(f"Global validation set: {X_val_global.shape[0]} samples")
    print(f"Number of buildings: {len(all_building_ids)}")

    # ------------------------------------------------------------------
    # Phase 3: Train single global LightGBM model
    # ------------------------------------------------------------------
    print("\nPhase 3: Training global LightGBM model...")
    
    global_model = LGBMRegressor(
        max_depth=-1,
        n_estimators=500,
        learning_rate=0.05,
        n_jobs=24,
        verbose=10  # Show progress
    )

    global_model.fit(
        X_train_global, y_train_global,
        eval_set=[(X_train_global, y_train_global), (X_val_global, y_val_global)],
        eval_names=['train', 'val'],
        eval_metric='l2'
    )

    # Save loss curves
    evals_result = global_model.evals_result_
    train_loss = evals_result['train']['l2']
    val_loss = evals_result['val']['l2']

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.xlabel('Boosting iteration')
    plt.ylabel('L2 loss')
    plt.title('Global LightGBM - Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path / 'global_lightgbm_train_val_loss.png')
    plt.close()

    print(f"Final training loss: {train_loss[-1]:.6f}")
    print(f"Final validation loss: {val_loss[-1]:.6f}")

    # ------------------------------------------------------------------
    # Phase 4: Evaluate on each building's test set
    # ------------------------------------------------------------------
    print("\nPhase 4: Evaluating on test sets...")
    
    for building_id, test_info in building_test_data.items():
        
        dataset_name = building_info[building_id]['dataset_name']
        building_name = building_info[building_id]['building_name']
        building_type = building_info[building_id]['building_type']
        
        print(f'Testing on {dataset_name} {building_name}')
        
        test_set = test_info['test_set']
        feature_cols = test_info['feature_cols']
        
        # Set up building type mask for metrics
        if building_type == BuildingTypes.COMMERCIAL:
            building_types_mask = (
                BuildingTypes.COMMERCIAL_INT * torch.ones([1, 24, 1])
            ).bool()
        else:
            building_types_mask = (
                BuildingTypes.RESIDENTIAL_INT * torch.ones([1, 24, 1])
            ).bool()
        
        # Encode this building's ID
        building_id_encoded = building_encoder.transform([building_id])[0]
        
        # Forecasting using autoregressive approach
        pred_days = (len(test_set) - lag - 24) // 24
        if pred_days <= 0:
            print(f'{building_id} has too few test samples for prediction windows.')
            continue

        for i in range(pred_days):
            seq_ptr = lag + 24 * i

            last_window = test_set['power'].iloc[seq_ptr - lag: seq_ptr].values.astype(float).copy()
            ground_truth_df = test_set.iloc[seq_ptr: seq_ptr + 24]

            gt_vals = ground_truth_df['power'].values.astype(float)
            exog_block = ground_truth_df[feature_cols].values

            preds = []
            for step in range(24):
                # Build feature vector: lagged values + exog + building_id
                x_step = np.concatenate([
                    last_window, 
                    exog_block[step],
                    [building_id_encoded]
                ])
                y_hat = global_model.predict(x_step.reshape(1, -1))[0]
                preds.append(y_hat)
                # Update window
                last_window = np.concatenate([last_window[1:], [y_hat]])

            preds = np.array(preds, dtype=float)

            # Record metrics
            metrics_manager(
                dataset_name,
                f'{building_name}',
                torch.from_numpy(gt_vals).float().view(1, 24, 1),
                torch.from_numpy(preds).float().view(1, 24, 1),
                building_types_mask
            )

    # ------------------------------------------------------------------
    # Phase 5: Save results
    # ------------------------------------------------------------------
    print('\nPhase 5: Generating summaries...')
    variant_name = f':{args.variant_name}' if args.variant_name != '' else ''
    metrics_file = results_path / f'TL_metrics_global_lightgbm{variant_name}.csv'

    metrics_df = metrics_manager.summary()
    metrics_df.to_csv(metrics_file, index=False)
    
    print(f"\nResults saved to {metrics_file}")
    print("\nSummary statistics:")
    print(metrics_df.describe())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path', type=str, default='results/global_lgbm/')
    parser.add_argument(
        '--benchmark', nargs='+', type=str, default=['all'],
        help='Which datasets in the benchmark to run. Default is ["all"]'
    )
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--variant_name', type=str, default='',
        help='Name of the variant. Optional. Used for results files.'
    )
    parser.add_argument('--include_outliers', action='store_true')
    parser.add_argument(
        '--num_training_days', type=int, default=180,
        help='Number of days for training'
    )
    parser.add_argument(
        '--dont_subsample_buildings', action='store_true', default=False,
        help='Evaluate on all instead of a subsample.'
    )
    parser.add_argument(
        '--use_temperature_input', action='store_true',
        help='Include temperature as an additional feature.'
    )

    args = parser.parse_args()
    utils.set_seed(args.seed)

    results_path = Path(args.results_path)
    if args.include_outliers:
        results_path = results_path / 'buildingsbench_with_outliers'
    results_path.mkdir(parents=True, exist_ok=True)

    global_lightgbm(args, results_path)
