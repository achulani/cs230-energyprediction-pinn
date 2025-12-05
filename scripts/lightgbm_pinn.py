from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    print_metrics_summary
)


SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


class PhysicsInformedNN(nn.Module):
    """Physics-Informed Neural Network for residual correction."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=24):
        super(PhysicsInformedNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def compute_smoothness_loss(predictions):
    """
    Temporal smoothness loss: penalizes rapid oscillations.
    L_smooth = Σ(ŷ_t - 2*ŷ_{t-1} + ŷ_{t-2})²
    """
    if predictions.shape[0] < 3:
        return torch.tensor(0.0, device=predictions.device)
    
    second_diff = predictions[2:] - 2 * predictions[1:-1] + predictions[:-2]
    return torch.mean(second_diff ** 2)


def compute_peak_pattern_loss(predictions, hour_features):
    """
    Peak load pattern loss: penalizes unrealistic daily peak patterns.
    """
    # Compute variance of predictions (should have clear peaks, not flat)
    variance = torch.var(predictions)
    flatness_penalty = torch.exp(-variance)  # Penalize flat predictions
    
    # Penalize negative predictions
    negative_penalty = torch.mean(torch.relu(-predictions))
    
    return flatness_penalty + negative_penalty


def compute_rate_of_change_loss(predictions, max_rate=0.3):
    """
    Rate of change loss: penalizes unrealistic hour-to-hour changes.
    max_rate: maximum allowed fractional change per hour (default 30%)
    """
    if predictions.shape[0] < 2:
        return torch.tensor(0.0, device=predictions.device)
    
    # Compute hour-to-hour changes
    changes = predictions[1:] - predictions[:-1]
    
    # Fractional change relative to mean prediction
    mean_pred = torch.mean(predictions) + 1e-6  # Avoid division by zero
    fractional_changes = torch.abs(changes) / mean_pred
    
    # Penalize changes exceeding max_rate
    violations = torch.relu(fractional_changes - max_rate)
    
    return torch.mean(violations ** 2)


def prepare_pinn_input(lgbm_pred, weather_features, temporal_features, context_stats):
    """
    Prepare input features for PINN.
    
    Args:
        lgbm_pred: (24,) LightGBM predictions
        weather_features: (24, num_weather) weather for next 24 hours
        temporal_features: (24, num_temporal) temporal features
        context_stats: dict with statistics from context window
    
    Returns:
        Flattened feature vector
    """
    features = []
    
    # LightGBM predictions (24 values)
    features.append(lgbm_pred.flatten())
    
    # Summary statistics from LightGBM predictions
    features.append(np.array([
        np.mean(lgbm_pred),
        np.std(lgbm_pred),
        np.min(lgbm_pred),
        np.max(lgbm_pred)
    ]))
    
    # Weather features (flatten)
    if weather_features is not None and len(weather_features) > 0:
        features.append(weather_features.flatten())
    
    # Temporal features (flatten)
    if temporal_features is not None and len(temporal_features) > 0:
        features.append(temporal_features.flatten())
    
    # Context statistics
    if context_stats is not None:
        features.append(np.array([
            context_stats.get('mean', 0),
            context_stats.get('std', 1),
            context_stats.get('min', 0),
            context_stats.get('max', 1)
        ]))
    
    return np.concatenate(features)


def train_pinn(pinn, train_data, val_data, args, device):
    """
    Train the PINN with physics-informed losses.
    
    Args:
        pinn: PhysicsInformedNN model
        train_data: list of (input_features, lgbm_pred, ground_truth, temporal_features)
        val_data: validation data in same format
        args: training arguments
        device: torch device
    
    Returns:
        Trained model and loss history
    """
    optimizer = optim.Adam(pinn.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        pinn.train()
        epoch_train_loss = 0
        epoch_train_mse = 0
        epoch_train_smooth = 0
        epoch_train_peak = 0
        epoch_train_roc = 0
        
        for input_features, lgbm_pred, ground_truth, temporal_features in train_data:
            optimizer.zero_grad()
            
            # Forward pass
            input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(device)
            residual_pred = pinn(input_tensor).squeeze(0)  # (24,)
            
            # Hybrid prediction
            lgbm_tensor = torch.FloatTensor(lgbm_pred).to(device)
            hybrid_pred = lgbm_tensor + residual_pred
            
            # Ground truth
            gt_tensor = torch.FloatTensor(ground_truth).to(device)
            
            # MSE loss (data fitting)
            mse_loss = nn.MSELoss()(hybrid_pred, gt_tensor)
            
            # Physics losses
            smooth_loss = compute_smoothness_loss(hybrid_pred)
            peak_loss = compute_peak_pattern_loss(hybrid_pred, temporal_features)
            roc_loss = compute_rate_of_change_loss(hybrid_pred)
            
            # Total loss with weights
            total_loss = mse_loss
            
            if args.use_smoothness:
                total_loss += args.lambda_smooth * smooth_loss
            if args.use_peak_pattern:
                total_loss += args.lambda_peak * peak_loss
            if args.use_rate_of_change:
                total_loss += args.lambda_roc * roc_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            epoch_train_mse += mse_loss.item()
            epoch_train_smooth += smooth_loss.item()
            epoch_train_peak += peak_loss.item()
            epoch_train_roc += roc_loss.item()
        
        epoch_train_loss /= len(train_data)
        epoch_train_mse /= len(train_data)
        epoch_train_smooth /= len(train_data)
        epoch_train_peak /= len(train_data)
        epoch_train_roc /= len(train_data)
        
        # Validation
        pinn.eval()
        epoch_val_loss = 0
        epoch_val_mse = 0
        
        with torch.no_grad():
            for input_features, lgbm_pred, ground_truth, temporal_features in val_data:
                input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(device)
                residual_pred = pinn(input_tensor).squeeze(0)
                
                lgbm_tensor = torch.FloatTensor(lgbm_pred).to(device)
                hybrid_pred = lgbm_tensor + residual_pred
                
                gt_tensor = torch.FloatTensor(ground_truth).to(device)
                
                mse_loss = nn.MSELoss()(hybrid_pred, gt_tensor)
                
                smooth_loss = compute_smoothness_loss(hybrid_pred)
                peak_loss = compute_peak_pattern_loss(hybrid_pred, temporal_features)
                roc_loss = compute_rate_of_change_loss(hybrid_pred)
                
                total_loss = mse_loss
                if args.use_smoothness:
                    total_loss += args.lambda_smooth * smooth_loss
                if args.use_peak_pattern:
                    total_loss += args.lambda_peak * peak_loss
                if args.use_rate_of_change:
                    total_loss += args.lambda_roc * roc_loss
                
                epoch_val_loss += total_loss.item()
                epoch_val_mse += mse_loss.item()
        
        epoch_val_loss /= len(val_data)
        epoch_val_mse /= len(val_data)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f'    Epoch [{epoch+1}/{args.epochs}] '
                  f'Train Loss: {epoch_train_loss:.6f} '
                  f'(MSE: {epoch_train_mse:.6f}, Smooth: {epoch_train_smooth:.6f}, '
                  f'Peak: {epoch_train_peak:.6f}, RoC: {epoch_train_roc:.6f}) '
                  f'Val Loss: {epoch_val_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(epoch_val_loss)
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = pinn.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'    Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    if best_model_state is not None:
        pinn.load_state_dict(best_model_state)
    
    return pinn, train_losses, val_losses


def pinn_training(args, results_path: Path):
    """Main function for PINN training and evaluation."""
    
    global benchmark_registry
    lag = 168
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    results_path.mkdir(parents=True, exist_ok=True)
    loss_curves_dir = results_path / 'loss_curves'
    loss_curves_dir.mkdir(parents=True, exist_ok=True)
    physics_metrics_dir = results_path / 'physics_metrics'
    physics_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset setup
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
    
    # Store all physics metrics for analysis
    all_physics_metrics = {
        'lgbm': [],
        'hybrid': []
    }
    
    total_buildings = 0
    
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
            
            # Train LightGBM baseline
            print('  Training LightGBM...')
            feature_cols = [c for c in training_set.columns if c != 'power']
            values = training_set['power'].values
            exog_vals = training_set[feature_cols].values
            n = len(training_set)
            
            if n <= lag + 1:
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
            max_val_samples = min(30 * 24, len(y) // 5)
            val_size = max(1, max_val_samples)
            if val_size >= len(y):
                val_size = max(1, len(y) - 1)
            
            split_idx = len(y) - val_size
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]
            
            lgbm_model = LGBMRegressor(
                max_depth=-1,
                n_estimators=500,
                learning_rate=0.05,
                n_jobs=24,
                verbose=-1
            )
            
            lgbm_model.fit(X_train, y_train)
            
            # Collect PINN training data from validation set
            print('  Collecting PINN training data...')
            pinn_train_data = []
            pinn_val_data = []
            
            # Use validation set to create PINN training data
            val_start_idx = split_idx
            num_val_samples = len(y_val)
            
            # Split validation set into PINN train/val (80/20)
            pinn_train_size = int(0.8 * num_val_samples)
            
            for i in range(num_val_samples):
                idx = val_start_idx + i
                
                if idx + 24 <= len(y):
                    # Get next 24 hours of ground truth
                    gt_24h = y[idx:idx+24]
                    
                    # Get next 24 hours of features
                    if idx + 24 <= len(exog_vals):
                        exog_24h = exog_vals[idx:idx+24]
                    else:
                        continue
                    
                    # Compute context statistics from last 168 hours
                    context_window = values[max(0, idx-lag):idx]
                    context_stats = {
                        'mean': np.mean(context_window),
                        'std': np.std(context_window),
                        'min': np.min(context_window),
                        'max': np.max(context_window)
                    }
                    
                    # Create a simple 24-hour LightGBM forecast
                    lgbm_forecast_24h = []
                    last_window = values[idx-lag:idx].copy()
                    
                    for h in range(24):
                        if h < len(exog_24h):
                            x_step = np.concatenate([last_window, exog_24h[h]])
                        else:
                            x_step = np.concatenate([last_window, exog_vals[min(idx, len(exog_vals)-1)]])
                        
                        pred_h = lgbm_model.predict(x_step.reshape(1, -1))[0]
                        lgbm_forecast_24h.append(pred_h)
                        last_window = np.concatenate([last_window[1:], [pred_h]])
                    
                    lgbm_forecast_24h = np.array(lgbm_forecast_24h)
                    
                    # Prepare PINN input
                    pinn_input = prepare_pinn_input(
                        lgbm_forecast_24h,
                        exog_24h,
                        exog_24h,  # temporal features
                        context_stats
                    )
                    
                    # Add to training or validation data
                    data_tuple = (pinn_input, lgbm_forecast_24h, gt_24h, exog_24h)
                    
                    if i < pinn_train_size:
                        pinn_train_data.append(data_tuple)
                    else:
                        pinn_val_data.append(data_tuple)
            
            if len(pinn_train_data) == 0 or len(pinn_val_data) == 0:
                print('  Insufficient data for PINN training, skipping...')
                continue
            
            # Determine input dimension from first sample
            input_dim = len(pinn_train_data[0][0])
            
            # Initialize and train PINN
            print(f'  Training PINN (input_dim={input_dim}, {len(pinn_train_data)} train samples)...')
            pinn = PhysicsInformedNN(
                input_dim=input_dim,
                hidden_dims=args.hidden_dims,
                output_dim=24
            ).to(device)
            
            pinn, train_losses, val_losses = train_pinn(
                pinn, pinn_train_data, pinn_val_data, args, device
            )
            
            # Save loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train')
            plt.plot(val_losses, label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{dataset_name} - {building_name} - PINN Training')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            safe_name = f"{dataset_name}_{building_name}".replace('/', '_').replace('\\', '_').replace(' ', '_')
            plt.savefig(loss_curves_dir / f'{safe_name}_pinn_loss.png')
            plt.close()
            
            # Evaluate on test set
            print('  Evaluating on test set...')
            feature_cols_test = [c for c in test_set.columns if c != 'power']
            
            pred_days = (len(test_set) - lag - 24) // 24
            if pred_days <= 0:
                continue
            
            for i in range(pred_days):
                seq_ptr = lag + 24 * i
                
                last_window = test_set['power'].iloc[seq_ptr - lag: seq_ptr].values.astype(float).copy()
                ground_truth_df = test_set.iloc[seq_ptr: seq_ptr + 24]
                
                gt_vals = ground_truth_df['power'].values.astype(float)
                exog_block = ground_truth_df[feature_cols_test].values
                
                # LightGBM predictions
                lgbm_preds = []
                lgbm_window = last_window.copy()
                
                for step in range(24):
                    x_step = np.concatenate([lgbm_window, exog_block[step]])
                    y_hat = lgbm_model.predict(x_step.reshape(1, -1))[0]
                    lgbm_preds.append(y_hat)
                    lgbm_window = np.concatenate([lgbm_window[1:], [y_hat]])
                
                lgbm_preds = np.array(lgbm_preds, dtype=float)
                
                # PINN correction
                context_stats = {
                    'mean': np.mean(last_window),
                    'std': np.std(last_window),
                    'min': np.min(last_window),
                    'max': np.max(last_window)
                }
                
                pinn_input = prepare_pinn_input(
                    lgbm_preds,
                    exog_block,
                    exog_block,
                    context_stats
                )
                
                with torch.no_grad():
                    pinn.eval()
                    input_tensor = torch.FloatTensor(pinn_input).unsqueeze(0).to(device)
                    residual = pinn(input_tensor).squeeze(0).cpu().numpy()
                
                hybrid_preds = lgbm_preds + residual
                
                # Compute physics metrics
                lgbm_physics = compute_physics_metrics(lgbm_preds, gt_vals)
                hybrid_physics = compute_physics_metrics(hybrid_preds, gt_vals)
                
                all_physics_metrics['lgbm'].append(lgbm_physics)
                all_physics_metrics['hybrid'].append(hybrid_physics)
                
                # Record metrics
                metrics_manager(
                    dataset_name,
                    f'{building_name}',
                    torch.from_numpy(gt_vals).float().view(1, 24, 1),
                    torch.from_numpy(hybrid_preds).float().view(1, 24, 1),
                    building_types_mask
                )
        
        print(f'\nDataset {dataset_name}: Processed {building_count} buildings')
    
    # Save results
    print('\n' + '='*70)
    print('Generating summaries...')
    print('='*70)
    
    # Model description for filename
    model_desc = 'pinn'
    if args.use_smoothness:
        model_desc += '_smooth'
    if args.use_peak_pattern:
        model_desc += '_peak'
    if args.use_rate_of_change:
        model_desc += '_roc'
    if not (args.use_smoothness or args.use_peak_pattern or args.use_rate_of_change):
        model_desc += '_mse_only'
    
    variant_name = f'_{args.variant_name}' if args.variant_name != '' else ''
    
    # Save standard metrics
    metrics_file = results_path / f'metrics_{model_desc}{variant_name}.csv'
    metrics_df = metrics_manager.summary()
    metrics_df.to_csv(metrics_file, index=False)
    
    print(f'\n✅ Standard metrics saved to {metrics_file}')
    print('\nStandard Metrics Summary:')
    print(metrics_df.describe())
    
    # Save physics metrics
    physics_summary = {}
    for key in ['lgbm', 'hybrid']:
        metrics_list = all_physics_metrics[key]
        if len(metrics_list) > 0:
            physics_summary[key] = aggregate_physics_metrics(metrics_list)
    
    physics_file = physics_metrics_dir / f'physics_metrics_{model_desc}{variant_name}.json'
    save_metrics_to_json(physics_summary, physics_file)
    
    # Print comparison
    if 'lgbm' in physics_summary and 'hybrid' in physics_summary:
        print_metrics_summary(physics_summary['lgbm'], "Physics Metrics - LightGBM (within PINN run)")
        print_metrics_summary(physics_summary['hybrid'], f"Physics Metrics - {model_desc.upper()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Physics-Informed Neural Network for Building Energy Forecasting')
    
    # Data arguments
    parser.add_argument('--results_path', type=str, default='results/pinn/')
    parser.add_argument('--benchmark', nargs='+', type=str, default=['all'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--variant_name', type=str, default='')
    parser.add_argument('--include_outliers', action='store_true')
    parser.add_argument('--num_training_days', type=int, default=180)
    parser.add_argument('--dont_subsample_buildings', action='store_true', default=False)
    parser.add_argument('--use_temperature_input', action='store_true')
    
    # PINN architecture
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[64, 32],
                        help='Hidden layer dimensions')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Physics loss arguments
    parser.add_argument('--use_smoothness', action='store_true',
                        help='Use temporal smoothness loss')
    parser.add_argument('--lambda_smooth', type=float, default=0.1,
                        help='Weight for smoothness loss')
    
    parser.add_argument('--use_peak_pattern', action='store_true',
                        help='Use peak pattern loss')
    parser.add_argument('--lambda_peak', type=float, default=0.05,
                        help='Weight for peak pattern loss')
    
    parser.add_argument('--use_rate_of_change', action='store_true',
                        help='Use rate of change loss')
    parser.add_argument('--lambda_roc', type=float, default=0.1,
                        help='Weight for rate of change loss')
    
    args = parser.parse_args()
    utils.set_seed(args.seed)
    
    results_path = Path(args.results_path)
    if args.include_outliers:
        results_path = results_path / 'buildingsbench_with_outliers'
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Model description
    model_desc = 'pinn'
    if args.use_smoothness:
        model_desc += '_smooth'
    if args.use_peak_pattern:
        model_desc += '_peak'
    if args.use_rate_of_change:
        model_desc += '_roc'
    if not (args.use_smoothness or args.use_peak_pattern or args.use_rate_of_change):
        model_desc += '_mse_only'
    
    print('\n' + '='*70)
    print('PHYSICS-INFORMED NEURAL NETWORK TRAINING')
    print('='*70)
    print(f'Configuration:')
    print(f'  Model: {model_desc}')
    print(f'  Hidden dims: {args.hidden_dims}')
    print(f'  Epochs: {args.epochs}')
    print(f'  Learning rate: {args.learning_rate}')
    print(f'  Physics losses:')
    print(f'    Smoothness: {args.use_smoothness} (λ={args.lambda_smooth})')
    print(f'    Peak pattern: {args.use_peak_pattern} (λ={args.lambda_peak})')
    print(f'    Rate of change: {args.use_rate_of_change} (λ={args.lambda_roc})')
    print('='*70 + '\n')
    
    pinn_training(args, results_path)
