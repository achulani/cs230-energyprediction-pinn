"""
Physics-Informed Neural Network (PINN) for Building Energy Forecasting

Trains per-building LightGBM + PINN models with optional physics constraints:
- Smoothness loss: Penalizes oscillations (thermal inertia)
- Temporal gradient loss: Limits rate of change (HVAC ramp rates)
- Weather consistency loss: Enforces temperature-power relationship

Outputs:
- Standard metrics CSV (CVRMSE, MAE)
- Physics metrics JSON (aggregated)
- Per-building physics metrics CSV
- Loss curves (training plots)
"""

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
    print_metrics_summary,
    create_per_building_metrics_df
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
    
    Physical motivation: Buildings have thermal inertia - temperature 
    and power consumption change gradually, not erratically.
    """
    if predictions.shape[0] < 3:
        return torch.tensor(0.0, device=predictions.device)
    
    second_diff = predictions[2:] - 2 * predictions[1:-1] + predictions[:-2]
    return torch.mean(second_diff ** 2)


def compute_temporal_gradient_loss(predictions, max_gradient=0.3):
    """
    Temporal gradient loss: limits hour-to-hour changes.
    
    Physical motivation: HVAC systems have finite ramp rates - 
    realistic power changes are typically <30% per hour.
    
    Args:
        predictions: (24,) tensor of predictions
        max_gradient: maximum allowed fractional change per hour
    """
    if predictions.shape[0] < 2:
        return torch.tensor(0.0, device=predictions.device)
    
    # Compute hour-to-hour changes
    changes = predictions[1:] - predictions[:-1]
    
    # Fractional change relative to mean prediction
    mean_pred = torch.mean(predictions) + 1e-6
    fractional_changes = torch.abs(changes) / mean_pred
    
    # Penalize changes exceeding max_gradient
    violations = torch.relu(fractional_changes - max_gradient)
    
    return torch.mean(violations ** 2)


def compute_weather_consistency_loss(predictions, temperature, baseline_temp=20.0):
    """
    Weather consistency loss: ensures power responds appropriately to temperature.
    
    Physical motivation: Temperature drives HVAC loads - heating increases 
    when cold, cooling increases when hot.
    
    Args:
        predictions: (24,) tensor of power predictions
        temperature: (24,) tensor of temperature forecasts
        baseline_temp: baseline temperature (typically 20°C)
    """
    if temperature is None or len(temperature) == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    # Convert temperature to tensor if needed
    if not isinstance(temperature, torch.Tensor):
        temperature = torch.FloatTensor(temperature).to(predictions.device)
    
    # Temperature deviation from baseline (comfort temperature)
    temp_deviation = torch.abs(temperature - baseline_temp)
    
    # Power should correlate with temperature deviation
    # (more extreme temperatures = more heating/cooling needed)
    power_mean = torch.mean(predictions)
    power_std = torch.std(predictions) + 1e-6
    
    # Normalize predictions
    power_normalized = (predictions - power_mean) / power_std
    
    # Normalize temperature deviations
    temp_mean = torch.mean(temp_deviation)
    temp_std = torch.std(temp_deviation) + 1e-6
    temp_normalized = (temp_deviation - temp_mean) / temp_std
    
    # Correlation loss: penalize negative correlation
    # (power should increase with temperature deviation)
    correlation = torch.mean(power_normalized * temp_normalized)
    
    # Penalize if correlation is negative (anti-physical)
    return torch.relu(-correlation) ** 2


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
        # If multi-dimensional, flatten
        if weather_features.ndim > 1:
            features.append(weather_features.flatten())
        else:
            features.append(weather_features)
    
    # Temporal features (flatten)
    if temporal_features is not None and len(temporal_features) > 0:
        if temporal_features.ndim > 1:
            features.append(temporal_features.flatten())
        else:
            features.append(temporal_features)
    
    # Context statistics
    if context_stats is not None:
        features.append(np.array([
            context_stats.get('mean', 0),
            context_stats.get('std', 1),
            context_stats.get('min', 0),
            context_stats.get('max', 1)
        ]))
    
    return np.concatenate(features)


def extract_temperature_from_features(exog_features):
    """
    Extract temperature from exogenous features.
    Assumes temperature is the first feature if available.
    """
    if exog_features is not None and len(exog_features) > 0:
        # If 2D array, take first column
        if exog_features.ndim > 1 and exog_features.shape[1] > 0:
            return exog_features[:, 0]
        # If 1D array, return as is
        elif exog_features.ndim == 1:
            return exog_features
    return None


def train_pinn(pinn, train_data, val_data, args, device):
    """
    Train the PINN with physics-informed losses.
    
    Args:
        pinn: PhysicsInformedNN model
        train_data: list of (input_features, lgbm_pred, ground_truth, weather, temporal)
        val_data: validation data in same format
        args: training arguments
        device: torch device
    
    Returns:
        Trained model and loss history
    """
    optimizer = optim.Adam(pinn.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
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
        epoch_train_temporal = 0
        epoch_train_weather = 0
        
        for input_features, lgbm_pred, ground_truth, weather, temporal in train_data:
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
            
            # Physics losses (initialize as zero)
            smooth_loss = torch.tensor(0.0, device=device)
            temporal_loss = torch.tensor(0.0, device=device)
            weather_loss = torch.tensor(0.0, device=device)
            
            if args.use_smoothness:
                smooth_loss = compute_smoothness_loss(hybrid_pred)
            
            if args.use_temporal:
                temporal_loss = compute_temporal_gradient_loss(hybrid_pred, max_gradient=args.max_gradient)
            
            if args.use_weather and weather is not None:
                weather_loss = compute_weather_consistency_loss(hybrid_pred, weather)
            
            # Total loss with weights
            total_loss = mse_loss
            
            if args.use_smoothness:
                total_loss += args.lambda_smooth * smooth_loss
            if args.use_temporal:
                total_loss += args.lambda_temporal * temporal_loss
            if args.use_weather:
                total_loss += args.lambda_weather * weather_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_train_loss += total_loss.item()
            epoch_train_mse += mse_loss.item()
            epoch_train_smooth += smooth_loss.item()
            epoch_train_temporal += temporal_loss.item()
            epoch_train_weather += weather_loss.item()
        
        epoch_train_loss /= len(train_data)
        epoch_train_mse /= len(train_data)
        epoch_train_smooth /= len(train_data)
        epoch_train_temporal /= len(train_data)
        epoch_train_weather /= len(train_data)
        
        # Validation
        pinn.eval()
        epoch_val_loss = 0
        epoch_val_mse = 0
        
        with torch.no_grad():
            for input_features, lgbm_pred, ground_truth, weather, temporal in val_data:
                input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(device)
                residual_pred = pinn(input_tensor).squeeze(0)
                
                lgbm_tensor = torch.FloatTensor(lgbm_pred).to(device)
                hybrid_pred = lgbm_tensor + residual_pred
                
                gt_tensor = torch.FloatTensor(ground_truth).to(device)
                
                mse_loss = nn.MSELoss()(hybrid_pred, gt_tensor)
                
                smooth_loss = torch.tensor(0.0, device=device)
                temporal_loss = torch.tensor(0.0, device=device)
                weather_loss = torch.tensor(0.0, device=device)
                
                if args.use_smoothness:
                    smooth_loss = compute_smoothness_loss(hybrid_pred)
                if args.use_temporal:
                    temporal_loss = compute_temporal_gradient_loss(hybrid_pred, max_gradient=args.max_gradient)
                if args.use_weather and weather is not None:
                    weather_loss = compute_weather_consistency_loss(hybrid_pred, weather)
                
                total_loss = mse_loss
                if args.use_smoothness:
                    total_loss += args.lambda_smooth * smooth_loss
                if args.use_temporal:
                    total_loss += args.lambda_temporal * temporal_loss
                if args.use_weather:
                    total_loss += args.lambda_weather * weather_loss
                
                epoch_val_loss += total_loss.item()
                epoch_val_mse += mse_loss.item()
        
        epoch_val_loss /= len(val_data)
        epoch_val_mse /= len(val_data)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'    Epoch [{epoch+1}/{args.epochs}] '
                  f'Train: {epoch_train_loss:.6f} '
                  f'(MSE: {epoch_train_mse:.6f}, S: {epoch_train_smooth:.6f}, '
                  f'T: {epoch_train_temporal:.6f}, W: {epoch_train_weather:.6f}) '
                  f'Val: {epoch_val_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(epoch_val_loss)
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
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
            
            # Filter to specific building if requested (for grid search)
            if args.building_filter and building_name != args.building_filter:
                continue
            
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
                    
                    # Create a 24-hour LightGBM forecast
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
                    
                    # Extract temperature if using weather loss
                    temperature = None
                    if args.use_weather:
                        temperature = extract_temperature_from_features(exog_24h)
                    
                    # Prepare PINN input
                    pinn_input = prepare_pinn_input(
                        lgbm_forecast_24h,
                        exog_24h,
                        exog_24h,  # temporal features
                        context_stats
                    )
                    
                    # Add to training or validation data
                    data_tuple = (pinn_input, lgbm_forecast_24h, gt_24h, temperature, exog_24h)
                    
                    if i < pinn_train_size:
                        pinn_train_data.append(data_tuple)
                    else:
                        pinn_val_data.append(data_tuple)
            
            if len(pinn_train_data) == 0 or len(pinn_val_data) == 0:
                print('  SKIP: Insufficient data for PINN training')
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
            plt.plot(train_losses, label='Train', linewidth=2)
            plt.plot(val_losses, label='Validation', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{dataset_name} - {building_name} - PINN Training')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            safe_name = f"{dataset_name}_{building_name}".replace('/', '_').replace('\\', '_').replace(' ', '_')
            plt.savefig(loss_curves_dir / f'{safe_name}_pinn_loss.png', dpi=150)
            plt.close()
            
            # Evaluate on test set
            print('  Evaluating on test set...')
            feature_cols_test = [c for c in test_set.columns if c != 'power']
            
            pred_days = (len(test_set) - lag - 24) // 24
            if pred_days <= 0:
                continue
            
            window_count = 0
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
                
                # Extract temperature
                temperature = None
                if args.use_weather:
                    temperature = extract_temperature_from_features(exog_block)
                
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
                
                # Compute physics metrics for both LightGBM and Hybrid
                lgbm_physics = compute_physics_metrics(lgbm_preds, gt_vals)
                lgbm_physics['dataset'] = dataset_name
                lgbm_physics['building'] = building_name
                lgbm_physics['window_idx'] = i
                all_physics_metrics['lgbm'].append(lgbm_physics)
                
                hybrid_physics = compute_physics_metrics(hybrid_preds, gt_vals)
                hybrid_physics['dataset'] = dataset_name
                hybrid_physics['building'] = building_name
                hybrid_physics['window_idx'] = i
                all_physics_metrics['hybrid'].append(hybrid_physics)
                
                # Record standard metrics (for hybrid only)
                metrics_manager(
                    dataset_name,
                    f'{building_name}',
                    torch.from_numpy(gt_vals).float().view(1, 24, 1),
                    torch.from_numpy(hybrid_preds).float().view(1, 24, 1),
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
    
    # Model description for filename
    model_desc = 'pinn'
    if args.use_smoothness:
        model_desc += '_smooth'
    if args.use_temporal:
        model_desc += '_temporal'
    if args.use_weather:
        model_desc += '_weather'
    if not (args.use_smoothness or args.use_temporal or args.use_weather):
        model_desc += '_mse_only'
    
    variant_name = f'_{args.variant_name}' if args.variant_name != '' else ''
    
    # 1. Standard metrics CSV
    print('\n1. Standard Metrics (CVRMSE, MAE)...')
    metrics_file = results_path / f'metrics_{model_desc}{variant_name}.csv'
    metrics_df = metrics_manager.summary()
    metrics_df.to_csv(metrics_file, index=False)
    print(f'   ✅ Saved to {metrics_file}')
    
    # 2. Aggregated physics metrics JSON
    print('\n2. Aggregated Physics Metrics...')
    physics_summary = {}
    for key in ['lgbm', 'hybrid']:
        if len(all_physics_metrics[key]) > 0:
            physics_summary[key] = aggregate_physics_metrics(all_physics_metrics[key])
    
    physics_file = physics_metrics_dir / f'physics_aggregated_{model_desc}{variant_name}.json'
    save_metrics_to_json(physics_summary, physics_file)
    
    # Print summaries
    if 'lgbm' in physics_summary:
        print_metrics_summary(physics_summary['lgbm'], "Physics Metrics - LightGBM Baseline")
    if 'hybrid' in physics_summary:
        print_metrics_summary(physics_summary['hybrid'], f"Physics Metrics - {model_desc.upper()}")
    
    # 3. Per-building physics metrics CSV
    print('\n3. Per-Building Physics Metrics...')
    for key in ['lgbm', 'hybrid']:
        if len(all_physics_metrics[key]) > 0:
            per_building_df = create_per_building_metrics_df(all_physics_metrics[key])
            per_building_file = physics_metrics_dir / f'physics_per_building_{key}_{model_desc}{variant_name}.csv'
            per_building_df.to_csv(per_building_file, index=False)
            print(f'   ✅ {key}: {per_building_file}')
    
    print(f'\n{"="*70}')
    print(f'✅ Training and evaluation completed!')
    print(f'   Total buildings processed: {total_buildings}')
    print(f'   Results saved to: {results_path}')
    print(f'{"="*70}')


def main():
    """Parse arguments and run PINN training."""
    parser = argparse.ArgumentParser(description='PINN Building Energy Forecasting')
    
    # Dataset arguments
    parser.add_argument('--benchmark', type=str, nargs='+', default=['all'],
                        help='Benchmark dataset(s) to use')
    parser.add_argument('--num_training_days', type=int, default=365,
                        help='Number of days to use for training')
    parser.add_argument('--include_outliers', action='store_true',
                        help='Include outliers in training data')
    parser.add_argument('--dont_subsample_buildings', action='store_true',
                        help='Use all buildings instead of subsampling')
    parser.add_argument('--building_filter', type=str, default='',
                        help='Filter to specific building (for grid search)')
    parser.add_argument('--use_temperature_input', action='store_true', default=True,
                        help='Include temperature as input feature')
    
    # PINN architecture
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 32],
                        help='Hidden layer dimensions for PINN')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Physics loss flags
    parser.add_argument('--use_smoothness', action='store_true', default=False,
                        help='Use smoothness loss (thermal inertia)')
    parser.add_argument('--use_temporal', action='store_true', default=False,
                        help='Use temporal gradient loss (HVAC ramp rates)')
    parser.add_argument('--use_weather', action='store_true', default=False,
                        help='Use weather consistency loss')
    
    # Physics loss weights
    parser.add_argument('--lambda_smooth', type=float, default=0.1,
                        help='Weight for smoothness loss')
    parser.add_argument('--lambda_temporal', type=float, default=0.1,
                        help='Weight for temporal gradient loss')
    parser.add_argument('--lambda_weather', type=float, default=0.1,
                        help='Weight for weather consistency loss')
    
    # Temporal gradient constraint
    parser.add_argument('--max_gradient', type=float, default=0.3,
                        help='Maximum allowed fractional change per hour (for temporal loss)')
    
    # Output
    parser.add_argument('--results_path', type=str, default='results/pinn',
                        help='Path to save results')
    parser.add_argument('--variant_name', type=str, default='',
                        help='Variant name suffix for output files')
    
    args = parser.parse_args()
    
    # Create results directory
    results_path = Path(args.results_path)
    
    # Run training
    pinn_training(args, results_path)


if __name__ == '__main__':
    main()
