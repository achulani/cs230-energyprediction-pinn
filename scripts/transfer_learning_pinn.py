"""
Train PINN Residual Networks for Building Energy Forecasting.

This script trains Physics-Informed Neural Network (PINN) residual networks
for each building independently, where a neural network corrects LightGBM predictions.

The workflow:
1. Train LightGBM on historical data to get initial predictions y_lgb
2. Train a residual network (PINN) to predict corrections r for each building
3. Final prediction: y_hat = y_lgb + r
4. Loss includes physics constraints (RC circuit, comfort, smoothness)

Note: Each building is trained independently from scratch (no weight transfer).
"""

from pathlib import Path
import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg

from buildings_bench import utils
from buildings_bench import load_pandas_dataset, benchmark_registry
from buildings_bench.data.datasets import keep_buildings, PandasTransformerDataset
from buildings_bench.evaluation.managers import DatasetMetricsManager
from buildings_bench import BuildingTypes
from buildings_bench.pinn_losses import PINNLoss, compute_pinn_loss
from buildings_bench.metadata_loader import load_building_metadata, get_default_metadata


SCRIPT_PATH = Path(os.path.realpath(__file__)).parent


class ResidualNetwork(nn.Module):
    """Simple residual network to correct LightGBM predictions."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 64], dropout: float = 0.1):
        """Initialize residual network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (single value for residual)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
        
        Returns:
            Residual predictions of shape (batch, seq_len, 1)
        """
        batch_size, seq_len, input_dim = x.shape
        # Use reshape instead of view to handle non-contiguous tensors safely
        x_flat = x.reshape(batch_size * seq_len, input_dim)
        residual_flat = self.network(x_flat)
        return residual_flat.reshape(batch_size, seq_len, 1)


def prepare_features(batch: dict, use_weather: bool = True) -> torch.Tensor:
    """Prepare input features for residual network.
    
    Args:
        batch: Dictionary with keys like 'load', 'day_of_year', 'hour_of_day', etc.
               All tensors should have shape (batch, seq_len, 1) where seq_len = context_len + pred_len
        use_weather: Whether to include weather features
    
    Returns:
        Feature tensor of shape (batch, seq_len, feature_dim)
    """
    features = []
    
    # Time features - shape (batch, seq_len, 1)
    features.append(batch['day_of_year'])
    features.append(batch['day_of_week'])
    features.append(batch['hour_of_day'])
    
    # Spatial features - shape (batch, seq_len, 1)
    features.append(batch['latitude'])
    features.append(batch['longitude'])
    
    # Building type - shape (batch, seq_len, 1)
    features.append(batch['building_type'].float())
    
    # Historical load - use full sequence for features
    # Note: For prediction window, we'll use context mean as a summary feature
    if 'load' in batch:
        # Use mean of context as a feature (repeated for all timesteps)
        context_len = batch['load'].shape[1] - 24  # Assuming last 24 is prediction window
        context_mean = batch['load'][:, :context_len].mean(dim=1, keepdim=True)  # (batch, 1, 1)
        context_mean = context_mean.expand(-1, batch['load'].shape[1], -1)  # (batch, seq_len, 1)
        features.append(context_mean)
    
    # Weather features - shape (batch, seq_len, 1)
    if use_weather and 'temperature' in batch:
        features.append(batch['temperature'])
    
    return torch.cat(features, dim=-1)


def aggregate_model_weights(weight_dicts: list) -> dict:
    """Aggregate multiple model state dicts by averaging weights.
    
    Args:
        weight_dicts: List of state dictionaries from multiple models
    
    Returns:
        Averaged state dictionary
    """
    if len(weight_dicts) == 0:
        raise ValueError("Cannot aggregate empty list of weights")
    
    if len(weight_dicts) == 1:
        return weight_dicts[0]
    
    # Initialize aggregated weights with first model
    aggregated = {}
    for key in weight_dicts[0].keys():
        aggregated[key] = weight_dicts[0][key].clone()
    
    # Average remaining models
    for weight_dict in weight_dicts[1:]:
        for key in aggregated.keys():
            if key in weight_dict:
                aggregated[key] += weight_dict[key]
    
    # Divide by number of models
    for key in aggregated.keys():
        aggregated[key] /= len(weight_dicts)
    
    return aggregated


def train_pinn_residual(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    lgbm_predictions_train: np.ndarray,
    lgbm_predictions_val: np.ndarray,
    metadata: dict,
    args,
    device: str,
    pretrained_weights: dict = None
):
    """Train residual network with PINN losses.
    
    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        lgbm_predictions_train: LightGBM predictions for training set
        lgbm_predictions_val: LightGBM predictions for validation set
        metadata: Building metadata dictionary
        args: Command line arguments
        device: Device to run on ('cuda' or 'cpu')
        pretrained_weights: Optional pretrained weights to initialize from
    """
    # Initialize residual network
    # Feature dimension: time (3) + spatial (2) + building_type (1) + load_context (context_len) + weather (1 if available)
    sample_batch = next(iter(train_dataloader))
    for k, v in sample_batch.items():
        sample_batch[k] = v.to(device)
    
    feature_dim = prepare_features(sample_batch, use_weather=args.use_temperature_input).shape[-1]
    
    residual_net = ResidualNetwork(
        input_dim=feature_dim,
        hidden_dims=[64, 64],
        dropout=0.1
    ).to(device)
    
    # Load pretrained weights if provided
    if pretrained_weights is not None:
        try:
            # Try to load all matching weights
            residual_net.load_state_dict(pretrained_weights, strict=False)
            print("Loaded pretrained weights (some layers may not match)")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Training from scratch instead")
    
    # Initialize PINN loss
    pinn_loss_fn = PINNLoss(
        metadata=metadata,
        lambda_rc=args.lambda_rc,
        lambda_comfort=args.lambda_comfort,
        lambda_smooth=args.lambda_smooth,
        use_rc_loss=args.use_rc_loss,
        use_comfort_loss=args.use_comfort_loss,
        use_smooth_loss=args.use_smooth_loss
    ).to(device)
    
    # Optimizer - use fine-tuning LR if pretrained weights were loaded
    lr = args.finetune_lr if (pretrained_weights is not None and args.finetune_lr > 0) else args.lr
    optimizer = torch.optim.AdamW(residual_net.parameters(), lr=lr)
    if pretrained_weights is not None and args.finetune_lr > 0:
        print(f"Using fine-tuning learning rate: {lr} (pretrained model)")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    loss_history = {
        'epoch': [],
        'train': {
            'total': [],
            'data': [],
            'rc': [],
            'comfort': [],
            'smooth': []
        },
        'val': {
            'total': [],
            'data': [],
            'rc': [],
            'comfort': [],
            'smooth': []
        },
        'metrics': {
            'train_mse': [],
            'train_mae': [],
            'val_mse': [],
            'val_mae': [],
            'residual_mean': [],
            'residual_std': [],
            'grad_norm': []
        }
    }
    
    for epoch in range(args.max_epochs):
        residual_net.train()
        train_losses = []
        train_loss_components = {'data': [], 'rc': [], 'comfort': [], 'smooth': []}
        train_mses = []
        train_maes = []
        residual_values = []
        grad_norms = []
        
        lgbm_idx = 0
        for batch_idx, batch in enumerate(train_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            # Prepare features
            features = prepare_features(batch, use_weather=args.use_temperature_input)
            
            # Get LightGBM predictions for this batch
            batch_size = batch['load'].shape[0]
            y_lgb_batch = torch.from_numpy(
                lgbm_predictions_train[lgbm_idx:lgbm_idx+batch_size]
            ).float().to(device)
            lgbm_idx += batch_size
            
            # Get targets (prediction window)
            y_true = batch['load'][:, -24:]  # Last 24 hours
            
            # Get outdoor temperature
            if 'temperature' in batch:
                T_out = batch['temperature'][:, -24:]  # Last 24 hours
            else:
                # Use a default temperature if not available
                T_out = torch.ones_like(y_true) * 20.0  # 20°C default
            
            # Predict residual
            residual = residual_net(features[:, -24:])  # (batch, seq_len, 1)
            residual = residual.squeeze(-1)            # (batch, T_resid) or (T_resid,)
            
            # Ensure both tensors have explicit batch dimension
            if y_lgb_batch.dim() == 1:
                y_lgb_batch = y_lgb_batch.unsqueeze(0)
            if residual.dim() == 1:
                residual = residual.unsqueeze(0)
            
            # Align LightGBM predictions, residual, and targets length defensively
            T = min(y_lgb_batch.shape[1], residual.shape[1], y_true.shape[1], T_out.shape[1])
            y_lgb_batch = y_lgb_batch[:, :T]
            residual = residual[:, :T]
            y_true = y_true[:, :T]
            T_out = T_out[:, :T]
            
            # Final prediction
            y_hat = y_lgb_batch + residual
            
            # Compute PINN loss
            loss, loss_components = pinn_loss_fn(
                y_hat, y_true.squeeze(-1), T_out.squeeze(-1),
                return_components=True
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Track gradient norm
            total_grad_norm = 0.0
            for p in residual_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            grad_norms.append(total_grad_norm)
            
            optimizer.step()
            
            # Track losses and metrics
            train_losses.append(loss.item())
            train_loss_components['data'].append(loss_components['data'])
            train_loss_components['rc'].append(loss_components['rc'])
            train_loss_components['comfort'].append(loss_components['comfort'])
            train_loss_components['smooth'].append(loss_components['smooth'])
            
            # Track prediction metrics
            mse = torch.mean((y_hat - y_true.squeeze(-1)) ** 2).item()
            mae = torch.mean(torch.abs(y_hat - y_true.squeeze(-1))).item()
            train_mses.append(mse)
            train_maes.append(mae)
            
            # Track residual statistics
            residual_values.extend(residual.detach().cpu().numpy().flatten().tolist())
        
        # Validation
        residual_net.eval()
        val_losses = []
        val_loss_components = {'data': [], 'rc': [], 'comfort': [], 'smooth': []}
        val_mses = []
        val_maes = []
        
        lgbm_idx = 0
        with torch.no_grad():
            for batch in val_dataloader:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                
                features = prepare_features(batch, use_weather=args.use_temperature_input)
                
                batch_size = batch['load'].shape[0]
                y_lgb_batch = torch.from_numpy(
                    lgbm_predictions_val[lgbm_idx:lgbm_idx+batch_size]
                ).float().to(device)
                lgbm_idx += batch_size
                
                y_true = batch['load'][:, -24:]
                
                if 'temperature' in batch:
                    T_out = batch['temperature'][:, -24:]
                else:
                    T_out = torch.ones_like(y_true) * 20.0
                
                residual = residual_net(features[:, -24:]).squeeze(-1)
                # Ensure both tensors have explicit batch dimension
                if y_lgb_batch.dim() == 1:
                    y_lgb_batch = y_lgb_batch.unsqueeze(0)
                if residual.dim() == 1:
                    residual = residual.unsqueeze(0)
                # Align lengths as in training loop
                T = min(y_lgb_batch.shape[1], residual.shape[1], y_true.shape[1], T_out.shape[1])
                y_lgb_batch = y_lgb_batch[:, :T]
                residual = residual[:, :T]
                y_true = y_true[:, :T]
                T_out = T_out[:, :T]
                y_hat = y_lgb_batch + residual
                
                loss, loss_components = pinn_loss_fn(
                    y_hat, y_true.squeeze(-1), T_out.squeeze(-1),
                    return_components=True
                )
                val_losses.append(loss.item())
                val_loss_components['data'].append(loss_components['data'])
                val_loss_components['rc'].append(loss_components['rc'])
                val_loss_components['comfort'].append(loss_components['comfort'])
                val_loss_components['smooth'].append(loss_components['smooth'])
                
                # Track prediction metrics
                mse = torch.mean((y_hat - y_true.squeeze(-1)) ** 2).item()
                mae = torch.mean(torch.abs(y_hat - y_true.squeeze(-1))).item()
                val_mses.append(mse)
                val_maes.append(mae)
        
        # Compute averages
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Record loss history
        loss_history['epoch'].append(epoch)
        loss_history['train']['total'].append(avg_train_loss)
        loss_history['train']['data'].append(np.mean(train_loss_components['data']))
        loss_history['train']['rc'].append(np.mean(train_loss_components['rc']))
        loss_history['train']['comfort'].append(np.mean(train_loss_components['comfort']))
        loss_history['train']['smooth'].append(np.mean(train_loss_components['smooth']))
        
        loss_history['val']['total'].append(avg_val_loss)
        loss_history['val']['data'].append(np.mean(val_loss_components['data']))
        loss_history['val']['rc'].append(np.mean(val_loss_components['rc']))
        loss_history['val']['comfort'].append(np.mean(val_loss_components['comfort']))
        loss_history['val']['smooth'].append(np.mean(val_loss_components['smooth']))
        
        # Record metrics
        loss_history['metrics']['train_mse'].append(np.mean(train_mses))
        loss_history['metrics']['train_mae'].append(np.mean(train_maes))
        loss_history['metrics']['val_mse'].append(np.mean(val_mses))
        loss_history['metrics']['val_mae'].append(np.mean(val_maes))
        loss_history['metrics']['residual_mean'].append(np.mean(residual_values) if residual_values else 0.0)
        loss_history['metrics']['residual_std'].append(np.std(residual_values) if residual_values else 0.0)
        loss_history['metrics']['grad_norm'].append(np.mean(grad_norms) if grad_norms else 0.0)
        
        print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        print(f'  Components - Data: {np.mean(train_loss_components["data"]):.4f}, '
              f'RC: {np.mean(train_loss_components["rc"]):.4f}, '
              f'Comfort: {np.mean(train_loss_components["comfort"]):.4f}, '
              f'Smooth: {np.mean(train_loss_components["smooth"]):.4f}')
        print(f'  Metrics - Train MSE: {np.mean(train_mses):.4f}, Val MSE: {np.mean(val_mses):.4f}, '
              f'Grad Norm: {np.mean(grad_norms):.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(residual_net.state_dict(), 'best_pinn_residual.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    # Load best model
    residual_net.load_state_dict(torch.load('best_pinn_residual.pt'))
    return residual_net, loss_history, residual_net.state_dict()  # Return weights for aggregation


def train_and_evaluate_pinn_models(args, results_path: Path):
    """Train and evaluate PINN residual networks for each building.
    
    Supports two modes:
    1. Pretraining: Train on all buildings and aggregate weights
    2. Fine-tuning: Load pretrained weights and fine-tune on new buildings
    """
    global benchmark_registry
    device = args.device
    lag = 168
    
    # Remove synthetic
    benchmark_registry = [b for b in benchmark_registry if b != 'buildings-900k-test']
    if args.benchmark[0] == 'all':
        args.benchmark = benchmark_registry
    
    metrics_manager = DatasetMetricsManager()
    
    # Load pretrained weights if provided
    pretrained_weights = None
    if args.pretrained_weights:
        pretrained_path = Path(args.pretrained_weights)
        if pretrained_path.exists():
            pretrained_weights = torch.load(pretrained_path, map_location=device)
            print(f"Loaded pretrained weights from {pretrained_path}")
        else:
            print(f"Warning: Pretrained weights file not found: {pretrained_path}")
    
    target_buildings = []
    if not args.dont_subsample_buildings:
        metadata_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata')
        with open(metadata_dir / 'transfer_learning_commercial_buildings.txt', 'r') as f:
            target_buildings += f.read().splitlines()
        with open(metadata_dir / 'transfer_learning_residential_buildings.txt', 'r') as f:
            target_buildings += f.read().splitlines()
    
    # Collect weights for aggregation (pretraining mode)
    collected_weights = []
    
    for dataset_name in args.benchmark:
        dataset_generator = load_pandas_dataset(
            dataset_name,
            feature_set='transformer',
            include_outliers=args.include_outliers,
            weather_inputs=['temperature'] if args.use_temperature_input else None
        )
        
        if len(target_buildings) > 0:
            dataset_generator = keep_buildings(dataset_generator, target_buildings)
        
        for building_name, bldg_df in dataset_generator:
            if len(bldg_df) < (args.num_training_days + 30) * 24:
                print(f'{dataset_name} {building_name} has too few days {len(bldg_df)}')
                continue
            
            print(f'dataset {dataset_name} building {building_name}')
            
            metrics_manager.add_building_to_dataset_if_missing(dataset_name, building_name)
            
            # Split data
            start_timestamp = bldg_df.index[0]
            end_timestamp = start_timestamp + pd.Timedelta(days=args.num_training_days)
            historical_date_range = pd.date_range(start=start_timestamp, end=end_timestamp, freq='H')
            
            training_set = bldg_df.loc[historical_date_range]
            training_start_timestamp = training_set.index[0]
            training_end_timestamp = training_start_timestamp + pd.Timedelta(days=args.num_training_days - 30)
            
            train_date_range = pd.date_range(start=training_start_timestamp, end=training_end_timestamp, freq='H')
            training_set_ = training_set.loc[train_date_range]
            validation_set = training_set[~training_set.index.isin(train_date_range)]
            
            test_set = bldg_df.loc[~bldg_df.index.isin(historical_date_range)]
            test_start_timestamp = test_set.index[0]
            test_end_timestamp = test_start_timestamp + pd.Timedelta(days=180)
            test_set = test_set[test_set.index <= test_end_timestamp]
            
            print(f'Training set: {training_set_.index[0]} to {training_set_.index[-1]}')
            print(f'Test set: {test_set.index[0]} to {test_set.index[-1]}')
            
            # Load building metadata
            try:
                metadata = load_building_metadata(building_name)
            except:
                print(f'Warning: Could not load metadata for {building_name}, using defaults')
                metadata = get_default_metadata()
            
            # Train LightGBM
            print('Training LightGBM...')
            forecaster = ForecasterAutoreg(
                regressor=LGBMRegressor(max_depth=-1, n_estimators=100, n_jobs=24, verbose=-1),
                lags=lag
            )
            forecaster.fit(
                y=training_set_['load'],
                exog=training_set_[[key for key in training_set_.keys() if key != 'load']]
            )
            
            # Create PyTorch datasets
            # For transformer-style datasets, we optionally include temperature as
            # an additional input feature via the `weather_inputs` argument.
            weather_inputs = ['temperature'] if args.use_temperature_input else None
            train_dataset = PandasTransformerDataset(
                training_set_, sliding_window=24, weather_inputs=weather_inputs
            )
            val_dataset = PandasTransformerDataset(
                validation_set, sliding_window=24, weather_inputs=weather_inputs
            )
            test_dataset = PandasTransformerDataset(
                test_set, sliding_window=24, weather_inputs=weather_inputs
            )
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=360, shuffle=False)
            
            # Get LightGBM predictions for training/validation
            print('Generating LightGBM predictions...')
            lgbm_train_preds = []
            lgbm_val_preds = []
            
            for i in range(len(train_dataset)):
                seq_ptr = lag + 24 * i
                if seq_ptr + 24 > len(training_set_):
                    break
                last_window = training_set_.iloc[seq_ptr - lag:seq_ptr]
                pred = forecaster.predict(
                    steps=24,
                    last_window=last_window['load'],
                    exog=training_set_.iloc[seq_ptr:seq_ptr+24][[k for k in training_set_.keys() if k != 'load']]
                )
                lgbm_train_preds.append(pred.values)
            
            for i in range(len(val_dataset)):
                seq_ptr = lag + 24 * i
                if seq_ptr + 24 > len(validation_set):
                    break
                last_window = validation_set.iloc[seq_ptr - lag:seq_ptr]
                pred = forecaster.predict(
                    steps=24,
                    last_window=last_window['load'],
                    exog=validation_set.iloc[seq_ptr:seq_ptr+24][[k for k in validation_set.keys() if k != 'load']]
                )
                lgbm_val_preds.append(pred.values)
            
            lgbm_train_preds = np.concatenate(lgbm_train_preds, axis=0)
            lgbm_val_preds = np.concatenate(lgbm_val_preds, axis=0)
            
            # Train PINN residual network
            print('Training PINN residual network...')
            residual_net, loss_history, trained_weights = train_pinn_residual(
                train_loader, val_loader,
                lgbm_train_preds, lgbm_val_preds,
                metadata, args, device,
                pretrained_weights=pretrained_weights
            )
            
            # Collect weights for aggregation (if in pretraining mode)
            if args.pretrain_mode:
                collected_weights.append(trained_weights)
                # Save individual building weights
                exp_suffix = f'_{args.experiment_name}' if args.experiment_name else ''
                building_weight_file = results_path / f'weights_{dataset_name}_{building_name}{exp_suffix}.pt'
                torch.save(trained_weights, building_weight_file)
            
            # Save loss history
            exp_suffix = f'_{args.experiment_name}' if args.experiment_name else ''
            loss_file = results_path / f'loss_history_{dataset_name}_{building_name}{exp_suffix}.json'
            with open(loss_file, 'w') as f:
                json.dump(loss_history, f, indent=2)
            print(f'Saved loss history to {loss_file}')
            
            # Evaluate on test set
            print('Evaluating on test set...')
            residual_net.eval()
            
            pred_days = (len(test_set) - lag - 24) // 24
            for i in range(pred_days):
                seq_ptr = lag + 24 * i
                
                last_window = test_set.iloc[seq_ptr - lag:seq_ptr]
                ground_truth = test_set.iloc[seq_ptr:seq_ptr + 24]
                
                # LightGBM prediction
                lgbm_pred = forecaster.predict(
                    steps=24,
                    last_window=last_window['load'],
                    exog=ground_truth[[key for key in test_set.keys() if key != 'load']]
                )
                
                # Get batch for residual network
                batch = test_dataset[i]
                for k, v in batch.items():
                    batch[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                
                features = prepare_features(batch, use_weather=args.use_temperature_input)
                residual = (
                    residual_net(features[:, -24:])
                    .squeeze(-1)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                
                # Final prediction
                final_pred = lgbm_pred.values.flatten() + residual
                
                # Compute metrics
                # Determine building type from dataset
                try:
                    building_type = dataset_generator.building_type
                except AttributeError:
                    building_type = BuildingTypes.RESIDENTIAL
                
                if building_type == BuildingTypes.COMMERCIAL:
                    building_types_mask = (BuildingTypes.COMMERCIAL_INT * torch.ones([1, 24, 1])).bool()
                else:
                    building_types_mask = (BuildingTypes.RESIDENTIAL_INT * torch.ones([1, 24, 1])).bool()
                
                metrics_manager(
                    dataset_name,
                    building_name,
                    torch.from_numpy(ground_truth['load'].values).float().view(1, 24, 1),
                    torch.from_numpy(final_pred).float().view(1, 24, 1),
                    building_types_mask
                )
    
    # Save results
    print('Generating summaries...')
    variant_name = f':{args.variant_name}' if args.variant_name != '' else ''
    exp_suffix = f'_{args.experiment_name}' if args.experiment_name else ''
    metrics_file = results_path / f'TL_metrics_pinn{variant_name}{exp_suffix}.csv'
    
    metrics_df = metrics_manager.summary()
    if metrics_file.exists():
        metrics_df.to_csv(metrics_file, mode='a', index=False, header=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)
    
    # Aggregate weights if in pretraining mode
    if args.pretrain_mode and len(collected_weights) > 0:
        print(f"\nAggregating weights from {len(collected_weights)} buildings...")
        aggregated_weights = aggregate_model_weights(collected_weights)
        aggregated_path = results_path / 'pretrained_pinn_weights.pt'
        torch.save(aggregated_weights, aggregated_path)
        print(f"Saved aggregated pretrained weights to {aggregated_path}")
        print(f"You can use these weights for fine-tuning new buildings with --pretrained_weights {aggregated_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PINN Residual Networks for Building Energy Forecasting')
    
    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--benchmark', nargs='+', type=str, default=['all'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--variant_name', type=str, default='')
    parser.add_argument('--include_outliers', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Training data configuration
    parser.add_argument('--num_training_days', type=int, default=180)
    parser.add_argument('--dont_subsample_buildings', action='store_true', default=False)
    parser.add_argument('--use_temperature_input', action='store_true')
    
    # PINN loss weights
    parser.add_argument('--lambda_rc', type=float, default=1.0, help='Weight for RC circuit loss')
    parser.add_argument('--lambda_comfort', type=float, default=0.1, help='Weight for comfort loss')
    parser.add_argument('--lambda_smooth', type=float, default=0.01, help='Weight for smoothness loss')
    
    # PINN loss component flags
    parser.add_argument('--use_rc_loss', action='store_true', default=True, help='Enable RC circuit loss')
    parser.add_argument('--no_rc_loss', dest='use_rc_loss', action='store_false', help='Disable RC circuit loss')
    parser.add_argument('--use_comfort_loss', action='store_true', default=True, help='Enable comfort loss')
    parser.add_argument('--no_comfort_loss', dest='use_comfort_loss', action='store_false', help='Disable comfort loss')
    parser.add_argument('--use_smooth_loss', action='store_true', default=True, help='Enable smoothness loss')
    parser.add_argument('--no_smooth_loss', dest='use_smooth_loss', action='store_false', help='Disable smoothness loss')
    
    # Experiment configuration
    parser.add_argument('--experiment_name', type=str, default='', help='Name for this experiment (used in output filenames)')
    
    # Training hyperparameters
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training from scratch')
    parser.add_argument('--finetune_lr', type=float, default=1e-5, help='Learning rate for fine-tuning (when using pretrained weights). Set to 0 to use --lr')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=5)
    
    # Transfer learning options
    parser.add_argument('--pretrain_mode', action='store_true', 
                       help='Pretraining mode: collect and aggregate weights from all buildings')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                       help='Path to pretrained weights file for fine-tuning new buildings')
    
    args = parser.parse_args()
    utils.set_seed(args.seed)
    
    results_path = Path(args.results_path)
    if args.include_outliers:
        results_path = results_path / 'buildingsbench_with_outliers'
    results_path.mkdir(parents=True, exist_ok=True)
    
    train_and_evaluate_pinn_models(args, results_path)

