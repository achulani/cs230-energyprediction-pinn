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
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg

from buildings_bench import utils
from buildings_bench import load_pandas_dataset, benchmark_registry
from buildings_bench.data.datasets import keep_buildings, PandasTransformerDataset
from buildings_bench.evaluation.managers import DatasetMetricsManager
from buildings_bench import BuildingTypes
from buildings_bench.pinn_losses import PINNLoss, compute_pinn_loss
from buildings_bench.metadata_loader import load_building_metadata, get_default_metadata
from buildings_bench.models.transformers import TimeSeriesSinusoidalPeriodicEmbedding
from buildings_bench.models.lstm_residual import LSTMBasedResidualNetwork
from buildings_bench.models.transformer_residual import TransformerBasedResidualNetwork
import numpy as np
import tomli


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
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming init for ReLU MLP to avoid vanishing/exploding scales."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
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


def prepare_features(
    batch: dict, 
    use_weather: bool = True,
    use_temporal_embeddings: bool = True,
    use_context_stats: bool = True,
    use_feature_interactions: bool = True,
    lgbm_predictions: Optional[torch.Tensor] = None,
    temporal_embedding_dim: int = 32
) -> torch.Tensor:
    """Prepare enhanced input features for residual network.
    
    Args:
        batch: Dictionary with keys like 'load', 'day_of_year', 'hour_of_day', etc.
               All tensors should have shape (batch, seq_len, 1) where seq_len = context_len + pred_len
        use_weather: Whether to include weather features
        use_temporal_embeddings: Whether to use sinusoidal temporal embeddings
        use_context_stats: Whether to include full context statistics
        use_feature_interactions: Whether to include feature interactions
        lgbm_predictions: Optional LightGBM predictions to include as features (batch, seq_len, 1)
        temporal_embedding_dim: Dimension for temporal embeddings
    
    Returns:
        Feature tensor of shape (batch, seq_len, feature_dim)
    """
    features = []
    batch_size, seq_len = batch['load'].shape[0], batch['load'].shape[1]
    context_len = seq_len - 24  # Assuming last 24 is prediction window
    device = batch['load'].device
    
    # Initialize temporal embedding modules if needed (lazy initialization)
    if use_temporal_embeddings and not hasattr(prepare_features, '_temporal_embeddings'):
        prepare_features._temporal_embeddings = {
            'day_of_year': TimeSeriesSinusoidalPeriodicEmbedding(temporal_embedding_dim).to(device),
            'day_of_week': TimeSeriesSinusoidalPeriodicEmbedding(temporal_embedding_dim).to(device),
            'hour_of_day': TimeSeriesSinusoidalPeriodicEmbedding(temporal_embedding_dim).to(device)
        }
    
    # Time features with optional embeddings
    if use_temporal_embeddings:
        # Normalize time features to [-1, 1] range for sinusoidal embedding
        # day_of_year: [0, 365] -> [-1, 1]
        day_of_year_norm = (batch['day_of_year'] / 182.5) - 1.0
        # day_of_week: [0, 6] -> [-1, 1]
        day_of_week_norm = (batch['day_of_week'] / 3.0) - 1.0
        # hour_of_day: [0, 23] -> [-1, 1]
        hour_of_day_norm = (batch['hour_of_day'] / 11.5) - 1.0
        
        features.append(prepare_features._temporal_embeddings['day_of_year'](day_of_year_norm))
        features.append(prepare_features._temporal_embeddings['day_of_week'](day_of_week_norm))
        features.append(prepare_features._temporal_embeddings['hour_of_day'](hour_of_day_norm))
    else:
        # Raw time features
        features.append(batch['day_of_year'])
        features.append(batch['day_of_week'])
        features.append(batch['hour_of_day'])
    
    # Spatial features - shape (batch, seq_len, 1)
    features.append(batch['latitude'])
    features.append(batch['longitude'])
    
    # Building type - shape (batch, seq_len, 1)
    features.append(batch['building_type'].float())
    
    # Enhanced historical load features
    if 'load' in batch:
        load_data = batch['load']  # (batch, seq_len, 1)
        
        if use_context_stats:
            # Context window statistics
            context_load = load_data[:, :context_len, :]  # (batch, context_len, 1)
            
            # Mean, std, min, max
            context_mean = context_load.mean(dim=1, keepdim=True)  # (batch, 1, 1)
            context_std = context_load.std(dim=1, keepdim=True)  # (batch, 1, 1)
            context_min = context_load.min(dim=1, keepdim=True)[0]  # (batch, 1, 1)
            context_max = context_load.max(dim=1, keepdim=True)[0]  # (batch, 1, 1)
            
            # Trend (linear regression slope over context)
            context_indices = torch.arange(context_len, device=device, dtype=torch.float32).view(1, -1, 1)
            context_flat = context_load.squeeze(-1)  # (batch, context_len)
            indices_flat = context_indices.squeeze(-1)  # (batch, context_len)
            # Simple trend: mean of differences
            context_trend = (context_flat[:, -1:] - context_flat[:, :1]) / (context_len - 1)  # (batch, 1)
            context_trend = context_trend.unsqueeze(-1)  # (batch, 1, 1)
            
            # Expand to full sequence length
            for stat in [context_mean, context_std, context_min, context_max, context_trend]:
                stat_expanded = stat.expand(-1, seq_len, -1)  # (batch, seq_len, 1)
                features.append(stat_expanded)
            
            # Recent history features (last 6h, 12h, 24h, 48h means)
            for window in [6, 12, 24, 48]:
                if context_len >= window:
                    recent_window = context_load[:, -window:, :]  # (batch, window, 1)
                    recent_mean = recent_window.mean(dim=1, keepdim=True)  # (batch, 1, 1)
                    recent_mean = recent_mean.expand(-1, seq_len, -1)  # (batch, seq_len, 1)
                    features.append(recent_mean)
        else:
            # Simple context mean (original behavior)
            context_mean = load_data[:, :context_len].mean(dim=1, keepdim=True)  # (batch, 1, 1)
            context_mean = context_mean.expand(-1, seq_len, -1)  # (batch, seq_len, 1)
            features.append(context_mean)
    
    # Weather features - shape (batch, seq_len, 1)
    if use_weather and 'temperature' in batch:
        features.append(batch['temperature'])
    
    # LightGBM predictions as input feature (if provided)
    if lgbm_predictions is not None:
        # Ensure same shape
        if lgbm_predictions.dim() == 2:
            lgbm_predictions = lgbm_predictions.unsqueeze(-1)  # (batch, seq_len, 1)
        elif lgbm_predictions.shape[1] != seq_len:
            # If predictions are only for prediction window, pad with zeros for context
            pred_only = lgbm_predictions.shape[1] == 24
            if pred_only:
                zeros = torch.zeros(batch_size, context_len, 1, device=device)
                lgbm_predictions = torch.cat([zeros, lgbm_predictions], dim=1)
        features.append(lgbm_predictions)
    
    # Feature interactions (if enabled)
    if use_feature_interactions:
        # Temperature × hour_of_day (captures diurnal temperature patterns)
        if use_weather and 'temperature' in batch:
            temp_hour = batch['temperature'] * batch['hour_of_day']
            features.append(temp_hour)
        
        # Load × temperature (captures HVAC load-temperature relationship)
        if 'load' in batch and use_weather and 'temperature' in batch:
            load_temp = load_data * batch['temperature']
            features.append(load_temp)
        
        # Hour × day_of_week (captures weekday vs weekend patterns)
        hour_dow = batch['hour_of_day'] * batch['day_of_week']
        features.append(hour_dow)
    
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
    pretrained_weights: dict = None,
    checkpoint_path: Path = Path('best_pinn_residual.pt')
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
    
    # Determine feature dimension with enhanced features
    use_temporal_emb = getattr(args, 'use_temporal_embeddings', True)
    use_context_stats = getattr(args, 'use_context_stats', True)
    use_feature_interactions = getattr(args, 'use_feature_interactions', True)
    feature_dim = prepare_features(
        sample_batch, 
        use_weather=args.use_temperature_input,
        use_temporal_embeddings=use_temporal_emb,
        use_context_stats=use_context_stats,
        use_feature_interactions=use_feature_interactions
    ).shape[-1]
    
    # Architecture selection
    architecture = getattr(args, 'architecture', 'mlp').lower()
    context_len = getattr(args, 'context_len', 168)
    pred_len = getattr(args, 'pred_len', 24)
    
    if architecture == 'lstm':
        # LSTM-based residual network
        residual_net = LSTMBasedResidualNetwork(
            input_dim=feature_dim,
            hidden_size=getattr(args, 'hidden_size', 128),
            num_layers=getattr(args, 'num_layers', 2),
            dropout=getattr(args, 'dropout', 0.1),
            use_attention=getattr(args, 'use_attention', True),
            use_residual=getattr(args, 'use_residual', True),
            use_layer_norm=getattr(args, 'use_layer_norm', True)
        ).to(device)
    elif architecture == 'transformer':
        # Transformer-based residual network
        residual_net = TransformerBasedResidualNetwork(
            input_dim=feature_dim,
            d_model=getattr(args, 'd_model', 128),
            nhead=getattr(args, 'nhead', 8),
            num_encoder_layers=getattr(args, 'num_encoder_layers', 2),
            num_decoder_layers=getattr(args, 'num_decoder_layers', 2),
            dim_feedforward=getattr(args, 'dim_feedforward', 256),
            dropout=getattr(args, 'dropout', 0.1),
            context_len=context_len,
            pred_len=pred_len
        ).to(device)
    else:
        # Default: Simple MLP (original ResidualNetwork)
        residual_net = ResidualNetwork(
            input_dim=feature_dim,
            hidden_dims=getattr(args, 'hidden_dims', [64, 64]),
            dropout=getattr(args, 'dropout', 0.1)
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
    
    # Initialize PINN loss with enhanced options
    pinn_loss_fn = PINNLoss(
        metadata=metadata,
        lambda_rc=args.lambda_rc,
        lambda_comfort=args.lambda_comfort,
        lambda_smooth=args.lambda_smooth,
        use_rc_loss=args.use_rc_loss,
        use_comfort_loss=args.use_comfort_loss,
        use_smooth_loss=args.use_smooth_loss,
        use_adaptive_weights=getattr(args, 'use_adaptive_weights', False),
        use_focal_loss=getattr(args, 'use_focal_loss', False),
        focal_alpha=getattr(args, 'focal_alpha', 1.0),
        focal_gamma=getattr(args, 'focal_gamma', 2.0),
        use_quantile_loss=getattr(args, 'use_quantile_loss', False),
        data_loss_type=getattr(args, 'data_loss_type', 'mse')
    ).to(device)
    
    # Optimizer - use fine-tuning LR if pretrained weights were loaded
    lr = args.finetune_lr if (pretrained_weights is not None and args.finetune_lr > 0) else args.lr
    weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 1e-5
    
    # Advanced optimizer configuration
    optimizer_type = getattr(args, 'optimizer', 'adamw').lower()
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            residual_net.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            residual_net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        # Default to AdamW
        optimizer = torch.optim.AdamW(
            residual_net.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    
    # Advanced learning rate scheduling
    scheduler = None
    scheduler_type = getattr(args, 'scheduler_type', 'cosine_restarts').lower()
    total_steps = args.max_epochs * len(train_dataloader)
    
    if scheduler_type == 'cosine_restarts':
        # Cosine annealing with warm restarts
        T_0 = getattr(args, 'scheduler_T_0', 10)  # Initial restart period
        T_mult = getattr(args, 'scheduler_T_mult', 2)  # Restart period multiplier
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=lr * 0.01
        )
    elif scheduler_type == 'onecycle':
        # OneCycleLR for faster convergence
        max_lr = lr * getattr(args, 'scheduler_max_lr_mult', 10.0)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=getattr(args, 'scheduler_pct_start', 0.3),
            anneal_strategy='cos'
        )
    elif scheduler_type == 'cosine':
        # Simple cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=lr * 0.01
        )
    elif scheduler_type == 'plateau':
        # ReduceLROnPlateau (original)
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        except TypeError:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )
    
    if pretrained_weights is not None and args.finetune_lr > 0:
        print(f"Using fine-tuning learning rate: {lr} (pretrained model)")
    
    # Mixed precision training scaler
    use_amp = getattr(args, 'use_mixed_precision', True) and device == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Gradient accumulation
    accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    
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
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in enumerate(train_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            # Prepare features with enhanced options
            use_temporal_emb = getattr(args, 'use_temporal_embeddings', True)
            use_context_stats = getattr(args, 'use_context_stats', True)
            use_feature_interactions = getattr(args, 'use_feature_interactions', True)
            
            # Get LightGBM predictions for this batch (for feature inclusion)
            batch_size = batch['load'].shape[0]
            y_lgb_batch_np = lgbm_predictions_train[lgbm_idx:lgbm_idx+batch_size]
            y_lgb_batch = torch.from_numpy(y_lgb_batch_np).float().to(device)
            
            # Prepare features (optionally include LGBM predictions)
            include_lgbm_as_feature = getattr(args, 'include_lgbm_as_feature', False)
            lgbm_feat = y_lgb_batch.unsqueeze(-1) if include_lgbm_as_feature else None
            features = prepare_features(
                batch, 
                use_weather=args.use_temperature_input,
                use_temporal_embeddings=use_temporal_emb,
                use_context_stats=use_context_stats,
                use_feature_interactions=use_feature_interactions,
                lgbm_predictions=lgbm_feat
            )
            lgbm_idx += batch_size
            
            # Get targets (prediction window)
            y_true = batch['load'][:, -24:]  # Last 24 hours
            
            # Get outdoor temperature
            if 'temperature' in batch:
                T_out = batch['temperature'][:, -24:]  # Last 24 hours
            else:
                # Use a default temperature if not available
                T_out = torch.ones_like(y_true) * 20.0  # 20°C default
            
            # Determine which features to use based on architecture
            # Transformer needs full sequence (context + prediction), LSTM/MLP can use full or just prediction window
            if architecture == 'transformer':
                # Transformer splits internally, needs full sequence
                residual_input = features  # (batch, seq_len, feature_dim)
            else:
                # LSTM and MLP: use full sequence for better context, but can also use just prediction window
                # Using full sequence gives better context for temporal models
                residual_input = features  # (batch, seq_len, feature_dim)
            
            # Mixed precision forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    # Predict residual
                    if architecture == 'transformer':
                        # Transformer expects full sequence and returns predictions for prediction window
                        residual = residual_net(residual_input, context_len=context_len)  # (batch, pred_len, 1)
                    else:
                        # LSTM/MLP: process full sequence, extract prediction window
                        residual_full = residual_net(residual_input)  # (batch, seq_len, 1)
                        residual = residual_full[:, -24:, :]  # Extract last 24 hours
                    
                    residual = residual.squeeze(-1)  # (batch, pred_len)
                    
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
                    loss = loss / accumulation_steps  # Scale loss for gradient accumulation
            else:
                # Predict residual
                if architecture == 'transformer':
                    # Transformer expects full sequence and returns predictions for prediction window
                    residual = residual_net(residual_input, context_len=context_len)  # (batch, pred_len, 1)
                else:
                    # LSTM/MLP: process full sequence, extract prediction window
                    residual_full = residual_net(residual_input)  # (batch, seq_len, 1)
                    residual = residual_full[:, -24:, :]  # Extract last 24 hours
                
                residual = residual.squeeze(-1)  # (batch, pred_len)
                
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
                
                # Check for NaN/Inf loss and skip if invalid
                if not torch.isfinite(loss):
                    print(f"Warning: Invalid loss (NaN/Inf) detected at batch {batch_idx}, skipping...")
                    print(f"  Loss value: {loss.item()}")
                    print(f"  Loss components: {loss_components}")
                    continue
                
                loss = loss / accumulation_steps  # Scale loss for gradient accumulation
            
            # Backward pass with mixed precision
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Track gradient norm before clipping (only on last accumulation step)
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    # Unscale gradients for norm computation
                    scaler.unscale_(optimizer)
                
                # Adaptive gradient clipping
                use_adaptive_clip = getattr(args, 'use_adaptive_grad_clip', False)
                if use_adaptive_clip:
                    # Clip based on parameter norms (not gradient norms)
                    clip_value = getattr(args, 'adaptive_clip_value', 0.01)
                    torch.nn.utils.clip_grad_norm_(
                        residual_net.parameters(), 
                        max_norm=float('inf'),
                        norm_type=2.0
                    )
                    # Then clip based on parameter norms
                    total_param_norm = torch.norm(
                        torch.stack([torch.norm(p.detach()) for p in residual_net.parameters()])
                    )
                    for p in residual_net.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(clip_value / max(total_param_norm, clip_value))
                else:
                    # Standard gradient clipping - use smaller value to prevent explosion
                    max_grad_norm = args.max_grad_norm if hasattr(args, 'max_grad_norm') else 1.0
                    # Cap at 0.5 to prevent explosion
                    clip_value = min(max_grad_norm, 0.5)
                    # Clip gradients and check for NaN/Inf
                    grad_norm = torch.nn.utils.clip_grad_norm_(residual_net.parameters(), clip_value)
                    
                    # Check for NaN/Inf gradients after clipping
                    grad_norm_tensor = torch.as_tensor(grad_norm)
                    if not torch.isfinite(grad_norm_tensor):
                        print(f"Warning: Invalid gradient norm (NaN/Inf) detected, skipping optimizer step...")
                        # Zero out gradients to prevent parameter corruption
                        optimizer.zero_grad()
                        continue
                    
                    # Warn if gradient norm is very large even after clipping
                    if grad_norm_tensor.item() > 10.0:
                        print(f"Warning: Large gradient norm after clipping: {grad_norm_tensor.item():.2f}")
                
                # Track gradient norm
                total_grad_norm = 0.0
                for p in residual_net.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** (1. / 2)
                grad_norms.append(total_grad_norm)
                
                # Optimizer step
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update learning rate scheduler (step-based schedulers)
                if scheduler is not None and scheduler_type in ['onecycle', 'cosine']:
                    scheduler.step()
            
            # Track losses and metrics (unscale for logging)
            unscaled_loss = loss.item() * accumulation_steps
            train_losses.append(unscaled_loss)
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
                
                # Handle different architectures
                if architecture == 'transformer':
                    residual = residual_net(features, context_len=context_len).squeeze(-1)
                else:
                    residual_full = residual_net(features)
                    residual = residual_full[:, -24:, :].squeeze(-1)
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
        
        # Learning rate scheduling (for epoch-based schedulers)
        if scheduler is not None and scheduler_type in ['plateau', 'cosine_restarts']:
            if scheduler_type == 'plateau':
                scheduler.step(avg_val_loss)
            elif scheduler_type == 'cosine_restarts':
                scheduler.step()
        
        # Early stopping
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(residual_net.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            # Only apply early stopping after warmup period
            if epoch >= args.min_epochs and patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch} (after {args.min_epochs} warmup epochs)')
                break
    
    # Load best model
    residual_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return residual_net, loss_history, residual_net.state_dict()  # Return weights for aggregation


def train_and_evaluate_pinn_models(args, results_path: Path, run_suffix: str):
    """Train and evaluate PINN residual networks for each building.
    
    Supports two modes:
    1. Pretraining: Train on all buildings and aggregate weights
    2. Fine-tuning: Load pretrained weights and fine-tune on new buildings
    """
    global benchmark_registry
    # Auto-detect device if not specified
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        # Validate device
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Using CPU instead.")
            device = 'cpu'
    print(f"Using device: {device}")
    lag = 168
    
    # Remove synthetic
    benchmark_registry = [b for b in benchmark_registry if b != 'buildings-900k-test']
    
    # Load target buildings first (before determining which datasets to use)
    target_buildings = []
    if not args.dont_subsample_buildings:
        metadata_dir = Path(os.environ.get('BUILDINGS_BENCH', ''), 'metadata')
        with open(metadata_dir / 'transfer_learning_commercial_buildings.txt', 'r') as f:
            target_buildings += f.read().splitlines()
        with open(metadata_dir / 'transfer_learning_residential_buildings.txt', 'r') as f:
            target_buildings += f.read().splitlines()
    
    # Determine which datasets to search
    # If target_buildings is specified, search the specified benchmarks (or all if 'all' is specified)
    # to find target buildings within those benchmarks
    if len(target_buildings) > 0:
        print(f"\n{'='*80}")
        print(f"Target buildings specified: {len(target_buildings)} buildings")
        if args.benchmark[0] == 'all':
            print(f"Searching ALL benchmark datasets to find all target buildings")
            datasets_to_search = benchmark_registry
        else:
            print(f"Searching specified benchmarks: {args.benchmark}")
            print(f"to find target buildings within those benchmarks")
            datasets_to_search = args.benchmark
        print(f"{'='*80}\n")
    elif args.benchmark[0] == 'all':
        datasets_to_search = benchmark_registry
    else:
        # Only use specified benchmark if no target_buildings
        datasets_to_search = args.benchmark
    
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
    
    # Collect weights for aggregation (pretraining mode)
    collected_weights = []
    
    # Track which buildings we've found to avoid duplicates
    found_buildings = set()
    
    for dataset_name in datasets_to_search:
        dataset_generator = load_pandas_dataset(
            dataset_name,
            feature_set='transformer',
            include_outliers=args.include_outliers,
            weather_inputs=['temperature'] if args.use_temperature_input else None
        )
        
        # Filter to target buildings if specified
        if len(target_buildings) > 0:
            dataset_generator = keep_buildings(dataset_generator, target_buildings)
            # Count how many target buildings are in this dataset
            buildings_in_dataset = set(dataset_generator.building_datasets.keys())
            new_buildings = buildings_in_dataset - found_buildings
            if len(new_buildings) > 0:
                print(f"Dataset {dataset_name}: {len(new_buildings)} new target buildings (total found: {len(found_buildings) + len(new_buildings)}/{len(target_buildings)})")
                found_buildings.update(new_buildings)
            elif len(buildings_in_dataset) > 0:
                print(f"Dataset {dataset_name}: {len(buildings_in_dataset)} target buildings (all already found)")
            else:
                print(f"Dataset {dataset_name}: 0 target buildings")
        
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
            checkpoint_path = results_path / f'best_pinn_residual_{dataset_name}_{building_name}_{run_suffix}.pt'
            residual_net, loss_history, trained_weights = train_pinn_residual(
                train_loader, val_loader,
                lgbm_train_preds, lgbm_val_preds,
                metadata, args, device,
                pretrained_weights=pretrained_weights,
                checkpoint_path=checkpoint_path
            )
            
            # Collect weights for aggregation (if in pretraining mode)
            if args.pretrain_mode:
                collected_weights.append(trained_weights)
                # Save individual building weights
                exp_suffix = ''
                if args.experiment_name:
                    exp_suffix += f'_{args.experiment_name}'
                exp_suffix += f'_{run_suffix}'
                building_weight_file = results_path / f'weights_{dataset_name}_{building_name}{exp_suffix}.pt'
                torch.save(trained_weights, building_weight_file)
            
            # Save loss history
            exp_suffix = ''
            if args.experiment_name:
                exp_suffix += f'_{args.experiment_name}'
            exp_suffix += f'_{run_suffix}'
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
                
                # Prepare features with enhanced options
                use_temporal_emb = getattr(args, 'use_temporal_embeddings', True)
                use_context_stats = getattr(args, 'use_context_stats', True)
                use_feature_interactions = getattr(args, 'use_feature_interactions', True)
                include_lgbm_as_feature = getattr(args, 'include_lgbm_as_feature', False)
                
                # Optionally include LGBM prediction as feature
                lgbm_feat = None
                if include_lgbm_as_feature:
                    lgbm_feat = torch.from_numpy(lgbm_pred.values.flatten()).float().unsqueeze(0).unsqueeze(-1).to(device)
                
                features = prepare_features(
                    batch,
                    use_weather=args.use_temperature_input,
                    use_temporal_embeddings=use_temporal_emb,
                    use_context_stats=use_context_stats,
                    use_feature_interactions=use_feature_interactions,
                    lgbm_predictions=lgbm_feat
                )
                
                # Handle different architectures
                architecture = getattr(args, 'architecture', 'mlp').lower()
                if architecture == 'transformer':
                    # Transformer needs full sequence (context + prediction)
                    context_len = getattr(args, 'context_len', 168)
                    residual = (
                        residual_net(features, context_len=context_len)
                        .squeeze(-1)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    # LSTM and MLP: process full sequence, extract prediction window
                    residual_full = residual_net(features)
                    residual = (
                        residual_full[:, -24:, :]
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
    
    # Print summary of buildings found
    if len(target_buildings) > 0:
        print(f"\n{'='*80}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Total target buildings: {len(target_buildings)}")
        print(f"Buildings found across all datasets: {len(found_buildings)}")
        if len(found_buildings) < len(target_buildings):
            missing = len(target_buildings) - len(found_buildings)
            print(f"Missing buildings (not in any benchmark dataset): {missing}")
        print(f"{'='*80}\n")
    
    # Save results
    print('Generating summaries...')
    variant_name = f':{args.variant_name}' if args.variant_name != '' else ''
    exp_suffix = ''
    if args.experiment_name:
        exp_suffix += f'_{args.experiment_name}'
    exp_suffix += f'_{run_suffix}'
    metrics_file = results_path / f'TL_metrics_pinn{variant_name}{exp_suffix}.csv'
    
    metrics_df = metrics_manager.summary()
    if metrics_file.exists():
        metrics_df.to_csv(metrics_file, mode='a', index=False, header=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)
    
    # Aggregate weights if in pretraining mode
    if args.pretrain_mode and len(collected_weights) > 0:
        print(f"\n{'='*80}")
        print(f"PRETRAINING SUMMARY")
        print(f"{'='*80}")
        if len(target_buildings) > 0:
            print(f"Total target buildings: {len(target_buildings)}")
            print(f"Buildings found and trained: {len(collected_weights)}")
            print(f"Coverage: {len(collected_weights)}/{len(target_buildings)} ({100*len(collected_weights)/len(target_buildings):.1f}%)")
        else:
            print(f"Buildings trained: {len(collected_weights)}")
        print(f"{'='*80}\n")
        print(f"Aggregating weights from {len(collected_weights)} buildings...")
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
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to use (cuda/cpu). If not specified, auto-detects based on availability.')
    
    # Training data configuration
    parser.add_argument('--num_training_days', type=int, default=180)
    parser.add_argument('--dont_subsample_buildings', action='store_true', default=False)
    parser.add_argument('--use_temperature_input', action='store_true')
    parser.add_argument('--run_suffix', type=str, default='', help='Unique suffix for outputs (default: timestamp)')
    
    # PINN loss weights
    parser.add_argument('--lambda_rc', type=float, default=0.1, help='Weight for RC circuit loss (default: 0.1, reduced from 1.0 to prevent RC loss dominance)')
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
    parser.add_argument('--min_epochs', type=int, default=0, help='Minimum epochs before early stopping can trigger (warmup period). Set to 0 to disable warmup.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training from scratch (increased from 1e-4 for better convergence)')
    parser.add_argument('--finetune_lr', type=float, default=1e-4, help='Learning rate for fine-tuning (when using pretrained weights). Set to 0 to use --lr')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping (default: 1.0)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for AdamW optimizer')
    parser.add_argument('--use_scheduler', action='store_true', help='Use ReduceLROnPlateau scheduler to reduce LR when validation loss plateaus')
    
    # Transfer learning options
    parser.add_argument('--pretrain_mode', action='store_true', 
                       help='Pretraining mode: collect and aggregate weights from all buildings')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                       help='Path to pretrained weights file for fine-tuning new buildings')
    
    # Architecture selection
    parser.add_argument('--architecture', type=str, default='mlp', choices=['mlp', 'lstm', 'transformer'],
                       help='Architecture type: mlp (simple MLP), lstm (LSTM-based), or transformer (Transformer-based)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to TOML config file (overrides individual arguments)')
    
    # Architecture-specific hyperparameters
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for LSTM/Transformer')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers for LSTM')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension for Transformer')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads for Transformer')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers for Transformer')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of decoder layers for Transformer')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Feed-forward dimension for Transformer')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64], help='Hidden dimensions for MLP')
    parser.add_argument('--use_attention', action='store_true', default=True, help='Use attention in LSTM')
    parser.add_argument('--use_residual', action='store_true', default=True, help='Use residual connections')
    parser.add_argument('--use_layer_norm', action='store_true', default=True, help='Use layer normalization')
    parser.add_argument('--context_len', type=int, default=168, help='Context window length')
    parser.add_argument('--pred_len', type=int, default=24, help='Prediction window length')
    
    # Advanced training options
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam'],
                       help='Optimizer type')
    parser.add_argument('--scheduler_type', type=str, default='cosine_restarts',
                       choices=['cosine_restarts', 'onecycle', 'cosine', 'plateau'],
                       help='Learning rate scheduler type')
    parser.add_argument('--scheduler_T_0', type=int, default=10, help='Initial restart period for cosine_restarts')
    parser.add_argument('--scheduler_T_mult', type=int, default=2, help='Restart period multiplier for cosine_restarts')
    parser.add_argument('--scheduler_max_lr_mult', type=float, default=10.0, help='Max LR multiplier for onecycle')
    parser.add_argument('--scheduler_pct_start', type=float, default=0.3, help='Percentage of warmup for onecycle')
    parser.add_argument('--use_mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps')
    parser.add_argument('--use_adaptive_grad_clip', action='store_true',
                       help='Use adaptive gradient clipping based on parameter norms')
    parser.add_argument('--adaptive_clip_value', type=float, default=0.01,
                       help='Adaptive gradient clipping value')
    
    # Feature engineering options
    parser.add_argument('--use_temporal_embeddings', action='store_true', default=True,
                       help='Use sinusoidal temporal embeddings')
    parser.add_argument('--use_context_stats', action='store_true', default=True,
                       help='Use full context statistics (mean, std, min, max, trend)')
    parser.add_argument('--use_feature_interactions', action='store_true', default=True,
                       help='Use feature interactions (temperature × hour, etc.)')
    parser.add_argument('--include_lgbm_as_feature', action='store_true',
                       help='Include LightGBM predictions as input features')
    
    # Loss function options
    parser.add_argument('--use_adaptive_weights', action='store_true',
                       help='Use learnable adaptive loss weights')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use focal loss for data fitting')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter')
    parser.add_argument('--use_quantile_loss', action='store_true',
                       help='Use quantile loss for uncertainty estimation')
    parser.add_argument('--data_loss_type', type=str, default='mse',
                       choices=['mse', 'focal_mse', 'quantile', 'huber'],
                       help='Type of data loss')
    
    args = parser.parse_args()
    
    # Generate a run-specific suffix for outputs to avoid collisions in concurrent runs
    if args.run_suffix and args.run_suffix.strip():
        run_suffix = args.run_suffix.strip()
    else:
        run_suffix = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Load config file if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'rb') as f:
                config = tomli.load(f)
            
            # Override args with config values
            if 'model' in config:
                for key, value in config['model'].items():
                    if hasattr(args, key):
                        setattr(args, key, value)
            
            if 'training' in config:
                for key, value in config['training'].items():
                    if hasattr(args, key):
                        setattr(args, key, value)
            
            if 'features' in config:
                for key, value in config['features'].items():
                    if hasattr(args, key):
                        setattr(args, key, value)
            
            if 'loss' in config:
                for key, value in config['loss'].items():
                    if hasattr(args, key):
                        setattr(args, key, value)
            
            print(f"Loaded configuration from {config_path}")
        else:
            print(f"Warning: Config file not found: {config_path}")
    
    utils.set_seed(args.seed)
    
    results_path = Path(args.results_path)
    if args.include_outliers:
        results_path = results_path / 'buildingsbench_with_outliers'
    results_path.mkdir(parents=True, exist_ok=True)
    
    train_and_evaluate_pinn_models(args, results_path, run_suffix)

