"""
Physics-Informed Neural Network (PINN) Loss Functions for Building Energy Prediction.

This module implements physics-constrained loss functions for residual learning,
where a neural network corrects LightGBM predictions to satisfy thermodynamic constraints.

The losses enforce:
1. RC circuit energy balance (thermal dynamics)
2. Comfort zone violations (indoor temperature bounds)
3. Smoothness regularization (temporal consistency)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


def extract_building_params(metadata: Dict) -> Tuple[float, float, float, float, float]:
    """Extract building physics parameters from metadata dictionary.
    
    Args:
        metadata: Dictionary containing building metadata with keys:
            - 'in.sqft': Building square footage
            - 'stat.average_dx_cooling_cop': Average cooling COP (optional)
            - 'in.tstat_clg_sp_f': Cooling setpoint in Fahrenheit (optional)
            - 'in.tstat_htg_sp_f': Heating setpoint in Fahrenheit (optional)
    
    Returns:
        Tuple of (C, R, hvac_eff, T_set, deltaT):
            - C: Thermal capacitance proxy (J/K), estimated from floor area
            - R: Envelope resistance proxy (K/W), estimated from floor area
            - hvac_eff: HVAC efficiency (COP)
            - T_set: Comfort setpoint in Celsius (mean of heating and cooling setpoints)
            - deltaT: Comfort band in Celsius
    """
    # Thermal capacitance proxy: scales with building size
    sqft = metadata.get("in.sqft", 2000.0)
    if np.isnan(sqft) or sqft <= 0:
        sqft = 2000.0  # Default to 2000 sqft
    
    C = 3.0e5 * sqft  # thermal capacitance proxy (J/K)
    
    # Envelope resistance proxy: inverse relationship with size
    R = 1.0 / (0.001 * sqft)  # envelope resistance proxy (K/W)
    
    # HVAC efficiency (COP)
    hvac_eff = metadata.get("stat.average_dx_cooling_cop", 3.0)
    if np.isnan(hvac_eff) or hvac_eff <= 0:
        hvac_eff = 3.0  # Default COP of 3.0
    
    # Comfort setpoint: mean of heating and cooling setpoints (convert Fahrenheit to Celsius)
    T_clg_f = metadata.get("in.tstat_clg_sp_f", 72.0)
    T_htg_f = metadata.get("in.tstat_htg_sp_f", 68.0)
    if np.isnan(T_clg_f):
        T_clg_f = 72.0
    if np.isnan(T_htg_f):
        T_htg_f = 68.0
    T_set_f = (T_clg_f + T_htg_f) / 2.0  # Mean of heating and cooling setpoints
    T_set = (T_set_f - 32.0) * 5.0 / 9.0  # Convert F → C
    
    # Comfort band
    deltaT = 2.0  # °C comfort band
    
    return C, R, hvac_eff, T_set, deltaT


def infer_temperature(
    y_hat: torch.Tensor,
    T_out: torch.Tensor,
    C: float,
    R: float,
    T0: float = 22.0,
    dt: float = 3600.0
) -> torch.Tensor:
    """Infer indoor temperature from predicted load using RC circuit dynamics.
    
    Uses the discrete RC circuit equation:
        C * (T_in(t+1) - T_in(t)) / dt = (T_out - T_in) / R + y_hat
    
    Args:
        y_hat: Predicted load (kW) of shape (T,) or (batch, T)
        T_out: Outdoor temperature (°C) of shape (T,) or (batch, T)
        C: Thermal capacitance (J/K)
        R: Envelope resistance (K/W)
        T0: Initial indoor temperature (°C). Default: 22.0
        dt: Time step in seconds. Default: 3600.0 (1 hour)
    
    Returns:
        T_hat: Inferred indoor temperature (°C) of same shape as y_hat
    """
    # Handle batch dimension
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(0)
        T_out = T_out.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, seq_len = y_hat.shape
    device = y_hat.device
    
    # Initialize temperature sequence
    T = torch.zeros(batch_size, seq_len, device=device)
    T[:, 0] = T0
    
    # Convert load from kW to W for consistency with C and R
    # Assuming y_hat is in kW, convert to W
    y_hat_W = y_hat * 1000.0  # kW → W
    
    # Iteratively compute temperature
    for t in range(seq_len - 1):
        # Discrete RC equation: C * dT/dt = (T_out - T_in) / R + P
        # Rearranged: T_in(t+1) = T_in(t) + (dt/C) * ((T_out - T_in) / R + P)
        dT_dt = (T_out[:, t] - T[:, t]) / R + y_hat_W[:, t] / C
        T[:, t+1] = T[:, t] + (dt / C) * dT_dt
    
    if squeeze_output:
        T = T.squeeze(0)
    
    return T


def rc_loss(
    T_hat: torch.Tensor,
    y_hat: torch.Tensor,
    T_out: torch.Tensor,
    C: float,
    R: float,
    dt: float = 3600.0
) -> torch.Tensor:
    """RC circuit energy balance loss.
    
    Enforces the discrete RC equation:
        C * (T_hat(t+1) - T_hat(t)) / dt = (T_out - T_hat) / R + P
    
    Where P is power in W. The input y_hat is in kW and is converted to W.
    
    Args:
        T_hat: Inferred indoor temperature (°C) of shape (T,) or (batch, T)
        y_hat: Predicted load (kW) of shape (T,) or (batch, T)
        T_out: Outdoor temperature (°C) of shape (T,) or (batch, T)
        C: Thermal capacitance (J/K)
        R: Envelope resistance (K/W)
        dt: Time step in seconds. Default: 3600.0
    
    Returns:
        Scalar loss value
    """
    # Handle batch dimension
    if T_hat.dim() == 1:
        T_hat = T_hat.unsqueeze(0)
        y_hat = y_hat.unsqueeze(0)
        T_out = T_out.unsqueeze(0)
    
    # Compute temperature derivative
    dT = (T_hat[:, 1:] - T_hat[:, :-1]) / dt
    
    # Right-hand side of RC equation
    # Note: Following user's formulation: dT/dt = (T_out - T_in) / R + y_hat / C
    # If y_hat is in kW, we convert to W for consistency with C (J/K) and R (K/W)
    # However, the user's formulation suggests y_hat may already be in appropriate units
    # We'll convert kW to W to ensure dimensional consistency
    y_hat_W = y_hat[:, :-1] * 1000.0  # kW → W (if y_hat is in kW)
    rhs = (T_out[:, :-1] - T_hat[:, :-1]) / R + y_hat_W / C
    
    # Physics residual
    physics_residual = dT - rhs
    
    return torch.mean(physics_residual ** 2)


def comfort_loss(
    T_hat: torch.Tensor,
    T_set: float,
    deltaT: float = 2.0
) -> torch.Tensor:
    """Comfort violation loss.
    
    Penalizes predictions that imply indoor temperatures outside the comfort zone.
    The comfort zone is [T_set - deltaT, T_set + deltaT].
    
    Args:
        T_hat: Inferred indoor temperature (°C) of shape (T,) or (batch, T)
        T_set: Comfort setpoint (°C)
        deltaT: Comfort band half-width (°C). Default: 2.0
    
    Returns:
        Scalar loss value
    """
    # Compute violation: |T - T_set| - deltaT
    violation = torch.abs(T_hat - T_set) - deltaT
    
    # Only penalize violations (positive values)
    return torch.mean(torch.relu(violation))


def smoothness_loss(y_hat: torch.Tensor) -> torch.Tensor:
    """Thermal-inertia smoothness loss (second-order derivative).
    
    Enforces gradual changes in energy consumption consistent with thermal mass effects.
    Penalizes the second-order temporal derivative: (y_t - 2*y_{t-1} + y_{t-2})²
    
    Args:
        y_hat: Predicted load (kW) of shape (T,) or (batch, T)
    
    Returns:
        Scalar loss value
    """
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(0)
    
    # Compute second-order derivative: (y_t - 2*y_{t-1} + y_{t-2})
    # Need at least 3 timesteps
    if y_hat.shape[1] < 3:
        return torch.tensor(0.0, device=y_hat.device)
    
    d2y = y_hat[:, 2:] - 2.0 * y_hat[:, 1:-1] + y_hat[:, :-2]
    
    return torch.mean(d2y ** 2)


def focal_mse_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
    """Focal MSE loss that focuses on difficult-to-predict samples.
    
    Args:
        pred: Predictions
        target: Targets
        alpha: Weighting factor
        gamma: Focusing parameter (higher = more focus on hard examples)
    
    Returns:
        Focal MSE loss
    """
    mse = (pred - target) ** 2
    # Weight by error magnitude (harder examples get more weight)
    weights = (mse / (mse.mean() + 1e-8)) ** gamma
    return alpha * torch.mean(weights * mse)


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list = [0.1, 0.5, 0.9]) -> torch.Tensor:
    """Quantile loss for uncertainty estimation.
    
    Args:
        pred: Predictions (can be single value or quantile predictions)
        target: Targets
        quantiles: List of quantiles to predict
    
    Returns:
        Quantile loss
    """
    if pred.dim() == 2 and pred.shape[-1] == len(quantiles):
        # Multi-quantile prediction
        losses = []
        for i, q in enumerate(quantiles):
            error = target - pred[:, i]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss)
        return torch.mean(torch.stack(losses))
    else:
        # Single prediction - use median quantile
        error = target - pred.squeeze()
        return torch.mean(torch.max(0.5 * error, -0.5 * error))


class PINNLoss(nn.Module):
    """Combined PINN loss module for residual learning with advanced features.
    
    Implements the physics-informed loss functions for hybrid LightGBM + PINN forecasting:
    
    L_total = L_data + β_RC * L_RC + β_comfort * L_comfort + β_smooth * L_smooth
    
    Where:
    1. L_data: Data-fitting loss (MSE, Focal MSE, or Quantile)
    2. L_RC: Energy-balance loss from 1R1C thermal model
    3. L_comfort: Comfort-zone consistency loss
    4. L_smooth: Thermal-inertia smoothness loss (second-order derivative)
    
    Supports:
    - Adaptive loss weighting (learnable weights)
    - Focal loss for difficult samples
    - Quantile loss for uncertainty estimation
    - Loss balancing strategies
    
    Usage:
        loss_fn = PINNLoss(metadata, lambda_rc=1.0, lambda_comfort=0.1, lambda_smooth=0.01)
        loss = loss_fn(y_hat, y_true, T_out)
    """
    
    def __init__(
        self,
        metadata: Dict,
        lambda_rc: float = 1.0,
        lambda_comfort: float = 0.1,
        lambda_smooth: float = 0.01,
        dt: float = 3600.0,
        T0: float = 22.0,
        use_rc_loss: bool = True,
        use_comfort_loss: bool = True,
        use_smooth_loss: bool = True,
        use_adaptive_weights: bool = False,
        use_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        use_quantile_loss: bool = False,
        data_loss_type: str = 'mse'  # 'mse', 'focal_mse', 'quantile', 'huber'
    ):
        """Initialize PINN loss.
        
        Args:
            metadata: Building metadata dictionary
            lambda_rc: Weight for RC circuit loss. Default: 1.0
            lambda_comfort: Weight for comfort violation loss. Default: 0.1
            lambda_smooth: Weight for smoothness loss. Default: 0.01
            dt: Time step in seconds. Default: 3600.0 (1 hour)
            T0: Initial indoor temperature (°C). Default: 22.0
            use_rc_loss: Whether to include RC circuit loss. Default: True
            use_comfort_loss: Whether to include comfort loss. Default: True
            use_smooth_loss: Whether to include smoothness loss. Default: True
            use_adaptive_weights: Whether to use learnable adaptive weights. Default: False
            use_focal_loss: Whether to use focal loss for data fitting. Default: False
            focal_alpha: Focal loss alpha parameter. Default: 1.0
            focal_gamma: Focal loss gamma parameter. Default: 2.0
            use_quantile_loss: Whether to use quantile loss. Default: False
            data_loss_type: Type of data loss ('mse', 'focal_mse', 'quantile', 'huber'). Default: 'mse'
        """
        super().__init__()
        
        # Extract building parameters
        C, R, hvac_eff, T_set, deltaT = extract_building_params(metadata)
        
        # Register as buffers (not trainable parameters)
        self.register_buffer('C', torch.tensor(C, dtype=torch.float32))
        self.register_buffer('R', torch.tensor(R, dtype=torch.float32))
        self.register_buffer('T_set', torch.tensor(T_set, dtype=torch.float32))
        self.register_buffer('deltaT', torch.tensor(deltaT, dtype=torch.float32))
        self.register_buffer('dt', torch.tensor(dt, dtype=torch.float32))
        self.register_buffer('T0', torch.tensor(T0, dtype=torch.float32))
        
        # Adaptive loss weights (learnable)
        self.use_adaptive_weights = use_adaptive_weights
        if use_adaptive_weights:
            # Initialize with log-space for stability
            self.log_lambda_rc = nn.Parameter(torch.tensor(np.log(lambda_rc + 1e-8)))
            self.log_lambda_comfort = nn.Parameter(torch.tensor(np.log(lambda_comfort + 1e-8)))
            self.log_lambda_smooth = nn.Parameter(torch.tensor(np.log(lambda_smooth + 1e-8)))
        else:
            # Fixed weights
            self.register_buffer('lambda_rc', torch.tensor(lambda_rc, dtype=torch.float32))
            self.register_buffer('lambda_comfort', torch.tensor(lambda_comfort, dtype=torch.float32))
            self.register_buffer('lambda_smooth', torch.tensor(lambda_smooth, dtype=torch.float32))
        
        # Store as Python floats for use in functions
        self.C_val = C
        self.R_val = R
        self.T_set_val = T_set
        self.deltaT_val = deltaT
        self.dt_val = dt
        self.T0_val = T0
        
        # Store loss component flags
        self.use_rc_loss = use_rc_loss
        self.use_comfort_loss = use_comfort_loss
        self.use_smooth_loss = use_smooth_loss
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_quantile_loss = use_quantile_loss
        self.data_loss_type = data_loss_type
    
    def forward(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        T_out: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """Compute combined PINN loss.
        
        Args:
            y_hat: Final predictions (y_lgb + residual) of shape (batch, T) or (T,)
            y_true: Ground truth load (kW) of shape (batch, T) or (T,)
            T_out: Outdoor temperature (°C) of shape (batch, T) or (T,)
            return_components: If True, return individual loss components. Default: False
        
        Returns:
            Total loss (scalar), or tuple of (total_loss, loss_dict) if return_components=True
        """
        # Data loss with different formulations
        # Normalize by target variance to make loss scale-invariant across buildings
        # Use a more robust normalization to prevent explosion
        target_var = torch.var(y_true)
        target_mean_abs = torch.mean(torch.abs(y_true))
        
        # Use a combination of variance and mean for more stable normalization
        # This prevents explosion when variance is very small
        # Set minimum floor to prevent extreme normalization (prevents explosion)
        normalization_factor = torch.clamp(target_var, min=1e-4) + 0.1 * (target_mean_abs ** 2)
        normalization_factor = torch.clamp(normalization_factor, min=0.1)  # Minimum floor to prevent explosion
        
        # Compute raw data loss first
        if self.data_loss_type == 'focal_mse' or self.use_focal_loss:
            L_data_raw = focal_mse_loss(y_hat, y_true, self.focal_alpha, self.focal_gamma)
        elif self.data_loss_type == 'quantile' or self.use_quantile_loss:
            L_data_raw = quantile_loss(y_hat, y_true)
        elif self.data_loss_type == 'huber':
            L_data_raw = nn.functional.smooth_l1_loss(y_hat, y_true)
        else:
            # Standard MSE
            L_data_raw = torch.mean((y_hat - y_true) ** 2)
        
        # Normalize by robust normalization factor
        L_data = L_data_raw / normalization_factor
        
        # Infer indoor temperature
        T_hat = infer_temperature(
            y_hat, T_out, self.C_val, self.R_val, self.T0_val, self.dt_val
        )
        
        # RC circuit loss (only compute if enabled)
        L_rc = rc_loss(T_hat, y_hat, T_out, self.C_val, self.R_val, self.dt_val) if self.use_rc_loss else torch.tensor(0.0, device=y_hat.device)
        # Clip RC loss to prevent explosion
        if self.use_rc_loss:
            L_rc = torch.clamp(L_rc, max=1e6)  # Cap RC loss to prevent extreme values
        
        # Comfort violation loss (only compute if enabled)
        L_comfort = comfort_loss(T_hat, self.T_set_val, self.deltaT_val) if self.use_comfort_loss else torch.tensor(0.0, device=y_hat.device)
        
        # Smoothness loss (only compute if enabled)
        L_smooth = smoothness_loss(y_hat) if self.use_smooth_loss else torch.tensor(0.0, device=y_hat.device)
        
        # Normalize physics losses by their scale to prevent dominance
        # Use the same robust normalization factor for consistency
        if self.use_rc_loss:
            L_rc_norm = L_rc / normalization_factor if L_rc.item() > 0 else L_rc
            # Clip normalized RC loss more aggressively to prevent dominance
            # Even after normalization, RC loss can be huge and dominate training
            L_rc_norm = torch.clamp(L_rc_norm, max=100.0)  # Cap normalized RC loss to prevent dominance
        else:
            L_rc_norm = torch.tensor(0.0, device=y_hat.device)
        
        if self.use_comfort_loss:
            L_comfort_norm = L_comfort / normalization_factor if L_comfort.item() > 0 else L_comfort
        else:
            L_comfort_norm = torch.tensor(0.0, device=y_hat.device)
        
        if self.use_smooth_loss:
            L_smooth_norm = L_smooth / normalization_factor if L_smooth.item() > 0 else L_smooth
        else:
            L_smooth_norm = torch.tensor(0.0, device=y_hat.device)
        
        # Get loss weights (adaptive or fixed)
        if self.use_adaptive_weights:
            lambda_rc = torch.exp(self.log_lambda_rc)
            lambda_comfort = torch.exp(self.log_lambda_comfort)
            lambda_smooth = torch.exp(self.log_lambda_smooth)
        else:
            lambda_rc = self.lambda_rc
            lambda_comfort = self.lambda_comfort
            lambda_smooth = self.lambda_smooth
        
        # Combined loss with normalized components
        total_loss = L_data
        if self.use_rc_loss:
            total_loss = total_loss + lambda_rc * L_rc_norm
        if self.use_comfort_loss:
            total_loss = total_loss + lambda_comfort * L_comfort_norm
        if self.use_smooth_loss:
            total_loss = total_loss + lambda_smooth * L_smooth_norm
        
        # Check for NaN/Inf and clip to prevent explosion
        if not torch.isfinite(total_loss):
            # If loss is NaN/Inf, use a large but finite value to prevent training crash
            total_loss = torch.clamp(total_loss, min=-1e8, max=1e8)
            if torch.isnan(total_loss):
                total_loss = torch.tensor(1e6, device=y_hat.device, dtype=y_hat.dtype)
        else:
            # Clip total loss to prevent extreme values
            total_loss = torch.clamp(total_loss, max=1e8)
        
        if return_components:
            # Return both normalized (for training) and raw (for interpretation) losses
            loss_dict = {
                'data': L_data.item(),  # Normalized
                'data_raw': L_data_raw.item(),  # Raw MSE
                'rc': L_rc_norm.item() if self.use_rc_loss else 0.0,  # Normalized
                'rc_raw': L_rc.item() if self.use_rc_loss else 0.0,  # Raw
                'comfort': L_comfort_norm.item() if self.use_comfort_loss else 0.0,  # Normalized
                'comfort_raw': L_comfort.item() if self.use_comfort_loss else 0.0,  # Raw
                'smooth': L_smooth_norm.item() if self.use_smooth_loss else 0.0,  # Normalized
                'smooth_raw': L_smooth.item() if self.use_smooth_loss else 0.0,  # Raw
                'total': total_loss.item()  # Total normalized loss
            }
            if self.use_adaptive_weights:
                loss_dict['lambda_rc'] = lambda_rc.item()
                loss_dict['lambda_comfort'] = lambda_comfort.item()
                loss_dict['lambda_smooth'] = lambda_smooth.item()
            return total_loss, loss_dict
        
        return total_loss


def compute_pinn_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    T_out: torch.Tensor,
    metadata: Dict,
    lambda_rc: float = 1.0,
    lambda_comfort: float = 0.1,
    lambda_smooth: float = 0.01,
    dt: float = 3600.0,
    T0: float = 22.0,
    return_components: bool = False
) -> torch.Tensor:
    """Convenience function to compute PINN loss without creating a module.
    
    Args:
        y_hat: Final predictions (y_lgb + residual) of shape (batch, T) or (T,)
        y_true: Ground truth load (kW) of shape (batch, T) or (T,)
        T_out: Outdoor temperature (°C) of shape (batch, T) or (T,)
        metadata: Building metadata dictionary
        lambda_rc: Weight for RC circuit loss. Default: 1.0
        lambda_comfort: Weight for comfort violation loss. Default: 0.1
        lambda_smooth: Weight for smoothness loss. Default: 0.01
        dt: Time step in seconds. Default: 3600.0
        T0: Initial indoor temperature (°C). Default: 22.0
        return_components: If True, return individual loss components. Default: False
    
    Returns:
        Total loss (scalar), or tuple of (total_loss, loss_dict) if return_components=True
    """
    # Extract building parameters
    C, R, hvac_eff, T_set, deltaT = extract_building_params(metadata)
    
    # Data loss
    L_data = torch.mean((y_hat - y_true) ** 2)
    
    # Infer indoor temperature
    T_hat = infer_temperature(y_hat, T_out, C, R, T0, dt)
    
    # RC circuit loss
    L_rc = rc_loss(T_hat, y_hat, T_out, C, R, dt)
    
    # Comfort violation loss
    L_comfort = comfort_loss(T_hat, T_set, deltaT)
    
    # Smoothness loss
    L_smooth = smoothness_loss(y_hat)
    
    # Combined loss
    total_loss = (
        L_data +
        lambda_rc * L_rc +
        lambda_comfort * L_comfort +
        lambda_smooth * L_smooth
    )
    
    if return_components:
        loss_dict = {
            'data': L_data.item(),
            'rc': L_rc.item(),
            'comfort': L_comfort.item(),
            'smooth': L_smooth.item(),
            'total': total_loss.item()
        }
        return total_loss, loss_dict
    
    return total_loss

