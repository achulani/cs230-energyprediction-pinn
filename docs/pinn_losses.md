# Physics-Informed Neural Network (PINN) Loss Functions

This module implements physics-constrained loss functions for building energy prediction using residual learning. The approach combines LightGBM predictions with a neural network residual correction that enforces thermodynamic constraints.

## Overview

The PINN loss module enforces three key physics constraints:

1. **RC Circuit Energy Balance**: Enforces thermal dynamics using a simplified RC circuit model
2. **Comfort Violation Loss**: Penalizes predictions that imply unrealistic indoor temperatures
3. **Smoothness Regularization**: Suppresses unrealistic hour-to-hour spikes in load predictions

## Architecture

```
LightGBM Prediction (y_lgb) + Residual Network (r) → Final Prediction (y_hat)
                                                          ↓
                                              PINN Loss (data + physics)
```

## Quick Start

### Basic Usage

```python
import torch
from buildings_bench.pinn_losses import PINNLoss, compute_pinn_loss
from buildings_bench.metadata_loader import load_building_metadata

# Load building metadata
metadata = load_building_metadata('building_12345')

# Option 1: Using PINNLoss module
loss_fn = PINNLoss(
    metadata=metadata,
    lambda_rc=1.0,      # Weight for RC circuit loss
    lambda_comfort=0.1, # Weight for comfort violation loss
    lambda_smooth=0.01  # Weight for smoothness loss
)

# Compute loss
y_hat = y_lgb + residual  # Final predictions
loss = loss_fn(y_hat, y_true, T_out)

# Option 2: Using convenience function
loss = compute_pinn_loss(
    y_hat, y_true, T_out, metadata,
    lambda_rc=1.0, lambda_comfort=0.1, lambda_smooth=0.01
)
```

### Training Example

```python
import torch
import torch.nn as nn
from buildings_bench.pinn_losses import PINNLoss

# Initialize model and loss
model = YourResidualNetwork(...)
loss_fn = PINNLoss(metadata, lambda_rc=1.0, lambda_comfort=0.1, lambda_smooth=0.01)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for batch in dataloader:
    # Get LightGBM predictions
    y_lgb = get_lgbm_predictions(batch)
    
    # Predict residual
    residual = model(batch['features'])
    
    # Final prediction
    y_hat = y_lgb + residual
    
    # Compute loss
    loss = loss_fn(y_hat, batch['y_true'], batch['temperature'])
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## API Reference

### `PINNLoss`

Main loss module class.

**Parameters:**
- `metadata` (dict): Building metadata dictionary with keys:
  - `'in.sqft'`: Building square footage
  - `'stat.average_dx_cooling_cop'`: Average cooling COP (optional)
  - `'in.tstat_clg_sp_f'`: Cooling setpoint in Fahrenheit (optional)
- `lambda_rc` (float): Weight for RC circuit loss. Default: 1.0
- `lambda_comfort` (float): Weight for comfort violation loss. Default: 0.1
- `lambda_smooth` (float): Weight for smoothness loss. Default: 0.01
- `dt` (float): Time step in seconds. Default: 3600.0 (1 hour)
- `T0` (float): Initial indoor temperature (°C). Default: 22.0

**Methods:**
- `forward(y_hat, y_true, T_out, return_components=False)`: Compute combined PINN loss

### `compute_pinn_loss`

Convenience function to compute PINN loss without creating a module.

**Parameters:**
- `y_hat` (torch.Tensor): Final predictions (y_lgb + residual)
- `y_true` (torch.Tensor): Ground truth load (kW)
- `T_out` (torch.Tensor): Outdoor temperature (°C)
- `metadata` (dict): Building metadata dictionary
- `lambda_rc`, `lambda_comfort`, `lambda_smooth` (float): Loss weights
- `dt` (float): Time step in seconds
- `T0` (float): Initial indoor temperature
- `return_components` (bool): If True, return individual loss components

### Individual Loss Functions

- `rc_loss(T_hat, y_hat, T_out, C, R, dt)`: RC circuit energy balance loss
- `comfort_loss(T_hat, T_set, deltaT)`: Comfort violation loss
- `smoothness_loss(y_hat)`: Smoothness regularization loss
- `infer_temperature(y_hat, T_out, C, R, T0, dt)`: Infer indoor temperature from load
- `extract_building_params(metadata)`: Extract physics parameters from metadata

## Physics Formulation

### RC Circuit Model

The thermal dynamics are modeled using a simplified RC circuit:

```
C * dT_in/dt = (T_out - T_in) / R + P
```

Where:
- `C`: Thermal capacitance (J/K)
- `R`: Envelope resistance (K/W)
- `T_in`: Indoor temperature (°C)
- `T_out`: Outdoor temperature (°C)
- `P`: Load (W)

### Discrete Formulation

```
C * (T_in(t+1) - T_in(t)) / dt = (T_out - T_in) / R + P
```

### Comfort Zone

The comfort zone is defined as:
```
[T_set - deltaT, T_set + deltaT]
```

Where `T_set` is the thermostat setpoint and `deltaT` is the comfort band (default: 2°C).

## Metadata Loading

The `metadata_loader` module provides utilities to load building metadata from Buildings-900K parquet files:

```python
from buildings_bench.metadata_loader import load_building_metadata

# Load metadata for a specific building
metadata = load_building_metadata('building_12345')

# If metadata is not available, use defaults
from buildings_bench.metadata_loader import get_default_metadata
metadata = get_default_metadata()
```

## Running the Example Script

The `scripts/transfer_learning_pinn.py` script demonstrates a complete training pipeline:

```bash
python scripts/transfer_learning_pinn.py \
    --benchmark bdg-2:panther \
    --use_temperature_input \
    --lambda_rc 1.0 \
    --lambda_comfort 0.1 \
    --lambda_smooth 0.01 \
    --max_epochs 25 \
    --lr 1e-4 \
    --batch_size 16
```

## Hyperparameter Tuning

Recommended ranges for loss weights:

- `lambda_rc`: 0.1 - 10.0 (start with 1.0)
- `lambda_comfort`: 0.01 - 1.0 (start with 0.1)
- `lambda_smooth`: 0.001 - 0.1 (start with 0.01)

The relative magnitudes matter more than absolute values. If physics constraints are too strong, reduce `lambda_rc` and `lambda_comfort`. If predictions are too noisy, increase `lambda_smooth`.

## Notes

- The physics parameters (C, R) are proxies derived from building square footage. They regularize structure but don't need to be perfectly accurate.
- The RC circuit model is a simplified approximation. The goal is to enforce thermodynamic structure, not perfect simulation.
- Temperature inference assumes initial conditions. For long sequences, consider using a rolling window approach.
- All temperatures should be in Celsius. Loads should be in kW.

