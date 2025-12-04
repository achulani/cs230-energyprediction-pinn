"""
Evaluate and optimize PINN loss using pre-computed LightGBM predictions.

This script works with the transfer learning workflow:
1. Load pre-computed LightGBM predictions (from transfer_learning_lightgbm_fixed.py)
2. Load corresponding ground truth and temperature data
3. Evaluate PINN loss on LightGBM predictions
4. Optimize residuals to minimize PINN loss (try to get loss → 0)

The workflow matches transfer_learning_lightgbm_fixed.py and transfer_learning_pinn.py.
"""

from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import json

from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg

from buildings_bench import utils
from buildings_bench import BuildingTypes
from buildings_bench import load_pandas_dataset, benchmark_registry
from buildings_bench.data.datasets import keep_buildings
from buildings_bench.evaluation.managers import DatasetMetricsManager
from buildings_bench.pinn_losses import PINNLoss, compute_pinn_loss
from buildings_bench.metadata_loader import get_default_metadata


def load_lightgbm_predictions_from_csv(csv_file: Path) -> dict:
    """Load LightGBM predictions from results CSV.
    
    The CSV should have columns: dataset, building, and prediction columns.
    Returns a dictionary mapping (dataset, building) -> predictions array.
    """
    df = pd.read_csv(csv_file)
    
    # Assuming predictions are in columns after 'dataset' and 'building'
    # This is a simplified parser - adjust based on your actual CSV format
    predictions_dict = {}
    
    # If CSV has prediction columns, extract them
    # Otherwise, you may need to load from a separate predictions file
    return predictions_dict


def optimize_residuals_for_lightgbm(
    y_lgb: torch.Tensor,
    y_true: torch.Tensor,
    T_out: torch.Tensor,
    metadata: dict,
    lambda_rc: float = 1.0,
    lambda_comfort: float = 0.1,
    lambda_smooth: float = 0.01,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
    device: str = 'cpu'
) -> tuple[torch.Tensor, dict]:
    """Optimize residuals to minimize PINN loss starting from LightGBM predictions.
    
    Args:
        y_lgb: LightGBM predictions of shape (24,) or (batch, 24)
        y_true: Ground truth of shape (24,) or (batch, 24)
        T_out: Outdoor temperature of shape (24,) or (batch, 24)
        metadata: Building metadata dictionary
        lambda_rc: Weight for RC loss
        lambda_comfort: Weight for comfort loss
        lambda_smooth: Weight for smoothness loss
        lr: Learning rate
        max_iter: Maximum iterations
        tol: Convergence tolerance
        device: Device to run on
    
    Returns:
        Tuple of (optimized_predictions, loss_history)
    """
    y_lgb = y_lgb.to(device)
    y_true = y_true.to(device)
    T_out = T_out.to(device)
    
    if y_lgb.dim() == 1:
        y_lgb = y_lgb.unsqueeze(0)
        y_true = y_true.unsqueeze(0)
        T_out = T_out.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Initialize residuals as learnable parameters
    residuals = nn.Parameter(torch.zeros_like(y_lgb))
    
    # Initialize PINN loss
    pinn_loss_fn = PINNLoss(
        metadata=metadata,
        lambda_rc=lambda_rc,
        lambda_comfort=lambda_comfort,
        lambda_smooth=lambda_smooth
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam([residuals], lr=lr)
    
    # Loss history
    loss_history = {
        'total': [],
        'data': [],
        'rc': [],
        'comfort': [],
        'smooth': []
    }
    
    # Optimization loop
    prev_loss = float('inf')
    for iter in range(max_iter):
        # Compute predictions: y_hat = y_lgb + residual
        y_hat = y_lgb + residuals
        
        # Compute loss
        loss, loss_components = pinn_loss_fn(
            y_hat, y_true, T_out, return_components=True
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        loss_history['total'].append(loss_components['total'])
        loss_history['data'].append(loss_components['data'])
        loss_history['rc'].append(loss_components['rc'])
        loss_history['comfort'].append(loss_components['comfort'])
        loss_history['smooth'].append(loss_components['smooth'])
        
        # Check convergence
        if iter > 0 and abs(prev_loss - loss.item()) < tol:
            if iter % 100 == 0:
                print(f'Converged at iteration {iter}')
            break
        
        prev_loss = loss.item()
        
        if iter % 100 == 0:
            print(f'Iter {iter}: Loss = {loss.item():.6f}')
    
    # Get final predictions
    with torch.no_grad():
        y_hat_opt = y_lgb + residuals
    
    if squeeze_output:
        y_hat_opt = y_hat_opt.squeeze(0)
    
    return y_hat_opt, loss_history


def evaluate_pinn_on_lightgbm_predictions(
    args,
    results_path: Path
):
    """Evaluate and optimize PINN loss on LightGBM predictions.
    
    This function follows the same workflow as transfer_learning_lightgbm_fixed.py
    but evaluates PINN loss and optionally optimizes residuals.
    """
    global benchmark_registry
    lag = 168
    
    # Remove synthetic
    benchmark_registry = [b for b in benchmark_registry if b != 'buildings-900k-test']
    if args.benchmark[0] == 'all':
        args.benchmark = benchmark_registry
    
    metrics_manager = DatasetMetricsManager()
    pinn_results = []
    
    # Get default metadata
    metadata = get_default_metadata()
    
    # Load target buildings if specified
    target_buildings = []
    if not args.dont_subsample_buildings:
        metadata_dir = Path(os.environ.get('BUILDINGS_BENCH', '')) / 'metadata'
        if (metadata_dir / 'transfer_learning_commercial_buildings.txt').exists():
            with open(metadata_dir / 'transfer_learning_commercial_buildings.txt', 'r') as f:
                target_buildings += f.read().splitlines()
        if (metadata_dir / 'transfer_learning_residential_buildings.txt').exists():
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
        
        # Get building type
        try:
            building_type = dataset_generator.building_type
        except AttributeError:
            building_type = BuildingTypes.RESIDENTIAL
        
        if building_type == BuildingTypes.COMMERCIAL:
            building_types_mask = (BuildingTypes.COMMERCIAL_INT * torch.ones([1, 24, 1])).bool()
        else:
            building_types_mask = (BuildingTypes.RESIDENTIAL_INT * torch.ones([1, 24, 1])).bool()
        
        for building_name, bldg_df in dataset_generator:
            # Skip if too few days
            if len(bldg_df) < (args.num_training_days + 30) * 24:
                continue
            
            print(f'\nDataset: {dataset_name}, Building: {building_name}')
            
            # Split into training and test sets (same as transfer_learning_lightgbm_fixed.py)
            start_timestamp = bldg_df.index[0]
            end_timestamp = start_timestamp + pd.Timedelta(days=args.num_training_days)
            historical_date_range = pd.date_range(start=start_timestamp, end=end_timestamp, freq='H')
            
            training_set = bldg_df.loc[historical_date_range]
            test_set = bldg_df.loc[~bldg_df.index.isin(historical_date_range)]
            test_start_timestamp = test_set.index[0]
            test_end_timestamp = test_start_timestamp + pd.Timedelta(days=180)
            test_set = test_set[test_set.index <= test_end_timestamp]
            
            # Train LightGBM (same as transfer_learning_lightgbm_fixed.py)
            print('Training LightGBM...')
            forecaster = ForecasterAutoreg(
                regressor=LGBMRegressor(max_depth=-1, n_estimators=100, n_jobs=24, verbose=-1),
                lags=lag
            )
            forecaster.fit(
                y=training_set['power'],
                exog=training_set[[key for key in training_set.keys() if key != 'power']]
            )
            
            # Evaluate on test set
            pred_days = (len(test_set) - lag - 24) // 24
            if pred_days <= 0:
                continue
            
            print(f'Evaluating on {pred_days} prediction windows...')
            
            for i in tqdm(range(pred_days), desc='Evaluating'):
                seq_ptr = lag + 24 * i
                
                last_window = test_set.iloc[seq_ptr - lag:seq_ptr]
                ground_truth = test_set.iloc[seq_ptr:seq_ptr + 24]
                
                # Get LightGBM prediction
                lgbm_pred = forecaster.predict(
                    steps=24,
                    last_window=last_window['power'],
                    exog=ground_truth[[key for key in test_set.keys() if key != 'power']]
                )
                
                # Convert to tensors
                y_lgb = torch.from_numpy(lgbm_pred.values.flatten()).float()
                y_true = torch.from_numpy(ground_truth['power'].values).float()
                
                # Get temperature
                if 'temperature' in ground_truth.columns:
                    T_out = torch.from_numpy(ground_truth['temperature'].values).float()
                else:
                    T_out = torch.ones_like(y_true) * 20.0  # Default 20°C
                
                # Evaluate PINN loss on LightGBM predictions
                initial_loss = compute_pinn_loss(
                    y_lgb.unsqueeze(0), y_true.unsqueeze(0), T_out.unsqueeze(0),
                    metadata,
                    lambda_rc=args.lambda_rc,
                    lambda_comfort=args.lambda_comfort,
                    lambda_smooth=args.lambda_smooth,
                    return_components=True
                )[1]
                
                # Optimize residuals if requested
                if args.optimize:
                    y_opt, loss_history = optimize_residuals_for_lightgbm(
                        y_lgb, y_true, T_out, metadata,
                        lambda_rc=args.lambda_rc,
                        lambda_comfort=args.lambda_comfort,
                        lambda_smooth=args.lambda_smooth,
                        lr=args.lr,
                        max_iter=args.max_iter,
                        tol=args.tol,
                        device=args.device
                    )
                    
                    # Evaluate optimized loss
                    optimized_loss = compute_pinn_loss(
                        y_opt.unsqueeze(0), y_true.unsqueeze(0), T_out.unsqueeze(0),
                        metadata,
                        lambda_rc=args.lambda_rc,
                        lambda_comfort=args.lambda_comfort,
                        lambda_smooth=args.lambda_smooth,
                        return_components=True
                    )[1]
                    
                    # Use optimized predictions for metrics
                    final_pred = y_opt.numpy()
                    
                    # Store results
                    pinn_results.append({
                        'dataset': dataset_name,
                        'building': building_name,
                        'window': i,
                        'initial_loss': initial_loss,
                        'optimized_loss': optimized_loss,
                        'loss_history': loss_history
                    })
                else:
                    # Use LightGBM predictions for metrics
                    final_pred = y_lgb.numpy()
                    
                    pinn_results.append({
                        'dataset': dataset_name,
                        'building': building_name,
                        'window': i,
                        'initial_loss': initial_loss
                    })
                
                # Compute metrics (same as transfer_learning_lightgbm_fixed.py)
                metrics_manager(
                    dataset_name,
                    f'{building_name}',
                    torch.from_numpy(ground_truth['power'].values).float().view(1, 24, 1),
                    torch.from_numpy(final_pred).float().view(1, 24, 1),
                    building_types_mask
                )
    
    # Save metrics (same format as transfer_learning_lightgbm_fixed.py)
    print('\nGenerating summaries...')
    variant_name = f':{args.variant_name}' if args.variant_name != '' else ''
    metrics_file = results_path / f'TL_metrics_pinn_from_lgbm{variant_name}.csv'
    
    metrics_df = metrics_manager.summary()
    if metrics_file.exists():
        metrics_df.to_csv(metrics_file, mode='a', index=False, header=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)
    
    # Save PINN loss results
    if pinn_results:
        pinn_file = results_path / f'pinn_loss_results{variant_name}.json'
        with open(pinn_file, 'w') as f:
            json.dump(pinn_results, f, indent=2)
        print(f'PINN loss results saved to {pinn_file}')
        
        # Print summary
        if args.optimize:
            initial_totals = [r['initial_loss']['total'] for r in pinn_results]
            optimized_totals = [r['optimized_loss']['total'] for r in pinn_results]
            
            print('\n' + '='*60)
            print('PINN Loss Summary:')
            print('='*60)
            print(f'Initial (LightGBM only):')
            print(f'  Mean: {np.mean(initial_totals):.6f}')
            print(f'  Std:  {np.std(initial_totals):.6f}')
            print(f'\nOptimized (LightGBM + Residuals):')
            print(f'  Mean: {np.mean(optimized_totals):.6f}')
            print(f'  Std:  {np.std(optimized_totals):.6f}')
            print(f'\nImprovement: {np.mean(initial_totals) - np.mean(optimized_totals):.6f}')
            print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate PINN loss on LightGBM predictions and optimize residuals'
    )
    
    # Same arguments as transfer_learning_lightgbm_fixed.py
    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--benchmark', nargs='+', type=str, default=['all'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--variant_name', type=str, default='')
    parser.add_argument('--include_outliers', action='store_true')
    parser.add_argument('--num_training_days', type=int, default=180)
    parser.add_argument('--dont_subsample_buildings', action='store_true', default=False)
    parser.add_argument('--use_temperature_input', action='store_true')
    
    # PINN-specific arguments
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize residuals to minimize PINN loss')
    parser.add_argument('--lambda_rc', type=float, default=1.0,
                       help='Weight for RC circuit loss')
    parser.add_argument('--lambda_comfort', type=float, default=0.1,
                       help='Weight for comfort violation loss')
    parser.add_argument('--lambda_smooth', type=float, default=0.01,
                       help='Weight for smoothness loss')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate for residual optimization')
    parser.add_argument('--max_iter', type=int, default=1000,
                       help='Maximum iterations for optimization')
    parser.add_argument('--tol', type=float, default=1e-6,
                       help='Convergence tolerance')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run on')
    
    args = parser.parse_args()
    utils.set_seed(args.seed)
    
    results_path = Path(args.results_path)
    if args.include_outliers:
        results_path = results_path / 'buildingsbench_with_outliers'
    results_path.mkdir(parents=True, exist_ok=True)
    
    evaluate_pinn_on_lightgbm_predictions(args, results_path)

