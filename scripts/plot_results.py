"""
Plot comparison of LightGBM baseline vs PINN hybrid model results.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import json
import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_results(lightgbm_file: Path, pinn_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load results from CSV files."""
    lgbm_df = pd.read_csv(lightgbm_file)
    pinn_df = pd.read_csv(pinn_file)
    return lgbm_df, pinn_df


def aggregate_metrics(df: pd.DataFrame, groupby_cols: list = None) -> pd.DataFrame:
    """Aggregate metrics by building or dataset."""
    if groupby_cols is None:
        groupby_cols = ['dataset', 'building_id']
    
    # Filter to overall metrics (not per-hour breakdowns)
    overall = df[df['metric_type'] == 'overall'].copy()
    
    # Pivot to get metrics as columns
    metrics_pivot = overall.pivot_table(
        index=groupby_cols,
        columns='metric',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    return metrics_pivot


def plot_metric_comparison(lgbm_df: pd.DataFrame, pinn_df: pd.DataFrame, 
                          metric: str = 'cvrmse', save_path: Path = None):
    """Plot comparison of a specific metric between LightGBM and PINN."""
    
    # Get overall metrics - try 'overall' first, then compute from hourly if needed
    lgbm_overall = lgbm_df[(lgbm_df['metric'] == metric) & 
                          (lgbm_df['metric_type'] == 'overall')].copy()
    pinn_overall = pinn_df[(pinn_df['metric'] == metric) & 
                          (pinn_df['metric_type'] == 'overall')].copy()
    
    # If no overall metrics, compute mean from hourly data
    if len(lgbm_overall) == 0:
        lgbm_hourly = lgbm_df[(lgbm_df['metric'].str.startswith(metric)) & 
                              (lgbm_df['metric_type'] == 'hour_of_day')].copy()
        if len(lgbm_hourly) > 0:
            lgbm_overall = lgbm_hourly.groupby(['dataset', 'building_id'])['value'].mean().reset_index()
            lgbm_overall['metric'] = metric
        else:
            print(f"No data found for LightGBM metric {metric}")
            return
    
    if len(pinn_overall) == 0:
        pinn_hourly = pinn_df[(pinn_df['metric'].str.startswith(metric)) & 
                              (pinn_df['metric_type'] == 'hour_of_day')].copy()
        if len(pinn_hourly) > 0:
            pinn_overall = pinn_hourly.groupby(['dataset', 'building_id'])['value'].mean().reset_index()
            pinn_overall['metric'] = metric
        else:
            print(f"No data found for PINN metric {metric}")
            return
    
    # Merge on building_id
    comparison = pd.merge(
        lgbm_overall[['dataset', 'building_id', 'value']].rename(columns={'value': 'LightGBM'}),
        pinn_overall[['dataset', 'building_id', 'value']].rename(columns={'value': 'PINN'}),
        on=['dataset', 'building_id'],
        how='inner'
    )
    
    if len(comparison) == 0:
        print(f"No matching buildings found for metric {metric}")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(comparison['LightGBM'], comparison['PINN'], alpha=0.6, s=50)
    
    # Diagonal line (y=x)
    min_val = min(comparison['LightGBM'].min(), comparison['PINN'].min())
    max_val = max(comparison['LightGBM'].max(), comparison['PINN'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (no improvement)')
    
    ax1.set_xlabel(f'LightGBM {metric.upper()}')
    ax1.set_ylabel(f'PINN {metric.upper()}')
    ax1.set_title(f'{metric.upper()} Comparison: LightGBM vs PINN')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Improvement histogram
    ax2 = axes[1]
    improvement = comparison['LightGBM'] - comparison['PINN']  # Positive = PINN better
    ax2.hist(improvement, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--', label='No improvement')
    ax2.axvline(improvement.mean(), color='g', linestyle='-', label=f'Mean: {improvement.mean():.4f}')
    ax2.set_xlabel(f'Improvement ({metric.upper()})')
    ax2.set_ylabel('Number of Buildings')
    ax2.set_title(f'Distribution of {metric.upper()} Improvement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print(f"\n{metric.upper()} Summary:")
    print(f"  LightGBM mean: {comparison['LightGBM'].mean():.4f} ± {comparison['LightGBM'].std():.4f}")
    print(f"  PINN mean:     {comparison['PINN'].mean():.4f} ± {comparison['PINN'].std():.4f}")
    print(f"  Mean improvement: {improvement.mean():.4f}")
    print(f"  Buildings improved: {(improvement > 0).sum()} / {len(comparison)} ({(improvement > 0).sum() / len(comparison) * 100:.1f}%)")


def plot_per_hour_metrics(lgbm_df: pd.DataFrame, pinn_df: pd.DataFrame,
                          metric: str = 'cvrmse', save_path: Path = None):
    """Plot metric vs hour of day."""
    
    # Filter to hour_of_day metrics
    lgbm_hourly = lgbm_df[(lgbm_df['metric'].str.startswith(metric)) & 
                          (lgbm_df['metric_type'] == 'hour_of_day')].copy()
    pinn_hourly = pinn_df[(pinn_df['metric'].str.startswith(metric)) & 
                          (pinn_df['metric_type'] == 'hour_of_day')].copy()
    
    # Extract hour from metric name (e.g., 'cvrmse_0' -> 0)
    lgbm_hourly['hour'] = lgbm_hourly['metric'].str.split('_').str[-1].astype(int)
    pinn_hourly['hour'] = pinn_hourly['metric'].str.split('_').str[-1].astype(int)
    
    # Aggregate across buildings
    lgbm_agg = lgbm_hourly.groupby('hour')['value'].mean().reset_index()
    pinn_agg = pinn_hourly.groupby('hour')['value'].mean().reset_index()
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(lgbm_agg['hour'], lgbm_agg['value'], 'o-', label='LightGBM', linewidth=2, markersize=6)
    plt.plot(pinn_agg['hour'], pinn_agg['value'], 's-', label='LightGBM→PINN', linewidth=2, markersize=6)
    plt.xlabel('Hour of Day')
    plt.ylabel(f'{metric.upper()}')
    plt.title(f'{metric.upper()} vs Hour of Day: LightGBM vs LightGBM→PINN (averaged across buildings)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_loss_curves_from_json(json_file: Path, save_path: Path = None, plot_components: bool = True):
    """Plot training and validation loss curves from a single loss history JSON file."""
    
    with open(json_file, 'r') as f:
        loss_history = json.load(f)
    
    epochs = loss_history['epoch']
    
    # Handle both old and new format
    if isinstance(loss_history['train'], list):
        # Old format
        train_losses = loss_history['train']
        val_losses = loss_history['val']
        plot_components = False
    else:
        # New format with components
        train_losses = loss_history['train']['total']
        val_losses = loss_history['val']['total']
    
    if plot_components and isinstance(loss_history['train'], dict):
        # Plot with components
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total loss
        ax = axes[0, 0]
        ax.plot(epochs, train_losses, 'o-', label='Training', linewidth=2, markersize=4)
        ax.plot(epochs, val_losses, 's-', label='Validation', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss components
        ax = axes[0, 1]
        ax.plot(epochs, loss_history['train']['data'], 'o-', label='Data (train)', linewidth=1.5, markersize=3)
        ax.plot(epochs, loss_history['train']['rc'], 's-', label='RC (train)', linewidth=1.5, markersize=3)
        ax.plot(epochs, loss_history['train']['comfort'], '^-', label='Comfort (train)', linewidth=1.5, markersize=3)
        ax.plot(epochs, loss_history['train']['smooth'], 'v-', label='Smooth (train)', linewidth=1.5, markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Component')
        ax.set_title('Training Loss Components')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Validation loss components
        ax = axes[1, 0]
        ax.plot(epochs, loss_history['val']['data'], 'o-', label='Data (val)', linewidth=1.5, markersize=3)
        ax.plot(epochs, loss_history['val']['rc'], 's-', label='RC (val)', linewidth=1.5, markersize=3)
        ax.plot(epochs, loss_history['val']['comfort'], '^-', label='Comfort (val)', linewidth=1.5, markersize=3)
        ax.plot(epochs, loss_history['val']['smooth'], 'v-', label='Smooth (val)', linewidth=1.5, markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Component')
        ax.set_title('Validation Loss Components')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Metrics
        ax = axes[1, 1]
        if 'metrics' in loss_history:
            ax.plot(epochs, loss_history['metrics']['train_mse'], 'o-', label='Train MSE', linewidth=1.5, markersize=3)
            ax.plot(epochs, loss_history['metrics']['val_mse'], 's-', label='Val MSE', linewidth=1.5, markersize=3)
            ax2 = ax.twinx()
            ax2.plot(epochs, loss_history['metrics']['grad_norm'], '^-', label='Grad Norm', 
                    color='red', linewidth=1.5, markersize=3)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE', color='blue')
            ax2.set_ylabel('Gradient Norm', color='red')
            ax.set_title('Metrics & Gradient Norm')
            ax.legend(loc='upper left', fontsize=8)
            ax2.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')
        
        plt.suptitle(f'Training Progress: {json_file.stem}', fontsize=14, y=0.995)
        plt.tight_layout()
    else:
        # Simple plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=4)
        plt.plot(epochs, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss Curves\n{json_file.stem}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_aggregated_losses_from_json(results_dir: Path, save_path: Path = None, run_suffix: str = None, exclude_suffix: str = None):
    """Plot mean and std of losses across all buildings from JSON files.
    
    Args:
        results_dir: Directory containing loss history JSON files
        save_path: Path to save the plot
        run_suffix: If provided, only include files with this suffix
        exclude_suffix: If provided, exclude files with this suffix
    """
    
    if run_suffix:
        # Filter to only files with the specified suffix
        loss_files = list(results_dir.glob(f'loss_history_*{run_suffix}.json'))
        print(f"Filtering to files with suffix: {run_suffix}")
    elif exclude_suffix:
        # Exclude files with the specified suffix
        all_files = list(results_dir.glob('loss_history_*.json'))
        loss_files = [f for f in all_files if exclude_suffix not in f.name]
        print(f"Excluding files with suffix: {exclude_suffix}")
    else:
        loss_files = list(results_dir.glob('loss_history_*.json'))
    
    if len(loss_files) == 0:
        print(f"No loss history files found in {results_dir}")
        return
    
    print(f"Found {len(loss_files)} loss history files")
    
    # Collect all loss histories
    all_train = []
    all_val = []
    all_train_data = []
    all_train_rc = []
    all_train_comfort = []
    all_train_smooth = []
    max_epochs = 0
    
    for loss_file in loss_files:
        with open(loss_file, 'r') as f:
            loss_history = json.load(f)
        
        epochs = loss_history['epoch']
        
        if isinstance(loss_history['train'], list):
            train_losses = loss_history['train']
            val_losses = loss_history['val']
        else:
            train_losses = loss_history['train']['total']
            val_losses = loss_history['val']['total']
            if 'data' in loss_history['train']:
                all_train_data.append(loss_history['train']['data'])
                all_train_rc.append(loss_history['train']['rc'])
                all_train_comfort.append(loss_history['train']['comfort'])
                all_train_smooth.append(loss_history['train']['smooth'])
        
        max_epochs = max(max_epochs, len(epochs))
        all_train.append(train_losses)
        all_val.append(val_losses)
    
    # Pad to same length
    for i in range(len(all_train)):
        if len(all_train[i]) < max_epochs:
            # Repeat last value
            last_train = all_train[i][-1]
            last_val = all_val[i][-1]
            all_train[i].extend([last_train] * (max_epochs - len(all_train[i])))
            all_val[i].extend([last_val] * (max_epochs - len(all_val[i])))
            if all_train_data:
                all_train_data[i].extend([all_train_data[i][-1]] * (max_epochs - len(all_train_data[i])))
                all_train_rc[i].extend([all_train_rc[i][-1]] * (max_epochs - len(all_train_rc[i])))
                all_train_comfort[i].extend([all_train_comfort[i][-1]] * (max_epochs - len(all_train_comfort[i])))
                all_train_smooth[i].extend([all_train_smooth[i][-1]] * (max_epochs - len(all_train_smooth[i])))
    
    # Convert to numpy arrays
    all_train = np.array(all_train)
    all_val = np.array(all_val)
    
    # Compute mean and std
    train_mean = np.mean(all_train, axis=0)
    train_std = np.std(all_train, axis=0)
    val_mean = np.mean(all_val, axis=0)
    val_std = np.std(all_val, axis=0)
    
    epochs = np.arange(max_epochs)
    
    # Plot
    if all_train_data:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total loss
        ax = axes[0, 0]
        ax.plot(epochs, train_mean, 'o-', label='Training Loss (mean)', linewidth=2, markersize=4)
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.3)
        ax.plot(epochs, val_mean, 's-', label='Validation Loss (mean)', linewidth=2, markersize=4)
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title(f'Total Loss (aggregated across {len(loss_files)} buildings)')
        ax.set_yscale('log')  # Use log scale to handle large differences in scale
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss components
        all_train_data = np.array(all_train_data)
        all_train_rc = np.array(all_train_rc)
        all_train_comfort = np.array(all_train_comfort)
        all_train_smooth = np.array(all_train_smooth)
        
        ax = axes[0, 1]
        ax.plot(epochs, np.mean(all_train_data, axis=0), 'o-', label='Data', linewidth=1.5, markersize=3)
        
        # Only plot RC loss if it's non-zero (filter out disabled losses)
        # Hide RC loss if run_suffix indicates RC loss was disabled
        mean_rc = np.mean(all_train_rc, axis=0)
        if run_suffix and 'no_physics_losses' in run_suffix:
            # Explicitly hide RC loss for no_physics_losses runs (but show comfort and smooth)
            pass  # Don't plot RC loss
        elif np.any(mean_rc > 1e-6):  # Only plot if RC loss is significant
            ax.plot(epochs, mean_rc, 's-', label='RC', linewidth=1.5, markersize=3)
        
        # Always show comfort and smooth losses (they're not disabled in no_physics_losses runs)
        mean_comfort = np.mean(all_train_comfort, axis=0)
        if np.any(mean_comfort > 1e-6):  # Only plot if comfort loss is significant
            ax.plot(epochs, mean_comfort, '^-', label='Comfort', linewidth=1.5, markersize=3)
        
        mean_smooth = np.mean(all_train_smooth, axis=0)
        if np.any(mean_smooth > 1e-6):  # Only plot if smooth loss is significant
            ax.plot(epochs, mean_smooth, 'v-', label='Smooth', linewidth=1.5, markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Component')
        ax.set_title('Training Loss Components (mean)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Individual building curves (sample)
        ax = axes[1, 0]
        for i in range(min(5, len(all_train))):
            ax.plot(epochs, all_train[i], alpha=0.3, linewidth=1)
        ax.plot(epochs, train_mean, 'o-', label='Mean', linewidth=2, markersize=4, color='black')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Individual Building Curves (sample)')
        ax.set_yscale('log')  # Use log scale to handle large differences in scale
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation curves
        ax = axes[1, 1]
        for i in range(min(5, len(all_val))):
            ax.plot(epochs, all_val[i], alpha=0.3, linewidth=1)
        ax.plot(epochs, val_mean, 's-', label='Mean', linewidth=2, markersize=4, color='black')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Individual Building Curves (sample)')
        ax.set_yscale('log')  # Use log scale to handle large differences in scale
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
    else:
        # Simple aggregated plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_mean, 'o-', label='Training Loss (mean)', linewidth=2, markersize=4)
        plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.3)
        plt.plot(epochs, val_mean, 's-', label='Validation Loss (mean)', linewidth=2, markersize=4)
        plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')  # Use log scale to handle large differences in scale
        plt.title(f'Training and Validation Loss (aggregated across {len(loss_files)} buildings)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved aggregated plot to {save_path}")
    else:
        plt.show()


def plot_building_type_comparison(lgbm_df: pd.DataFrame, pinn_df: pd.DataFrame,
                                  metric: str = 'cvrmse', save_path: Path = None):
    """Plot comparison by building type."""
    
    # Normalize dataset names to handle case inconsistencies
    def normalize_dataset_name(name):
        """Normalize dataset names for consistent matching."""
        name_lower = str(name).lower()
        # Handle BDG-2 variants
        if 'bdg-2' in name_lower or name_lower == 'bdg-2':
            return 'BDG-2'
        # Handle Electricity variants
        if name_lower == 'electricity':
            return 'Electricity'
        # Keep other names as-is but normalize case
        return str(name)
    
    # Create normalized copies
    lgbm_df_norm = lgbm_df.copy()
    pinn_df_norm = pinn_df.copy()
    lgbm_df_norm['dataset_norm'] = lgbm_df_norm['dataset'].apply(normalize_dataset_name)
    pinn_df_norm['dataset_norm'] = pinn_df_norm['dataset'].apply(normalize_dataset_name)
    
    # Get overall metrics - try 'overall' first, then compute from hourly if needed
    lgbm_overall = lgbm_df_norm[(lgbm_df_norm['metric'] == metric) & 
                          (lgbm_df_norm['metric_type'] == 'overall')].copy()
    pinn_overall = pinn_df_norm[(pinn_df_norm['metric'] == metric) & 
                          (pinn_df_norm['metric_type'] == 'overall')].copy()
    
    # If no overall metrics, compute mean from hourly data
    if len(lgbm_overall) == 0:
        lgbm_hourly = lgbm_df_norm[(lgbm_df_norm['metric'].str.startswith(metric)) & 
                              (lgbm_df_norm['metric_type'] == 'hour_of_day')].copy()
        if len(lgbm_hourly) > 0:
            lgbm_overall = lgbm_hourly.groupby(['dataset_norm', 'building_id', 'building_type'])['value'].mean().reset_index()
        else:
            print(f"No data found for LightGBM metric {metric}")
            return
    
    if len(pinn_overall) == 0:
        pinn_hourly = pinn_df_norm[(pinn_df_norm['metric'].str.startswith(metric)) & 
                              (pinn_df_norm['metric_type'] == 'hour_of_day')].copy()
        if len(pinn_hourly) > 0:
            pinn_overall = pinn_hourly.groupby(['dataset_norm', 'building_id', 'building_type'])['value'].mean().reset_index()
        else:
            print(f"No data found for PINN metric {metric}")
            return
    
    # Merge using normalized dataset names
    comparison = pd.merge(
        lgbm_overall[['dataset_norm', 'building_id', 'building_type', 'value']].rename(columns={'value': 'LightGBM', 'dataset_norm': 'dataset'}),
        pinn_overall[['dataset_norm', 'building_id', 'building_type', 'value']].rename(columns={'value': 'PINN', 'dataset_norm': 'dataset'}),
        on=['dataset', 'building_id', 'building_type'],
        how='inner'
    )
    
    if len(comparison) == 0:
        print(f"No matching buildings found for metric {metric}")
        return
    
    # Group by building type
    fig, ax = plt.subplots(figsize=(10, 6))
    
    building_types = comparison['building_type'].unique()
    x = np.arange(len(building_types))
    width = 0.35
    
    lgbm_means = [comparison[comparison['building_type'] == bt]['LightGBM'].mean() 
                  for bt in building_types]
    pinn_means = [comparison[comparison['building_type'] == bt]['PINN'].mean() 
                  for bt in building_types]
    
    ax.bar(x - width/2, lgbm_means, width, label='LightGBM', alpha=0.8)
    ax.bar(x + width/2, pinn_means, width, label='PINN', alpha=0.8)
    
    ax.set_xlabel('Building Type')
    ax.set_ylabel(f'{metric.upper()}')
    ax.set_title(f'{metric.upper()} by Building Type')
    ax.set_xticks(x)
    ax.set_xticklabels(building_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot comparison of LightGBM vs PINN results')
    parser.add_argument('--lightgbm_file', type=str, 
                       default='results/TL_metrics_lightgbm.csv',
                       help='Path to LightGBM results CSV')
    parser.add_argument('--pinn_file', type=str,
                       default='results/TL_metrics_pinn.csv',
                       help='Path to PINN results CSV')
    parser.add_argument('--output_dir', type=str, default='results/plots',
                       help='Directory to save plots')
    parser.add_argument('--metric', type=str, default='cvrmse',
                       choices=['cvrmse', 'mae', 'rmse', 'mape'],
                       help='Metric to plot')
    parser.add_argument('--plot_losses', action='store_true',
                       help='Plot training loss curves from JSON files')
    parser.add_argument('--loss_json', type=str, default=None,
                       help='Path to a specific loss history JSON file (if plotting single building)')
    parser.add_argument('--results_dir', type=str, default='results/',
                       help='Directory containing loss history JSON files (for aggregated plots)')
    parser.add_argument('--run_suffix', type=str, default='',
                       help='Suffix to add to plot filenames (e.g., from --run_suffix flag in training). If provided, only plots files with this suffix.')
    parser.add_argument('--exclude_suffix', type=str, default='',
                       help='Exclude files with this suffix from plotting')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss curves from JSON files
    if args.plot_losses:
        if args.loss_json:
            # Plot single building
            print(f"\nPlotting loss curves from {args.loss_json}...")
            plot_loss_curves_from_json(
                Path(args.loss_json),
                save_path=output_dir / f'{Path(args.loss_json).stem}_losses.png',
                plot_components=True
            )
        else:
            # Plot aggregated losses
            print(f"\nPlotting aggregated loss curves from {args.results_dir}...")
            plot_suffix = f'_{args.run_suffix}' if args.run_suffix else ''
            plot_aggregated_losses_from_json(
                Path(args.results_dir),
                save_path=output_dir / f'aggregated_training_losses{plot_suffix}.png',
                run_suffix=args.run_suffix if args.run_suffix else None,
                exclude_suffix=args.exclude_suffix if args.exclude_suffix else None
            )
        print(f"\nLoss plots saved to {output_dir}/")
        return
    
    # Load CSV data for metrics comparison
    lgbm_df = pd.read_csv(args.lightgbm_file)
    pinn_df = pd.read_csv(args.pinn_file)
    
    print(f"Loaded {len(lgbm_df)} LightGBM records and {len(pinn_df)} PINN records")
    
    # Add suffix to plot filenames if provided
    plot_suffix = f'_{args.run_suffix}' if args.run_suffix else ''
    
    # Plot 1: Overall comparison
    print(f"\nPlotting {args.metric.upper()} comparison...")
    plot_metric_comparison(
        lgbm_df, pinn_df, 
        metric=args.metric,
        save_path=output_dir / f'{args.metric}_comparison{plot_suffix}.png'
    )
    
    # Plot 2: Per-hour breakdown
    print(f"\nPlotting {args.metric.upper()} vs hour of day...")
    plot_per_hour_metrics(
        lgbm_df, pinn_df,
        metric=args.metric,
        save_path=output_dir / f'{args.metric}_per_hour{plot_suffix}.png'
    )
    
    # Plot 3: By building type
    print(f"\nPlotting {args.metric.upper()} by building type...")
    plot_building_type_comparison(
        lgbm_df, pinn_df,
        metric=args.metric,
        save_path=output_dir / f'{args.metric}_by_building_type{plot_suffix}.png'
    )
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()

