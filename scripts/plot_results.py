"""
Plot comparison of LightGBM baseline vs PINN hybrid model results.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

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
    plt.plot(pinn_agg['hour'], pinn_agg['value'], 's-', label='PINN', linewidth=2, markersize=6)
    plt.xlabel('Hour of Day')
    plt.ylabel(f'{metric.upper()}')
    plt.title(f'{metric.upper()} vs Hour of Day (averaged across buildings)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_building_type_comparison(lgbm_df: pd.DataFrame, pinn_df: pd.DataFrame,
                                  metric: str = 'cvrmse', save_path: Path = None):
    """Plot comparison by building type."""
    
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
            lgbm_overall = lgbm_hourly.groupby(['dataset', 'building_id', 'building_type'])['value'].mean().reset_index()
        else:
            print(f"No data found for LightGBM metric {metric}")
            return
    
    if len(pinn_overall) == 0:
        pinn_hourly = pinn_df[(pinn_df['metric'].str.startswith(metric)) & 
                              (pinn_df['metric_type'] == 'hour_of_day')].copy()
        if len(pinn_hourly) > 0:
            pinn_overall = pinn_hourly.groupby(['dataset', 'building_id', 'building_type'])['value'].mean().reset_index()
        else:
            print(f"No data found for PINN metric {metric}")
            return
    
    # Merge
    comparison = pd.merge(
        lgbm_overall[['dataset', 'building_id', 'building_type', 'value']].rename(columns={'value': 'LightGBM'}),
        pinn_overall[['dataset', 'building_id', 'building_type', 'value']].rename(columns={'value': 'PINN'}),
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
    
    args = parser.parse_args()
    
    # Load data
    lgbm_df = pd.read_csv(args.lightgbm_file)
    pinn_df = pd.read_csv(args.pinn_file)
    
    print(f"Loaded {len(lgbm_df)} LightGBM records and {len(pinn_df)} PINN records")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Overall comparison
    print(f"\nPlotting {args.metric.upper()} comparison...")
    plot_metric_comparison(
        lgbm_df, pinn_df, 
        metric=args.metric,
        save_path=output_dir / f'{args.metric}_comparison.png'
    )
    
    # Plot 2: Per-hour breakdown
    print(f"\nPlotting {args.metric.upper()} vs hour of day...")
    plot_per_hour_metrics(
        lgbm_df, pinn_df,
        metric=args.metric,
        save_path=output_dir / f'{args.metric}_per_hour.png'
    )
    
    # Plot 3: By building type
    print(f"\nPlotting {args.metric.upper()} by building type...")
    plot_building_type_comparison(
        lgbm_df, pinn_df,
        metric=args.metric,
        save_path=output_dir / f'{args.metric}_by_building_type.png'
    )
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()

