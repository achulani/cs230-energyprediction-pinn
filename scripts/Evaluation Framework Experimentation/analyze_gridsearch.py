import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_gridsearch(loss_type):
    """Analyze grid search results for a given loss type."""
    
    results = []
    # Map loss_type to directory name and file prefix
    dir_map = {
        'smoothness': 'ideal_smooth',
        'temporal': 'ideal_temporal',
        'weather': 'ideal_weather'
    }
    file_prefix_map = {
        'smoothness': 'smooth',
        'temporal': 'temporal',
        'weather': 'weather'
    }
    
    gridsearch_dir = Path(f'results/gridsearch/{dir_map[loss_type]}/')
    file_prefix = file_prefix_map[loss_type]
    
    if not gridsearch_dir.exists():
        print(f"Warning: Directory not found: {gridsearch_dir}")
        return None, None
    
    # Collect all results - pattern: metrics_pinn_{prefix}_ideal_{prefix}_{lambda}.csv
    pattern = f'metrics_pinn_{file_prefix}_ideal_{file_prefix}_*.csv'
    
    for metrics_file in gridsearch_dir.glob(pattern):
        # Extract lambda from filename
        # e.g., metrics_pinn_smooth_ideal_smooth_0.1.csv -> 0.1
        parts = metrics_file.stem.split('_')
        lambda_val = float(parts[-1])
        
        # Load standard metrics (in long format)
        df = pd.read_csv(metrics_file)
        
        # Convert from long to wide format
        df_wide = df.pivot_table(
            index=['dataset', 'building_id', 'building_type'],
            columns='metric',
            values='value'
        ).reset_index()
        
        # Average hourly CVRMSE metrics (cvrmse_0 through cvrmse_23)
        cvrmse_cols = [col for col in df_wide.columns if col.startswith('cvrmse_')]
        if cvrmse_cols:
            df_wide['cvrmse'] = df_wide[cvrmse_cols].mean(axis=1) * 100  # Convert to percentage
        else:
            print(f"Warning: No cvrmse columns found in {metrics_file}")
            continue
        
        # Load physics metrics (optional, just for visualization)
        physics_file = gridsearch_dir / 'physics_metrics' / f'physics_aggregated_gridsearch_{loss_type}_{lambda_val}.json'
        if physics_file.exists():
            with open(physics_file) as f:
                physics = json.load(f)
            smoothness_mean = physics['smoothness_violation_mean']['mean']
            max_gradient_mean = physics['max_gradient']['mean']
        else:
            # Try alternative naming
            physics_file = gridsearch_dir / 'physics_metrics' / f'physics_aggregated_{loss_type}_{lambda_val}.json'
            if physics_file.exists():
                with open(physics_file) as f:
                    physics = json.load(f)
                smoothness_mean = physics['smoothness_violation_mean']['mean']
                max_gradient_mean = physics['max_gradient']['mean']
            else:
                smoothness_mean = None
                max_gradient_mean = None
        
        # Aggregate across all buildings in this run
        result = {
            'lambda': lambda_val,
            'cvrmse_mean': df_wide['cvrmse'].mean()
        }
        if smoothness_mean is not None:
            result['smoothness_mean'] = smoothness_mean
            result['max_gradient_mean'] = max_gradient_mean
        
        results.append(result)
    
    if not results:
        print(f"No results found for {loss_type}")
        return None, None
    
    results_df = pd.DataFrame(results).sort_values('lambda')
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # CVRMSE vs lambda
    axes[0, 0].plot(results_df['lambda'], results_df['cvrmse_mean'], 'o-')
    axes[0, 0].set_xlabel('Lambda')
    axes[0, 0].set_ylabel('CVRMSE (%)')
    axes[0, 0].set_title(f'CVRMSE vs Lambda ({loss_type})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Smoothness vs lambda (only if physics data available)
    if 'smoothness_mean' in results_df.columns:
        axes[0, 1].plot(results_df['lambda'], results_df['smoothness_mean'], 'o-', color='orange')
        axes[0, 1].set_xlabel('Lambda')
        axes[0, 1].set_ylabel('Smoothness Violation')
        axes[0, 1].set_title(f'Smoothness vs Lambda ({loss_type})')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Physics metrics\nnot available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # Gradient vs lambda (only if physics data available)
    if 'max_gradient_mean' in results_df.columns:
        axes[1, 0].plot(results_df['lambda'], results_df['max_gradient_mean'], 'o-', color='green')
        axes[1, 0].set_xlabel('Lambda')
        axes[1, 0].set_ylabel('Max Gradient')
        axes[1, 0].set_title(f'Gradient Violations vs Lambda ({loss_type})')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Physics metrics\nnot available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Tradeoff: CVRMSE vs Physics (only if physics data available)
    if 'smoothness_mean' in results_df.columns:
        scatter = axes[1, 1].scatter(results_df['cvrmse_mean'], results_df['smoothness_mean'], 
                           c=results_df['lambda'], cmap='viridis', s=100)
        axes[1, 1].set_xlabel('CVRMSE (%)')
        axes[1, 1].set_ylabel('Smoothness Violation')
        axes[1, 1].set_title(f'Accuracy vs Physics Tradeoff ({loss_type})')
        axes[1, 1].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Lambda')
    else:
        axes[1, 1].text(0.5, 0.5, 'Physics metrics\nnot available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig(f'results/gridsearch/{dir_map[loss_type]}/{loss_type}_gridsearch_analysis.png', dpi=150)
    plt.close()
    
    # Find best lambda using ONLY CVRMSE
    best_lambda = results_df.loc[results_df['cvrmse_mean'].idxmin(), 'lambda']
    
    print(f"\n{'='*60}")
    print(f"Grid Search Results: {loss_type.upper()}")
    print(f"{'='*60}")
    # Only show lambda and CVRMSE columns
    display_df = results_df[['lambda', 'cvrmse_mean']].copy()
    display_df.columns = ['Lambda', 'CVRMSE (%)']
    print(display_df.to_string(index=False))
    print(f"\n✅ Best lambda (lowest CVRMSE): {best_lambda}")
    print(f"   CVRMSE at best lambda: {results_df.loc[results_df['cvrmse_mean'].idxmin(), 'cvrmse_mean']:.2f}%")
    print(f"{'='*60}\n")
    
    return best_lambda, results_df

# Analyze all loss types
best_lambdas = {}
for loss_type in ['smoothness', 'temporal', 'weather']:
    best_lambda, results = analyze_gridsearch(loss_type)
    if best_lambda is not None:
        best_lambdas[loss_type] = best_lambda

if not best_lambdas:
    print("No results found. Check your file paths and naming conventions.")
    exit(1)

# Save best lambdas
print("\n" + "="*60)
print("OPTIMAL HYPERPARAMETERS")
print("="*60)
for loss_type, lambda_val in best_lambdas.items():
    print(f"  lambda_{loss_type}: {lambda_val}")
print("="*60)

with open('best_lambdas.txt', 'w') as f:
    for loss_type, lambda_val in best_lambdas.items():
        f.write(f"--lambda_{loss_type} {lambda_val} ")

print("\n✅ Saved to best_lambdas.txt")
print("\nUse in final run:")
print(f"python lightgbm_pinn.py --use_smoothness --use_temporal --use_weather $(cat best_lambdas.txt) --benchmark all")
