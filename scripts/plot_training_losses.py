"""
Plot training and validation loss curves from PINN training.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

# Set style
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


def plot_loss_curves(loss_file: Path, save_path: Path = None, plot_components: bool = True):
    """Plot training and validation loss curves from a single loss history file."""
    
    with open(loss_file, 'r') as f:
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
        
        plt.suptitle(f'Training Progress: {loss_file.stem}', fontsize=14, y=0.995)
        plt.tight_layout()
    else:
        # Simple plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=4)
        plt.plot(epochs, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss Curves\n{loss_file.stem}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def plot_all_loss_curves(results_dir: Path, save_path: Path = None):
    """Plot loss curves for all buildings in a results directory."""
    
    # Find all loss history files
    loss_files = list(results_dir.glob('loss_history_*.json'))
    
    if len(loss_files) == 0:
        print(f"No loss history files found in {results_dir}")
        return
    
    print(f"Found {len(loss_files)} loss history files")
    
    # Create subplots
    n_files = len(loss_files)
    n_cols = min(3, n_files)
    n_rows = (n_files + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_files == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, loss_file in enumerate(loss_files):
        with open(loss_file, 'r') as f:
            loss_history = json.load(f)
        
        epochs = loss_history['epoch']
        if isinstance(loss_history['train'], list):
            train_losses = loss_history['train']
            val_losses = loss_history['val']
        else:
            train_losses = loss_history['train']['total']
            val_losses = loss_history['val']['total']
        
        ax = axes[idx]
        ax.plot(epochs, train_losses, 'o-', label='Train', linewidth=1.5, markersize=3)
        ax.plot(epochs, val_losses, 's-', label='Val', linewidth=1.5, markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(loss_file.stem.replace('loss_history_', ''), fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_files, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined plot to {save_path}")
    else:
        plt.show()


def plot_aggregated_losses(results_dir: Path, save_path: Path = None):
    """Plot mean and std of losses across all buildings."""
    
    loss_files = list(results_dir.glob('loss_history_*.json'))
    
    if len(loss_files) == 0:
        print(f"No loss history files found in {results_dir}")
        return
    
    # First pass: collect all losses to compute global outlier thresholds
    all_train_values = []
    all_val_values = []
    
    for loss_file in loss_files:
        try:
            with open(loss_file, 'r') as f:
                loss_history = json.load(f)
            
            if isinstance(loss_history['train'], list):
                train_losses = np.array(loss_history['train'])
                val_losses = np.array(loss_history['val'])
            else:
                train_losses = np.array(loss_history['train']['total'])
                val_losses = np.array(loss_history['val']['total'])
            
            # Collect valid values
            valid_train = train_losses[np.isfinite(train_losses) & (train_losses > 0)]
            valid_val = val_losses[np.isfinite(val_losses) & (val_losses > 0)]
            
            if len(valid_train) > 0:
                all_train_values.extend(valid_train.tolist())
            if len(valid_val) > 0:
                all_val_values.extend(valid_val.tolist())
        except Exception:
            continue
    
    # Compute global outlier thresholds using robust method
    # Use 95th percentile as base, then cap at reasonable maximum
    if len(all_train_values) > 0:
        train_95th = np.percentile(all_train_values, 95)
        # Cap at 10x the 95th percentile, but never more than 1e7 (10 million)
        max_train = min(1e7, max(1e6, train_95th * 10))
    else:
        max_train = 1e7
    
    if len(all_val_values) > 0:
        val_95th = np.percentile(all_val_values, 95)
        # Cap at 10x the 95th percentile, but never more than 1e7 (10 million)
        max_val = min(1e7, max(1e6, val_95th * 10))
    else:
        max_val = 1e7
    
    # Second pass: collect losses with their epoch numbers, filtering outliers
    all_train_data = []  # List of (epochs, losses) tuples
    all_val_data = []    # List of (epochs, losses) tuples
    
    for loss_file in loss_files:
        try:
            with open(loss_file, 'r') as f:
                loss_history = json.load(f)
            
            epochs = np.array(loss_history['epoch'])
            if isinstance(loss_history['train'], list):
                train_losses = np.array(loss_history['train'])
                val_losses = np.array(loss_history['val'])
            else:
                train_losses = np.array(loss_history['train']['total'])
                val_losses = np.array(loss_history['val']['total'])
            
            # Filter out invalid values and extreme outliers
            valid_mask = (np.isfinite(train_losses) & (train_losses > 0) & (train_losses < max_train) &
                          np.isfinite(val_losses) & (val_losses > 0) & (val_losses < max_val))
            if np.sum(valid_mask) == 0:
                continue
            
            valid_epochs = epochs[valid_mask]
            valid_train = train_losses[valid_mask]
            valid_val = val_losses[valid_mask]
            
            all_train_data.append((valid_epochs, valid_train))
            all_val_data.append((valid_epochs, valid_val))
        except Exception as e:
            continue
    
    if len(all_train_data) == 0:
        print("No valid loss data found")
        return
    
    # Find the common epoch range across all files
    all_epochs = set()
    for epochs, _ in all_train_data:
        all_epochs.update(epochs)
    all_epochs = sorted(all_epochs)
    
    if len(all_epochs) == 0:
        print("No valid epochs found")
        return
    
    # Interpolate all curves to common epoch grid
    epoch_grid = np.array(all_epochs)
    interpolated_train = []
    interpolated_val = []
    
    for (epochs, train_losses), (_, val_losses) in zip(all_train_data, all_val_data):
        # Interpolate to common grid (forward fill for missing values)
        train_interp = np.interp(epoch_grid, epochs, train_losses, 
                                 left=train_losses[0] if len(train_losses) > 0 else np.nan,
                                 right=train_losses[-1] if len(train_losses) > 0 else np.nan)
        val_interp = np.interp(epoch_grid, epochs, val_losses,
                               left=val_losses[0] if len(val_losses) > 0 else np.nan,
                               right=val_losses[-1] if len(val_losses) > 0 else np.nan)
        
        # Filter out NaN values
        valid_mask = np.isfinite(train_interp) & np.isfinite(val_interp)
        if np.any(valid_mask):
            interpolated_train.append(train_interp)
            interpolated_val.append(val_interp)
    
    if len(interpolated_train) == 0:
        print("No valid interpolated data found")
        return
    
    # Convert to numpy arrays
    interpolated_train = np.array(interpolated_train)
    interpolated_val = np.array(interpolated_val)
    
    # Compute mean and std
    train_mean = np.nanmean(interpolated_train, axis=0)
    train_std = np.nanstd(interpolated_train, axis=0)
    val_mean = np.nanmean(interpolated_val, axis=0)
    val_std = np.nanstd(interpolated_val, axis=0)
    
    # Filter out NaN means
    valid_mask = np.isfinite(train_mean) & np.isfinite(val_mean)
    epoch_grid = epoch_grid[valid_mask]
    train_mean = train_mean[valid_mask]
    train_std = train_std[valid_mask]
    val_mean = val_mean[valid_mask]
    val_std = val_std[valid_mask]
    
    if len(epoch_grid) == 0:
        print("No valid data after filtering")
        return
    
    # Plot with log scale if losses are very large
    use_log_scale = np.max(train_mean) > 1000 or np.max(val_mean) > 1000
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_grid, train_mean, 'o-', label='Training Loss (mean)', linewidth=2, markersize=4)
    plt.fill_between(epoch_grid, train_mean - train_std, train_mean + train_std, alpha=0.3)
    plt.plot(epoch_grid, val_mean, 's-', label='Validation Loss (mean)', linewidth=2, markersize=4)
    plt.fill_between(epoch_grid, val_mean - val_std, val_mean + val_std, alpha=0.3)
    
    if use_log_scale:
        plt.yscale('log')
        plt.ylabel('Loss (log scale)')
    else:
        plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (aggregated across {len(loss_files)} buildings)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved aggregated plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot PINN training loss curves')
    parser.add_argument('--loss_file', type=str, default=None,
                       help='Path to a specific loss history JSON file')
    parser.add_argument('--results_dir', type=str, default='results/',
                       help='Directory containing loss history files')
    parser.add_argument('--output_dir', type=str, default='results/plots',
                       help='Directory to save plots')
    parser.add_argument('--mode', type=str, default='aggregated',
                       choices=['single', 'all', 'aggregated'],
                       help='Plotting mode: single file, all files, or aggregated')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'single':
        if args.loss_file is None:
            # Find first loss file
            loss_files = list(results_dir.glob('loss_history_*.json'))
            if len(loss_files) == 0:
                print(f"No loss history files found in {results_dir}")
                return
            loss_file = loss_files[0]
            print(f"Using first loss file: {loss_file}")
        else:
            loss_file = Path(args.loss_file)
        
        plot_loss_curves(
            loss_file,
            save_path=output_dir / f'{loss_file.stem}.png'
        )
    
    elif args.mode == 'all':
        plot_all_loss_curves(
            results_dir,
            save_path=output_dir / 'all_training_losses.png'
        )
    
    elif args.mode == 'aggregated':
        plot_aggregated_losses(
            results_dir,
            save_path=output_dir / 'aggregated_training_losses.png'
        )


if __name__ == '__main__':
    main()

