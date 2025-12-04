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
    
    # Collect all loss histories
    all_train = []
    all_val = []
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
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mean, 'o-', label='Training Loss (mean)', linewidth=2, markersize=4)
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.3)
    plt.plot(epochs, val_mean, 's-', label='Validation Loss (mean)', linewidth=2, markersize=4)
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.3)
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

