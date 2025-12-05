"""
Evaluation metrics for building energy forecasting.
Includes both standard ML metrics and physics-based metrics.
"""

import numpy as np
import json
from pathlib import Path


def compute_physics_metrics(predictions, ground_truth):
    """
    Compute physics-based evaluation metrics.
    
    Args:
        predictions: numpy array of predicted values (24,)
        ground_truth: numpy array of ground truth values (24,)
    
    Returns:
        dict with physics-based metrics:
        - smoothness_violation: mean absolute second derivative
        - peak_ratio: ratio of max to mean
        - negative_count: number of negative predictions
        - negative_energy: sum of absolute negative energy
        - max_rate_of_change: maximum fractional hour-to-hour change
        - mean_rate_of_change: mean fractional hour-to-hour change
    """
    metrics = {}
    
    # Smoothness violation (penalizes oscillations)
    if len(predictions) >= 3:
        second_diff = predictions[2:] - 2 * predictions[1:-1] + predictions[:-2]
        metrics['smoothness_violation'] = float(np.mean(np.abs(second_diff)))
    else:
        metrics['smoothness_violation'] = 0.0
    
    # Peak ratio (realistic daily patterns should have reasonable peaks)
    mean_pred = np.mean(predictions)
    if mean_pred > 0:
        metrics['peak_ratio'] = float(np.max(predictions) / mean_pred)
    else:
        metrics['peak_ratio'] = 0.0
    
    # Negative predictions (physically impossible)
    metrics['negative_count'] = int(np.sum(predictions < 0))
    metrics['negative_energy'] = float(np.sum(np.abs(predictions[predictions < 0])))
    
    # Rate of change violations
    if len(predictions) >= 2:
        changes = np.abs(predictions[1:] - predictions[:-1])
        if mean_pred > 0:
            fractional_changes = changes / (mean_pred + 1e-6)
            metrics['max_rate_of_change'] = float(np.max(fractional_changes))
            metrics['mean_rate_of_change'] = float(np.mean(fractional_changes))
        else:
            metrics['max_rate_of_change'] = 0.0
            metrics['mean_rate_of_change'] = 0.0
    else:
        metrics['max_rate_of_change'] = 0.0
        metrics['mean_rate_of_change'] = 0.0
    
    return metrics


def aggregate_physics_metrics(metrics_list):
    """
    Aggregate a list of physics metrics into summary statistics.
    
    Args:
        metrics_list: list of dicts from compute_physics_metrics()
    
    Returns:
        dict with aggregated statistics (mean, std, median, min, max) for each metric
    """
    if len(metrics_list) == 0:
        return {}
    
    summary = {}
    
    # Get all metric keys (excluding metadata)
    all_keys = metrics_list[0].keys()
    metric_keys = [k for k in all_keys if k not in ['dataset', 'building', 'window_idx']]
    
    for metric_key in metric_keys:
        values = [m[metric_key] for m in metrics_list]
        summary[metric_key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }
    
    return summary


def save_metrics_to_json(metrics, filepath):
    """Save metrics dictionary to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics saved to {filepath}")


def print_metrics_summary(metrics, title="Metrics Summary"):
    """Pretty print metrics summary."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    for metric_name, stats in metrics.items():
        if isinstance(stats, dict):
            print(f"\n{metric_name}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name:<10} {value:>12.4f}")
        else:
            print(f"{metric_name:<30} {stats:>12.4f}")
    
    print(f"{'='*70}\n")
