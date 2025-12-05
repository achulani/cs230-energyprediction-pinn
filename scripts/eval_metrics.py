"""
Evaluation metrics for building energy forecasting.
Includes both standard ML metrics and physics-based metrics.

Physics metrics focus on UNIVERSAL quality indicators:
- Smoothness: Measures oscillation severity (thermal inertia violation)
- Gradient: Measures rate-of-change violations (HVAC ramp rate limits)
- Negatives: Physically impossible predictions
"""

import numpy as np
import json
from pathlib import Path


def compute_physics_metrics(predictions, ground_truth):
    """
    Compute physics-based evaluation metrics.
    
    These metrics measure prediction quality through universal physical principles
    that apply to all buildings regardless of type, size, or climate.
    
    Args:
        predictions: numpy array of predicted values (24,)
        ground_truth: numpy array of ground truth values (24,)
    
    Returns:
        dict with physics-based metrics
    """
    metrics = {}
    
    # ========================================
    # SMOOTHNESS VIOLATIONS
    # ========================================
    # Physical principle: Buildings have thermal mass - power doesn't oscillate wildly
    # Measures second derivative (acceleration of power changes)
    # Lower is better (smoother predictions)
    
    if len(predictions) >= 3:
        second_diff = predictions[2:] - 2 * predictions[1:-1] + predictions[:-2]
        metrics['smoothness_violation_mean'] = float(np.mean(np.abs(second_diff)))
        metrics['smoothness_violation_max'] = float(np.max(np.abs(second_diff)))
        metrics['smoothness_violation_std'] = float(np.std(np.abs(second_diff)))
    else:
        metrics['smoothness_violation_mean'] = 0.0
        metrics['smoothness_violation_max'] = 0.0
        metrics['smoothness_violation_std'] = 0.0
    
    # ========================================
    # GRADIENT VIOLATIONS
    # ========================================
    # Physical principle: HVAC systems have finite ramp rates
    # Measures hour-to-hour fractional changes
    # Lower is better (more realistic ramp rates)
    
    if len(predictions) >= 2:
        changes = np.abs(predictions[1:] - predictions[:-1])
        mean_pred = np.mean(predictions)
        
        if mean_pred > 0:
            fractional_changes = changes / (mean_pred + 1e-6)
            
            # Statistics on gradients
            metrics['max_gradient'] = float(np.max(fractional_changes))
            metrics['mean_gradient'] = float(np.mean(fractional_changes))
            metrics['std_gradient'] = float(np.std(fractional_changes))
            
            # Count violations at different severity levels
            metrics['gradient_violations_severe'] = int(np.sum(fractional_changes > 0.5))  # >50%/hour
            metrics['gradient_violations_moderate'] = int(np.sum(fractional_changes > 0.3))  # >30%/hour
            
            # Fraction of hours with violations
            metrics['gradient_violation_rate_severe'] = float(metrics['gradient_violations_severe'] / len(fractional_changes))
            metrics['gradient_violation_rate_moderate'] = float(metrics['gradient_violations_moderate'] / len(fractional_changes))
        else:
            metrics['max_gradient'] = 0.0
            metrics['mean_gradient'] = 0.0
            metrics['std_gradient'] = 0.0
            metrics['gradient_violations_severe'] = 0
            metrics['gradient_violations_moderate'] = 0
            metrics['gradient_violation_rate_severe'] = 0.0
            metrics['gradient_violation_rate_moderate'] = 0.0
    else:
        metrics['max_gradient'] = 0.0
        metrics['mean_gradient'] = 0.0
        metrics['std_gradient'] = 0.0
        metrics['gradient_violations_severe'] = 0
        metrics['gradient_violations_moderate'] = 0
        metrics['gradient_violation_rate_severe'] = 0.0
        metrics['gradient_violation_rate_moderate'] = 0.0
    
    # ========================================
    # NEGATIVE PREDICTIONS
    # ========================================
    # Physical principle: Power consumption cannot be negative
    # Lower is better (zero is ideal)
    
    negative_mask = predictions < 0
    metrics['negative_count'] = int(np.sum(negative_mask))
    metrics['negative_energy_total'] = float(np.sum(np.abs(predictions[negative_mask])))
    metrics['negative_fraction'] = float(metrics['negative_count'] / len(predictions))
    
    # Maximum magnitude of negative prediction
    if metrics['negative_count'] > 0:
        metrics['negative_max_magnitude'] = float(np.max(np.abs(predictions[negative_mask])))
    else:
        metrics['negative_max_magnitude'] = 0.0
    
    return metrics


def aggregate_physics_metrics(metrics_list):
    """
    Aggregate a list of physics metrics into summary statistics.
    
    Args:
        metrics_list: list of dicts from compute_physics_metrics()
    
    Returns:
        dict with aggregated statistics (mean, std, median, min, max, count) for each metric
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


def create_per_building_metrics_df(metrics_list):
    """
    Create a DataFrame with one row per building (averaged across its windows).
    
    Args:
        metrics_list: list of dicts with 'dataset', 'building', and metric values
    
    Returns:
        pandas DataFrame with per-building metrics
    """
    import pandas as pd
    
    # Group by building
    building_groups = {}
    for m in metrics_list:
        building_id = f"{m.get('dataset', '')}_{m.get('building', '')}"
        if building_id not in building_groups:
            building_groups[building_id] = {
                'dataset': m.get('dataset', ''),
                'building': m.get('building', ''),
                'metrics': []
            }
        building_groups[building_id]['metrics'].append(m)
    
    # Average metrics per building
    rows = []
    for building_id, data in building_groups.items():
        row = {
            'dataset': data['dataset'],
            'building': data['building'],
            'num_windows': len(data['metrics'])
        }
        
        # Get all metric keys (exclude metadata)
        metric_keys = [k for k in data['metrics'][0].keys() 
                      if k not in ['dataset', 'building', 'window_idx']]
        
        # Average each metric across windows for this building
        for key in metric_keys:
            values = [m[key] for m in data['metrics']]
            row[f'{key}_mean'] = np.mean(values)
            row[f'{key}_std'] = np.std(values)
            row[f'{key}_median'] = np.median(values)
            row[f'{key}_min'] = np.min(values)
            row[f'{key}_max'] = np.max(values)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


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
    
    # Group metrics by category
    smoothness_metrics = {}
    gradient_metrics = {}
    negative_metrics = {}
    
    for metric_name, stats in metrics.items():
        if 'smoothness' in metric_name:
            smoothness_metrics[metric_name] = stats
        elif 'gradient' in metric_name:
            gradient_metrics[metric_name] = stats
        elif 'negative' in metric_name:
            negative_metrics[metric_name] = stats
    
    def print_metric_group(group_name, group_metrics):
        if group_metrics:
            print(f"\n{group_name}:")
            for metric_name, stats in group_metrics.items():
                if isinstance(stats, dict):
                    print(f"  {metric_name}:")
                    for stat_name, value in stats.items():
                        if stat_name == 'count':
                            print(f"    {stat_name:<10} {value:>12.0f}")
                        else:
                            print(f"    {stat_name:<10} {value:>12.6f}")
    
    print_metric_group("SMOOTHNESS VIOLATIONS", smoothness_metrics)
    print_metric_group("GRADIENT VIOLATIONS", gradient_metrics)
    print_metric_group("NEGATIVE PREDICTIONS", negative_metrics)
    
    print(f"{'='*70}\n")
