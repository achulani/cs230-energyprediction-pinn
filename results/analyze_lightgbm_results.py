#!/usr/bin/env python3
"""
Script to analyze LightGBM transfer learning results and compare to published benchmarks.

Usage:
    python analyze_lightgbm_results.py [--results_file PATH]
"""

import pandas as pd
import argparse
from pathlib import Path

# Published NRMSE benchmarks from README (Transfer Learning section)
PUBLISHED_BENCHMARKS = {
    'commercial': 16.02,  # NRMSE %
    'residential': 80.07  # NRMSE %
}


def analyze_results(results_file: Path):
    """
    Analyze LightGBM results and compare to published benchmarks.

    Args:
        results_file: Path to the CSV file containing LightGBM metrics
    """
    # Read the results
    df = pd.read_csv(results_file)

    print("=" * 60)
    print("Raw data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("=" * 60)
    print()

    # Filter to only CVRMSE metrics (hourly metrics: cvrmse_0 through cvrmse_23)
    cvrmse_df = df[df['metric'].str.startswith('cvrmse_')]

    # Group by building to compute mean CVRMSE across all hours
    building_cvrmse = cvrmse_df.groupby(['dataset', 'building_id', 'building_type'])['value'].mean().reset_index()
    building_cvrmse.columns = ['dataset', 'building_id', 'building_type', 'mean_cvrmse']

    # Convert from decimal to percentage (multiply by 100)
    building_cvrmse['mean_cvrmse_pct'] = building_cvrmse['mean_cvrmse'] * 100

    # Count buildings by type
    building_counts = building_cvrmse.groupby('building_type')['building_id'].count()

    print("=" * 60)
    print("CORRECTED Results (only target buildings):")
    print("=" * 60)
    print(building_counts)
    print()

    # Calculate overall average CVRMSE by building type
    avg_cvrmse_by_type = building_cvrmse.groupby('building_type')['mean_cvrmse_pct'].mean()

    print("=" * 60)
    print("CORRECTED OVERALL AVERAGE CVRMSE:")
    print("=" * 60)
    for building_type in avg_cvrmse_by_type.index:
        print(f"{building_type.capitalize()}: {avg_cvrmse_by_type[building_type]:.2f}%")
    print()

    print("Published NRMSE benchmarks:")
    for building_type, benchmark in PUBLISHED_BENCHMARKS.items():
        print(f"{building_type.capitalize()}: {benchmark:.2f}%")
    print()

    # Calculate differences
    print("=" * 60)
    print("COMPARISON TO PUBLISHED BENCHMARKS:")
    print("= " * 60)
    for building_type in avg_cvrmse_by_type.index:
        actual = avg_cvrmse_by_type[building_type]
        benchmark = PUBLISHED_BENCHMARKS.get(building_type, None)
        if benchmark is not None:
            diff = actual - benchmark
            pct_change = (diff / benchmark) * 100
            comparison = "BETTER" if diff < 0 else "WORSE"
            print(f"{building_type.capitalize()}:")
            print(f"  Your result: {actual:.2f}%")
            print(f"  Benchmark:   {benchmark:.2f}%")
            print(f"  Difference:  {diff:+.2f}% ({pct_change:+.2f}% {comparison})")
            print()

    # Show per-dataset breakdown
    print("=" * 60)
    print("BREAKDOWN BY DATASET:")
    print("=" * 60)
    dataset_breakdown = building_cvrmse.groupby(['dataset', 'building_type']).agg({
        'building_id': 'count',
        'mean_cvrmse_pct': 'mean'
    }).reset_index()
    dataset_breakdown.columns = ['dataset', 'building_type', 'num_buildings', 'avg_cvrmse_pct']

    for building_type in dataset_breakdown['building_type'].unique():
        print(f"\n{building_type.upper()}:")
        type_data = dataset_breakdown[dataset_breakdown['building_type'] == building_type]
        for _, row in type_data.iterrows():
            print(f"  {row['dataset']:20s} ({row['num_buildings']:3.0f} buildings): {row['avg_cvrmse_pct']:6.2f}%")
    print()

    # Show best and worst performing buildings
    print("=" * 60)
    print("TOP 5 BEST PERFORMING BUILDINGS (by type):")
    print("=" * 60)
    for building_type in building_cvrmse['building_type'].unique():
        print(f"\n{building_type.upper()}:")
        type_buildings = building_cvrmse[building_cvrmse['building_type'] == building_type]
        best = type_buildings.nsmallest(5, 'mean_cvrmse_pct')
        for _, row in best.iterrows():
            print(f"  {row['dataset']:15s} / {row['building_id']:20s}: {row['mean_cvrmse_pct']:6.2f}%")

    print()
    print("=" * 60)
    print("TOP 5 WORST PERFORMING BUILDINGS (by type):")
    print("=" * 60)
    for building_type in building_cvrmse['building_type'].unique():
        print(f"\n{building_type.upper()}:")
        type_buildings = building_cvrmse[building_cvrmse['building_type'] == building_type]
        worst = type_buildings.nlargest(5, 'mean_cvrmse_pct')
        for _, row in worst.iterrows():
            print(f"  {row['dataset']:15s} / {row['building_id']:20s}: {row['mean_cvrmse_pct']:6.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze LightGBM transfer learning results and compare to benchmarks'
    )
    parser.add_argument(
        '--results_file',
        type=str,
        default='TL_metrics_lightgbm.csv',
        help='Path to the CSV file containing LightGBM metrics (default: TL_metrics_lightgbm.csv)'
    )

    args = parser.parse_args()
    results_file = Path(args.results_file)

    if not results_file.exists():
        # Try looking in the results directory
        alt_path = Path(__file__).parent / args.results_file
        if alt_path.exists():
            results_file = alt_path
        else:
            print(f"Error: Results file not found at {results_file}")
            print(f"Also tried: {alt_path}")
            return

    analyze_results(results_file)


if __name__ == '__main__':
    main()
