"""
Run ablation study on PINN loss components.

This script runs all loss combination scenarios:
1. LightGBM + Residuals (data loss only)
2. LightGBM + Residuals + RC loss
3. LightGBM + Residuals + Comfort loss
4. LightGBM + Residuals + Smoothness loss
5. LightGBM + Residuals + All losses (with tuned weights)
"""

import subprocess
import sys
from pathlib import Path

# Base command
BASE_CMD = [
    'python3', 'scripts/transfer_learning_pinn.py',
    '--device', 'cpu',
    '--use_temperature_input',
    '--results_path', 'results/'
]

# Experiment scenarios
SCENARIOS = [
    {
        'name': 'residuals_only',
        'description': 'LightGBM + Residuals (data loss only)',
        'args': [
            '--no_rc_loss',
            '--no_comfort_loss',
            '--no_smooth_loss',
            '--experiment_name', 'residuals_only'
        ]
    },
    {
        'name': 'residuals_rc',
        'description': 'LightGBM + Residuals + RC loss',
        'args': [
            '--use_rc_loss',
            '--no_comfort_loss',
            '--no_smooth_loss',
            '--lambda_rc', '1.0',
            '--experiment_name', 'residuals_rc'
        ]
    },
    {
        'name': 'residuals_comfort',
        'description': 'LightGBM + Residuals + Comfort loss',
        'args': [
            '--no_rc_loss',
            '--use_comfort_loss',
            '--no_smooth_loss',
            '--lambda_comfort', '0.1',
            '--experiment_name', 'residuals_comfort'
        ]
    },
    {
        'name': 'residuals_smooth',
        'description': 'LightGBM + Residuals + Smoothness loss',
        'args': [
            '--no_rc_loss',
            '--no_comfort_loss',
            '--use_smooth_loss',
            '--lambda_smooth', '0.01',
            '--experiment_name', 'residuals_smooth'
        ]
    },
    {
        'name': 'residuals_all',
        'description': 'LightGBM + Residuals + All losses',
        'args': [
            '--use_rc_loss',
            '--use_comfort_loss',
            '--use_smooth_loss',
            '--lambda_rc', '1.0',
            '--lambda_comfort', '0.1',
            '--lambda_smooth', '0.01',
            '--experiment_name', 'residuals_all'
        ]
    }
]

# Optional: After running individual scenarios, you can run with tuned weights
# This would be scenario 6, but weights should be tuned based on previous results
TUNED_WEIGHTS_SCENARIO = {
    'name': 'residuals_all_tuned',
    'description': 'LightGBM + Residuals + All losses (tuned weights)',
    'args': [
        '--use_rc_loss',
        '--use_comfort_loss',
        '--use_smooth_loss',
        '--lambda_rc', '1.0',  # Tune based on previous results
        '--lambda_comfort', '0.1',  # Tune based on previous results
        '--lambda_smooth', '0.01',  # Tune based on previous results
        '--experiment_name', 'residuals_all_tuned'
    ]
}


def run_scenario(scenario, benchmark='BDG-2', max_buildings=None):
    """Run a single scenario."""
    print(f"\n{'='*80}")
    print(f"Running: {scenario['description']}")
    print(f"Scenario: {scenario['name']}")
    print(f"{'='*80}\n")
    
    cmd = BASE_CMD + [
        '--benchmark', benchmark,
        '--experiment_name', scenario['name']
    ] + scenario['args']
    
    if max_buildings:
        # Limit to first N buildings for faster testing
        # This would require modifying the script to accept --max_buildings
        pass
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Completed: {scenario['name']}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {scenario['name']} (exit code {e.returncode})\n")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study on PINN loss components')
    parser.add_argument('--benchmark', type=str, default='BDG-2',
                       help='Dataset to run experiments on')
    parser.add_argument('--scenario', type=str, default=None,
                       choices=[s['name'] for s in SCENARIOS] + ['all', 'tuned'],
                       help='Which scenario to run (default: all)')
    parser.add_argument('--skip', nargs='+', type=str, default=[],
                       help='Scenarios to skip')
    parser.add_argument('--results_dir', type=str, default='results/',
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Determine which scenarios to run
    if args.scenario == 'all' or args.scenario is None:
        scenarios_to_run = [s for s in SCENARIOS if s['name'] not in args.skip]
    elif args.scenario == 'tuned':
        scenarios_to_run = [TUNED_WEIGHTS_SCENARIO]
    else:
        scenarios_to_run = [s for s in SCENARIOS if s['name'] == args.scenario]
        if not scenarios_to_run:
            print(f"Error: Scenario '{args.scenario}' not found")
            sys.exit(1)
    
    print(f"\nRunning {len(scenarios_to_run)} scenario(s) on benchmark: {args.benchmark}")
    print(f"Results will be saved to: {args.results_dir}\n")
    
    results = {}
    for scenario in scenarios_to_run:
        success = run_scenario(scenario, benchmark=args.benchmark)
        results[scenario['name']] = success
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nResults saved to: {args.results_dir}")
    print(f"Look for files with experiment names: {', '.join([s['name'] for s in scenarios_to_run])}")


if __name__ == '__main__':
    main()

