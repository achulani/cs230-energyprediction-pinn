"""
Grid search for PINN hyperparameters on the IDEAL dataset.
Runs experiments with 3 physics losses and 5 lambda values.
"""

import subprocess

def run_gridsearch():

    benchmark = "ideal"
    lambda_values = [0.01, 0.05, 0.1, 0.5, 1.0]

    total_experiments = len(lambda_values) * 3   # 3 losses
    experiment_count = 0

    print("\n" + "="*70)
    print("GRID SEARCH on IDEAL dataset")
    print("="*70)
    print(f"Total experiments to run: {total_experiments}")
    print("="*70 + "\n")

    # =====================================================
    # 1. Smoothness Loss
    # =====================================================
    print("\n" + "="*70)
    print("GRID SEARCH: Smoothness Loss")
    print("="*70 + "\n")

    for lambda_val in lambda_values:
        experiment_count += 1
        print(f"[{experiment_count}/{total_experiments}] λ_smooth={lambda_val}")
        print("-" * 70)

        cmd = [
            "python", "lightgbm_pinn.py",
            "--benchmark", benchmark,
            "--use_smoothness",
            "--lambda_smooth", str(lambda_val),
            "--variant_name", f"ideal_smooth_{lambda_val}",
            "--results_path", "results/gridsearch/ideal_smooth/",
            "--epochs", "50"
        ]

        subprocess.run(cmd)

    # =====================================================
    # 2. Temporal Loss
    # =====================================================
    print("\n" + "="*70)
    print("GRID SEARCH: Temporal Loss")
    print("="*70 + "\n")

    for lambda_val in lambda_values:
        experiment_count += 1
        print(f"[{experiment_count}/{total_experiments}] λ_temporal={lambda_val}")
        print("-" * 70)

        cmd = [
            "python", "lightgbm_pinn.py",
            "--benchmark", benchmark,
            "--use_temporal",
            "--lambda_temporal", str(lambda_val),
            "--variant_name", f"ideal_temporal_{lambda_val}",
            "--results_path", "results/gridsearch/ideal_temporal/",
            "--epochs", "50"
        ]

        subprocess.run(cmd)

    # =====================================================
    # 3. Weather Loss
    # =====================================================
    print("\n" + "="*70)
    print("GRID SEARCH: Weather Loss")
    print("="*70 + "\n")

    for lambda_val in lambda_values:
        experiment_count += 1
        print(f"[{experiment_count}/{total_experiments}] λ_weather={lambda_val}")
        print("-" * 70)

        cmd = [
            "python", "lightgbm_pinn.py",
            "--benchmark", benchmark,
            "--use_weather",
            "--lambda_weather", str(lambda_val),
            "--variant_name", f"ideal_weather_{lambda_val}",
            "--results_path", "results/gridsearch/ideal_weather/",
            "--epochs", "50"
        ]

        subprocess.run(cmd)

    # =====================================================
    # Summary
    # =====================================================
    print("\n" + "="*70)
    print("GRID SEARCH COMPLETE!")
    print("="*70)
    print(f"Total experiments run: {experiment_count}")
    print(f"Benchmark tested: {benchmark}")
    print(f"Lambda values per loss: {len(lambda_values)}")
    print(f"Losses: smoothness, temporal, weather")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_gridsearch()
