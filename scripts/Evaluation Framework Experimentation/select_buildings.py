import pandas as pd
import numpy as np

# Load results
standard = pd.read_csv('results/baseline/metrics_standard.csv')
physics = pd.read_csv('results/baseline/physics_per_building.csv')

# Convert standard from long to wide format
standard_wide = standard.pivot_table(
    index=['dataset', 'building_id', 'building_type'],
    columns='metric',
    values='value'
).reset_index()

# Average hourly metrics (cvrmse_0 through cvrmse_23, etc.)
# Find all cvrmse columns
cvrmse_cols = [col for col in standard_wide.columns if col.startswith('cvrmse_')]
if cvrmse_cols:
    standard_wide['cvrmse'] = standard_wide[cvrmse_cols].mean(axis=1)

# Do the same for other hourly metrics if they exist
for metric_prefix in ['nmbe', 'mae', 'rmse', 'mape']:
    metric_cols = [col for col in standard_wide.columns if col.startswith(f'{metric_prefix}_')]
    if metric_cols:
        standard_wide[metric_prefix] = standard_wide[metric_cols].mean(axis=1)

# Convert CVRMSE from fraction to percentage
standard_wide['cvrmse'] = standard_wide['cvrmse'] * 100

# Rename building_id to building to match physics dataframe
standard_wide = standard_wide.rename(columns={'building_id': 'building'})

# Merge
df = standard_wide.merge(physics, on=['dataset', 'building'])

# Create a composite difficulty score
# Prioritize CVRMSE, with moderate weights on physics violations
df['difficulty_score'] = (
    df['cvrmse'] * 0.6 +  # 60% weight on prediction accuracy
    df['smoothness_violation_mean_mean'] * 20 * 0.2 +  # 20% weight on smoothness
    df['max_gradient_mean'] * 100 * 0.2  # 20% weight on gradient violations
)

# Sort by composite score
df = df.sort_values('difficulty_score')

# Pick 3 buildings at different difficulty levels
easy_idx = len(df) // 4  # 25th percentile
medium_idx = len(df) // 2  # 50th percentile
hard_idx = (3 * len(df)) // 4  # 75th percentile

easy = df.iloc[easy_idx]
medium = df.iloc[medium_idx]
hard = df.iloc[hard_idx]

selected = [
    f"{easy['dataset']}/{easy['building']}",
    f"{medium['dataset']}/{medium['building']}",
    f"{hard['dataset']}/{hard['building']}"
]

print("Selected buildings for grid search:")
for i, building in enumerate(selected, 1):
    row = df[df.apply(lambda x: f"{x['dataset']}/{x['building']}" == building, axis=1)].iloc[0]
    print(f"{i}. {building}")
    print(f"   Difficulty Score: {row['difficulty_score']:.2f}")
    print(f"   CVRMSE: {row['cvrmse']:.1f}%")
    print(f"   Smoothness: {row['smoothness_violation_mean_mean']:.2f}")
    print(f"   Max Gradient: {row['max_gradient_mean']:.3f}\n")

# Save to file
with open('selected_buildings.txt', 'w') as f:
    for building in selected:
        f.write(building + '\n')

print("Saved to selected_buildings.txt")
