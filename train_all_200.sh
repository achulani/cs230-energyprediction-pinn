#!/bin/bash
# Train on all 200 target buildings using all available datasets

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BUILDINGS_BENCH="${SCRIPT_DIR}/data/buildings-bench"

python3 scripts/transfer_learning_pinn.py \
  --architecture lstm \
  --config buildings_bench/configs/LSTMResidual-S.toml \
  --benchmark bdg-2 borealis electricity ideal lcl sceaux smart \
  --use_temperature_input \
  --no_rc_loss  \
  --run_suffix no_physics_losses 
