#!/bin/bash
# Train transformer architecture with no RC loss on all 200 target buildings
# This script uses nohup to allow the computer to sleep during training

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
export BUILDINGS_BENCH="${SCRIPT_DIR}/data/buildings-bench"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for unique run identification
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_SUFFIX="transformer_no_rc_${TIMESTAMP}"

# Log file for this run
LOG_FILE="logs/train_transformer_no_rc_${TIMESTAMP}.log"

echo "=========================================="
echo "Starting Transformer Training (No RC Loss)"
echo "=========================================="
echo "Run suffix: ${RUN_SUFFIX}"
echo "Log file: ${LOG_FILE}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Run with nohup so it continues even if terminal closes or computer sleeps
# Redirect both stdout and stderr to log file
nohup python3 scripts/transfer_learning_pinn.py \
  --architecture transformer \
  --config buildings_bench/configs/TransformerResidual-S.toml \
  --benchmark bdg-2 borealis electricity ideal lcl sceaux smart \
  --use_temperature_input \
  --no_rc_loss \
  --run_suffix "${RUN_SUFFIX}" \
  > "${LOG_FILE}" 2>&1 &

# Get the process ID
PID=$!

echo "Training started in background!"
echo "Process ID: ${PID}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "To monitor progress, run:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To check if still running, run:"
echo "  ps -p ${PID}"
echo ""
echo "To stop training, run:"
echo "  kill ${PID}"
echo ""
echo "=========================================="

# Save PID to file for easy reference
echo "${PID}" > "logs/train_transformer_no_rc_${TIMESTAMP}.pid"
echo "PID saved to: logs/train_transformer_no_rc_${TIMESTAMP}.pid"

