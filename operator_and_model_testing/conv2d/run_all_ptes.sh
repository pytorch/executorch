#!/bin/bash

PTE_DIR="/home/sraut/ext_main/cad_rlc/executorch/operator_and_model_testing/conv2d/pte"
LOG_DIR="/home/sraut/ext_main/cad_rlc/executorch/operator_and_model_testing/conv2d/logs"

mkdir -p "$LOG_DIR"

# Run with cmake-out-generic/executor_runner
echo "############################################"
echo "# Running with cmake-out-generic/executor_runner"
echo "############################################"
for pte in "$PTE_DIR"/*.pte; do
    base=$(basename "$pte" .pte)
    log_file="${LOG_DIR}/${base}_generic.log"
    echo "===== Running (generic): $base ====="
    xt-run --turbo cmake-out-generic/executor_runner --model_path="$pte" 2>&1 | tee "$log_file"
    echo ""
done

# Run with cmake-out/executor_runner
echo "############################################"
echo "# Running with cmake-out/executor_runner"
echo "############################################"
for pte in "$PTE_DIR"/*.pte; do
    base=$(basename "$pte" .pte)
    log_file="${LOG_DIR}/${base}_opt.log"
    echo "===== Running (opt): $base ====="
    xt-run --turbo cmake-out/executor_runner --model_path="$pte" 2>&1 | tee "$log_file"
    echo ""
done

echo "All done. Logs saved in $LOG_DIR"
