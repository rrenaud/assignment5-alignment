#!/bin/bash
# Quick parameter sweep with reduced search space
# This runs 12 configurations testing the most important hyperparameters

echo "=================================="
echo "Quick Multi-Objective Param Sweep"
echo "=================================="
echo ""
echo "Testing 3 LRs × 2 warmups × 2 grad accum = 12 configs"
echo "Estimated time: ~15-20 minutes"
echo ""

python3 param_sweep.py \
    --sweep-dir ./quick_sweep_results \
    --benchmark-data benchmark_dataset.jsonl \
    --max-configs 12

echo ""
echo "Sweep complete! Analyzing results..."
echo ""

python3 visualize_sweep.py --sweep-dir ./quick_sweep_results
