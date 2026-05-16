#!/bin/bash
# Test 2: SVD-only LR sweep
# KaSA's SVD surgery on base weights + standard LoRA, no diag, no aux.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/outputs/svd_only_lr"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi

SEEDS=(0 42 123)
LRS=("1e-4" "3e-4" "4e-4" "1e-3" "3e-3")

echo "=== SVD-only LR sweep (SVD surgery + standard LoRA) ==="

for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="$LOG_DIR/svdonly_lr${lr}_s${seed}"
        if [ -f "$out/results.json" ]; then
            echo "[SKIP] lr=$lr s=$seed"
            continue
        fi
        echo "[$(date '+%H:%M:%S')] SVD-only lr=$lr s=$seed"
        CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_svd_only.py" \
            --task cola --seed "$seed" --num_epochs 50 \
            --lora_r 8 --lora_alpha 16 \
            --head_lr "$lr" --module_lr "$lr" \
            --bs 32 --max_length 512 --warmup_ratio 0.06 \
            --output_dir "$out" \
            2>&1 | tee "$LOG_DIR/svdonly_lr${lr}_s${seed}.log"
    done
done

echo "=== Complete: 15 runs ==="
