#!/bin/bash
# Test 1: KaSA-noaux (SVD + diag, β=γ=0) at 5 LRs
# Tests whether SVD+diag combo gives LR robustness without aux loss.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KASA_DIR="${KASA_DIR:-$PROJECT_DIR/KaSA}"
LOG_DIR="$PROJECT_DIR/outputs/kasa_noaux_lr"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi
KASA_PYTHONPATH="$KASA_DIR/peft/src:$PYTHONPATH"

SEEDS=(0 42 123)
LRS=("1e-4" "3e-4" "4e-4" "1e-3" "3e-3")

echo "=== KaSA-noaux LR sweep (β=γ=0, SVD+diag only) ==="

for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="$LOG_DIR/kasanoaux_lr${lr}_s${seed}"
        if [ -f "$out/results.json" ]; then
            echo "[SKIP] lr=$lr s=$seed"
            continue
        fi
        echo "[$(date '+%H:%M:%S')] KaSA-noaux lr=$lr s=$seed"
        PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python \
            "$PROJECT_DIR/scripts/train_kasa_fraction.py" \
            --task cola \
            --lora_r 8 --lora_alpha 16 --lora_dropout 0.0 \
            --num_epochs 50 --bs 32 --max_length 512 \
            --head_lr "$lr" --module_lr "$lr" \
            --beta 0 --gemma 0 \
            --weight_decay 0.0 --warmup_ratio 0.06 \
            --seed "$seed" --output_dir "$out" \
            2>&1 | tee "$LOG_DIR/kasanoaux_lr${lr}_s${seed}.log"
    done
done

echo "=== Complete: 15 runs ==="
