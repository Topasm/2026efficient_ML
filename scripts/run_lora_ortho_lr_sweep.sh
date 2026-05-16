#!/bin/bash
# LoRA + orthogonality penalty, LR sensitivity
# 5 LRs x 3 seeds = 15 runs. CoLA, 50 epochs.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/outputs/lora_ortho_lr"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi

SEEDS=(0 42 123)
LRS=("1e-4" "3e-4" "4e-4" "1e-3" "3e-3")

echo "=== LoRA + orthogonality penalty — LR sweep (CoLA, 50 epochs) ==="

for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="$LOG_DIR/loraortho_lr${lr}_s${seed}"
        if [ -f "$out/results.json" ]; then
            echo "[SKIP] lr=$lr s=$seed"
            continue
        fi
        echo "[$(date '+%H:%M:%S')] LoRA+ortho lr=$lr s=$seed"
        CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_lora_ortho.py" \
            --task cola --seed "$seed" --num_epochs 50 \
            --lora_r 8 --lora_alpha 16 \
            --head_lr "$lr" --module_lr "$lr" \
            --bs 32 --max_length 512 --warmup_ratio 0.06 \
            --gamma 1e-3 \
            --output_dir "$out" \
            2>&1 | tee "$LOG_DIR/loraortho_lr${lr}_s${seed}.log"
    done
done

echo "=== Complete: 15 runs ==="
