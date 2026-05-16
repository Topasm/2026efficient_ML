#!/bin/bash
# Priority 1: KaSA ablation — β=γ=0 (disable auxiliary loss)
# Tests whether KaSA's auxiliary loss contributes anything.
# 3 tasks x 3 seeds = 9 runs

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KASA_DIR="${KASA_DIR:-$PROJECT_DIR/KaSA}"
LOG_DIR="$PROJECT_DIR/outputs/ablation_noaux"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi
KASA_PYTHONPATH="$KASA_DIR/peft/src:$PYTHONPATH"

SEEDS=(0 42 123)
TASKS=("cola" "rte" "mrpc")
RANK=8
ALPHA=16

echo "=== Ablation: KaSA with β=γ=0 (no auxiliary loss) ==="

for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="$LOG_DIR/noaux_${task}_s${seed}"
        if [ -f "$out/results.json" ]; then
            echo "[SKIP] $task s=$seed"
            continue
        fi
        echo "[$(date '+%H:%M:%S')] KaSA-noaux | $task | seed=$seed"
        PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python \
            "$PROJECT_DIR/scripts/train_kasa_fraction.py" \
            --task "$task" \
            --lora_r $RANK --lora_alpha $ALPHA --lora_dropout 0.0 \
            --num_epochs 100 --bs 32 --max_length 512 \
            --head_lr 4e-4 --module_lr 4e-4 \
            --beta 0 --gemma 0 \
            --weight_decay 0.0 --warmup_ratio 0.06 \
            --seed "$seed" --output_dir "$out" \
            2>&1 | tee "$LOG_DIR/noaux_${task}_s${seed}.log"
    done
done

echo "=== Ablation complete: 9 runs ==="
