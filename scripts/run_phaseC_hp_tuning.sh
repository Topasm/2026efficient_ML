#!/bin/bash
# Phase C: Best-per-method LR on RTE, MRPC
# Tests whether "all methods ~1pt apart" holds at each method's best LR
#
# Best LRs from Priority 5 (CoLA sweep):
#   LoRA:  1e-3
#   PiSSA: 4e-4 (same as Phase 2 — no rerun needed)
#   KaSA:  1e-3
# 2 methods (LoRA, KaSA) × 2 tasks (RTE, MRPC) × 3 seeds = 12 runs

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KASA_DIR="${KASA_DIR:-$PROJECT_DIR/KaSA}"
LOG_DIR="$PROJECT_DIR/outputs/phaseC_hp_tuning"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi
KASA_PYTHONPATH="$KASA_DIR/peft/src:$PYTHONPATH"

SEEDS=(0 42 123)
TASKS=("rte" "mrpc")

declare -A BETA
BETA[rte]=0.24
BETA[mrpc]=0.1
declare -A GEMMA
GEMMA[rte]=0.00024
GEMMA[mrpc]=0.001

echo "=== Phase C: Best-per-method LR on RTE/MRPC ==="

for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do

        # LoRA at best LR=1e-3
        out="$LOG_DIR/lora_bestLR_${task}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] LoRA lr=1e-3 $task s=$seed"
            CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                --task "$task" --init_lora_weights True \
                --lora_r 8 --lora_alpha 16 \
                --head_lr 1e-3 --module_lr 1e-3 \
                --num_epochs 100 --bs 32 --max_length 512 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/lora_bestLR_${task}_s${seed}.log"
        fi

        # KaSA at best LR=1e-3
        out="$LOG_DIR/kasa_bestLR_${task}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] KaSA lr=1e-3 $task s=$seed"
            PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python \
                "$PROJECT_DIR/scripts/train_kasa_fraction.py" \
                --task "$task" \
                --lora_r 8 --lora_alpha 16 --lora_dropout 0.0 \
                --num_epochs 100 --bs 32 --max_length 512 \
                --head_lr 1e-3 --module_lr 1e-3 \
                --beta "${BETA[$task]}" --gemma "${GEMMA[$task]}" \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/kasa_bestLR_${task}_s${seed}.log"
        fi
    done
done

echo "=== Phase C complete: 12 runs ==="
