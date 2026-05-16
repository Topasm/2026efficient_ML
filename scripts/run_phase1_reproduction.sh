#!/bin/bash
# Phase 1: Reproduction of KaSA paper Table 1
# Model: RoBERTa-base, Tasks: CoLA/RTE/MRPC, r=8, 100 epochs, batch 32
# 3 seeds per task

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KASA_DIR="${KASA_DIR:-$PROJECT_DIR/KaSA}"
LOG_DIR="$PROJECT_DIR/outputs/phase1"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi
# KaSA needs its own PEFT fork; baselines use standard PEFT
KASA_PYTHONPATH="$KASA_DIR/peft/src:$PYTHONPATH"

SEEDS=(0 42 123)

# Task-specific beta/gamma from official run scripts
declare -A BETA
BETA[cola]=0.0001
BETA[rte]=0.24
BETA[mrpc]=0.1

declare -A GEMMA
GEMMA[cola]=0.001
GEMMA[rte]=0.00024
GEMMA[mrpc]=0.001

TASKS=("cola" "rte" "mrpc")

echo "=== Phase 1: KaSA Reproduction ==="
echo "KaSA directory: $KASA_DIR"
echo "Logs: $LOG_DIR"
echo ""

for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "[$(date '+%H:%M:%S')] Running KaSA | task=$task | seed=$seed | beta=${BETA[$task]} | gamma=${GEMMA[$task]}"

        cd "$KASA_DIR"
        PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python main.py \
            --model_name_or_path roberta-base \
            --dataset "$task" \
            --task "$task" \
            --peft kasa \
            --lora_r 8 \
            --lora_alpha 16 \
            --lora_dropout 0.0 \
            --num_epochs 100 \
            --bs 32 \
            --max_length 512 \
            --head_lr 4e-4 \
            --module_lr 4e-4 \
            --beta "${BETA[$task]}" \
            --gemma "${GEMMA[$task]}" \
            --weight_decay 0.0 \
            --seed "$seed" \
            2>&1 | tee "$LOG_DIR/${task}_seed${seed}.log"

        echo "[$(date '+%H:%M:%S')] Done: $task seed=$seed"
        echo ""
    done
done

echo "=== Phase 1 Complete ==="
echo "Check outputs in: $LOG_DIR"
