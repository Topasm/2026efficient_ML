#!/bin/bash
# Phase 2: Head-to-head comparison
# 4 methods x 3 tasks x 3 seeds = 36 runs
# All r=8, identical training conditions

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KASA_DIR="${KASA_DIR:-$PROJECT_DIR/KaSA}"
LOG_DIR="$PROJECT_DIR/outputs/phase2"
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

# Common hyperparams (from KaSA paper)
RANK=8
ALPHA=16
EPOCHS=100
BATCH=32
HEAD_LR=4e-4
MODULE_LR=4e-4

# Task-specific beta/gamma for KaSA
declare -A BETA
BETA[cola]=0.0001; BETA[rte]=0.24; BETA[mrpc]=0.1
declare -A GEMMA
GEMMA[cola]=0.001; GEMMA[rte]=0.00024; GEMMA[mrpc]=0.001

echo "=== Phase 2: Head-to-head Comparison ==="
echo "Methods: LoRA, PiSSA, EVA, KaSA"
echo ""

# --- Part A: LoRA / PiSSA / EVA via HF PEFT ---
# NOTE: CorDA excluded — preprocess_corda hook incompatible with RoBERTa 3D input in PEFT 0.19.0
METHODS_HF=("lora:True" "pissa:pissa" "eva:eva")

for method_pair in "${METHODS_HF[@]}"; do
    IFS=':' read -r method_name init_weights <<< "$method_pair"

    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            out="$LOG_DIR/${method_name}_${task}_seed${seed}"
            if [ -f "$out/results.json" ]; then
                echo "[SKIP] $method_name | $task | seed=$seed (already done)"
                continue
            fi

            echo "[$(date '+%H:%M:%S')] Running $method_name | task=$task | seed=$seed"

            CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                --model_name_or_path roberta-base \
                --task "$task" \
                --init_lora_weights "$init_weights" \
                --lora_r $RANK \
                --lora_alpha $ALPHA \
                --lora_dropout 0.0 \
                --head_lr $HEAD_LR \
                --module_lr $MODULE_LR \
                --num_epochs $EPOCHS \
                --bs $BATCH \
                --max_length 512 \
                --weight_decay 0.0 \
                --warmup_ratio 0.06 \
                --seed "$seed" \
                --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/${method_name}_${task}_seed${seed}.log"

            echo "[$(date '+%H:%M:%S')] Done: $method_name $task seed=$seed"
        done
    done
done

# --- Part B: KaSA via its own repo ---
for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "[$(date '+%H:%M:%S')] Running KaSA | task=$task | seed=$seed"

        cd "$KASA_DIR"
        PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python main.py \
            --model_name_or_path roberta-base \
            --dataset "$task" \
            --task "$task" \
            --peft kasa \
            --lora_r $RANK \
            --lora_alpha $ALPHA \
            --lora_dropout 0.0 \
            --num_epochs $EPOCHS \
            --bs $BATCH \
            --max_length 512 \
            --head_lr $HEAD_LR \
            --module_lr $MODULE_LR \
            --beta "${BETA[$task]}" \
            --gemma "${GEMMA[$task]}" \
            --weight_decay 0.0 \
            --seed "$seed" \
            2>&1 | tee "$LOG_DIR/kasa_${task}_seed${seed}.log"

        echo "[$(date '+%H:%M:%S')] Done: KaSA $task seed=$seed"
    done
done

echo ""
echo "=== Phase 2 Complete ==="
echo "Total runs: 36"
