#!/bin/bash
# Phase 4: Rank Scaling
# 3 methods x 6 ranks x 2 tasks x 3 seeds = 108 runs

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KASA_DIR="${KASA_DIR:-$PROJECT_DIR/KaSA}"
LOG_DIR="$PROJECT_DIR/outputs/phase4"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi
KASA_PYTHONPATH="$KASA_DIR/peft/src:$PYTHONPATH"

SEEDS=(0 42 123)
TASKS=("cola" "rte")
RANKS=(1 2 4 8 16 32)

EPOCHS=100
BATCH=32
HEAD_LR=4e-4
MODULE_LR=4e-4

# Task-specific beta/gamma for KaSA
declare -A BETA
BETA[cola]=0.0001; BETA[rte]=0.24
declare -A GEMMA
GEMMA[cola]=0.001; GEMMA[rte]=0.00024

echo "=== Phase 4: Rank Scaling ==="
echo ""

for task in "${TASKS[@]}"; do
    for rank in "${RANKS[@]}"; do
        # alpha = 2 * rank (consistent ratio)
        alpha=$((rank * 2))

        for seed in "${SEEDS[@]}"; do

            # --- LoRA ---
            out="$LOG_DIR/lora_${task}_r${rank}_s${seed}"
            if [ ! -f "$out/results.json" ]; then
                echo "[$(date '+%H:%M:%S')] LoRA | $task | r=$rank | seed=$seed"
                CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                    --model_name_or_path roberta-base \
                    --task "$task" --init_lora_weights True \
                    --lora_r "$rank" --lora_alpha "$alpha" --lora_dropout 0.0 \
                    --head_lr $HEAD_LR --module_lr $MODULE_LR \
                    --num_epochs $EPOCHS --bs $BATCH --max_length 512 \
                    --weight_decay 0.0 --warmup_ratio 0.06 \
                    --seed "$seed" --output_dir "$out" \
                    2>&1 | tee "$LOG_DIR/lora_${task}_r${rank}_s${seed}.log"
            fi

            # --- PiSSA ---
            out="$LOG_DIR/pissa_${task}_r${rank}_s${seed}"
            if [ ! -f "$out/results.json" ]; then
                echo "[$(date '+%H:%M:%S')] PiSSA | $task | r=$rank | seed=$seed"
                CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                    --model_name_or_path roberta-base \
                    --task "$task" --init_lora_weights pissa \
                    --lora_r "$rank" --lora_alpha "$alpha" --lora_dropout 0.0 \
                    --head_lr $HEAD_LR --module_lr $MODULE_LR \
                    --num_epochs $EPOCHS --bs $BATCH --max_length 512 \
                    --weight_decay 0.0 --warmup_ratio 0.06 \
                    --seed "$seed" --output_dir "$out" \
                    2>&1 | tee "$LOG_DIR/pissa_${task}_r${rank}_s${seed}.log"
            fi

            # --- KaSA ---
            out="$LOG_DIR/kasa_${task}_r${rank}_s${seed}"
            if [ ! -f "$out/results.json" ]; then
                echo "[$(date '+%H:%M:%S')] KaSA | $task | r=$rank | seed=$seed"
                PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_kasa_fraction.py" \
                    --model_name_or_path roberta-base \
                    --task "$task" \
                    --lora_r "$rank" --lora_alpha "$alpha" --lora_dropout 0.0 \
                    --num_epochs $EPOCHS --bs $BATCH --max_length 512 \
                    --head_lr $HEAD_LR --module_lr $MODULE_LR \
                    --beta "${BETA[$task]}" --gemma "${GEMMA[$task]}" \
                    --weight_decay 0.0 --warmup_ratio 0.06 \
                    --seed "$seed" --output_dir "$out" \
                    2>&1 | tee "$LOG_DIR/kasa_${task}_r${rank}_s${seed}.log"
            fi

        done
    done
done

echo ""
echo "=== Phase 4 Complete ==="
echo "Total runs: 108"
