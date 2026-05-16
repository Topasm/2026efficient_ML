#!/bin/bash
# Priority 5: Learning rate sensitivity on CoLA
# 3 methods x 5 LRs x 3 seeds = 45 runs
# Tests: does KaSA survive LR misspecification better/worse than LoRA/PiSSA?

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KASA_DIR="${KASA_DIR:-$PROJECT_DIR/KaSA}"
LOG_DIR="$PROJECT_DIR/outputs/lr_sensitivity"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi
KASA_PYTHONPATH="$KASA_DIR/peft/src:$PYTHONPATH"

SEEDS=(0 42 123)
TASK="cola"
RANK=8
ALPHA=16
EPOCHS=50  # shorter for speed
BATCH=32

LRS=("1e-4" "3e-4" "4e-4" "1e-3" "3e-3")  # 4e-4 is paper default

echo "=== Priority 5: Learning Rate Sensitivity ==="

for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do

        # LoRA
        out="$LOG_DIR/lora_lr${lr}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] LoRA lr=$lr s=$seed"
            CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                --task "$TASK" --init_lora_weights True \
                --lora_r $RANK --lora_alpha $ALPHA \
                --head_lr "$lr" --module_lr "$lr" \
                --num_epochs $EPOCHS --bs $BATCH --max_length 512 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/lora_lr${lr}_s${seed}.log"
        fi

        # PiSSA
        out="$LOG_DIR/pissa_lr${lr}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] PiSSA lr=$lr s=$seed"
            CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                --task "$TASK" --init_lora_weights pissa \
                --lora_r $RANK --lora_alpha $ALPHA \
                --head_lr "$lr" --module_lr "$lr" \
                --num_epochs $EPOCHS --bs $BATCH --max_length 512 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/pissa_lr${lr}_s${seed}.log"
        fi

        # KaSA
        out="$LOG_DIR/kasa_lr${lr}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] KaSA lr=$lr s=$seed"
            PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python \
                "$PROJECT_DIR/scripts/train_kasa_fraction.py" \
                --task "$TASK" \
                --lora_r $RANK --lora_alpha $ALPHA \
                --head_lr "$lr" --module_lr "$lr" \
                --num_epochs $EPOCHS --bs $BATCH --max_length 512 \
                --beta 0.0001 --gemma 0.001 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/kasa_lr${lr}_s${seed}.log"
        fi
    done
done

echo "=== LR sensitivity complete: 45 runs ==="
