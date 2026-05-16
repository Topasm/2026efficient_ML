#!/bin/bash
# Extended LR stress test: find KaSA's breakpoint.
# LRs: 1e-2, 3e-2, 1e-1 (going well past what any sane person would use)
# Methods: LoRA (crashes already at 3e-3), KaSA-full, KaSA-noaux
# 3 x 3 x 3 = 27 runs

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KASA_DIR="${KASA_DIR:-$PROJECT_DIR/KaSA}"
LOG_DIR="$PROJECT_DIR/outputs/lr_extreme"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi
KASA_PYTHONPATH="$KASA_DIR/peft/src:$PYTHONPATH"

SEEDS=(0 42 123)
LRS=("1e-2" "3e-2" "1e-1")

echo "=== Extended LR stress test (until KaSA breaks) ==="

for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do

        # LoRA
        out="$LOG_DIR/lora_lr${lr}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] LoRA lr=$lr s=$seed"
            CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                --task cola --init_lora_weights True \
                --lora_r 8 --lora_alpha 16 \
                --head_lr "$lr" --module_lr "$lr" \
                --num_epochs 50 --bs 32 --max_length 512 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/lora_lr${lr}_s${seed}.log"
        fi

        # KaSA-full
        out="$LOG_DIR/kasa_lr${lr}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] KaSA-full lr=$lr s=$seed"
            PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python \
                "$PROJECT_DIR/scripts/train_kasa_fraction.py" \
                --task cola \
                --lora_r 8 --lora_alpha 16 --lora_dropout 0.0 \
                --num_epochs 50 --bs 32 --max_length 512 \
                --head_lr "$lr" --module_lr "$lr" \
                --beta 0.0001 --gemma 0.001 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/kasa_lr${lr}_s${seed}.log"
        fi

        # KaSA-noaux (SVD + diag, no aux loss)
        out="$LOG_DIR/kasanoaux_lr${lr}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
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
        fi
    done
done

echo "=== Extended LR stress test complete: 27 runs ==="
