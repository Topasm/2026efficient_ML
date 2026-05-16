#!/bin/bash
# Phase B: RoBERTa-large scaling test
# B.1 head-to-head on CoLA/RTE/MRPC (27 runs)
# B.2 LR sensitivity on CoLA (45 runs)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KASA_DIR="${KASA_DIR:-$PROJECT_DIR/KaSA}"
LOG_DIR="$PROJECT_DIR/outputs/phaseB_roberta_large"
VENV="${VENV:-$PROJECT_DIR/.venv/bin/activate}"
mkdir -p "$LOG_DIR"

if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "[WARN] Virtualenv not found at $VENV; using current Python environment."
fi
KASA_PYTHONPATH="$KASA_DIR/peft/src:$PYTHONPATH"

MODEL="roberta-large"
SEEDS=(0 42 123)

declare -A BETA
BETA[cola]=0.0001; BETA[rte]=0.24; BETA[mrpc]=0.1
declare -A GEMMA
GEMMA[cola]=0.001; GEMMA[rte]=0.00024; GEMMA[mrpc]=0.001

# -----------------------------
# B.1: Head-to-head at r=8
# -----------------------------
echo "=== Phase B.1: RoBERTa-large head-to-head ==="

for task in cola rte mrpc; do
    for seed in "${SEEDS[@]}"; do

        out="$LOG_DIR/b1_lora_${task}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] LoRA-large $task s=$seed"
            CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                --model_name_or_path "$MODEL" \
                --task "$task" --init_lora_weights True \
                --lora_r 8 --lora_alpha 16 \
                --head_lr 4e-4 --module_lr 4e-4 \
                --num_epochs 100 --bs 32 --max_length 512 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/b1_lora_${task}_s${seed}.log"
        fi

        out="$LOG_DIR/b1_pissa_${task}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] PiSSA-large $task s=$seed"
            CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                --model_name_or_path "$MODEL" \
                --task "$task" --init_lora_weights pissa \
                --lora_r 8 --lora_alpha 16 \
                --head_lr 4e-4 --module_lr 4e-4 \
                --num_epochs 100 --bs 32 --max_length 512 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/b1_pissa_${task}_s${seed}.log"
        fi

        out="$LOG_DIR/b1_kasa_${task}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] KaSA-large $task s=$seed"
            PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python \
                "$PROJECT_DIR/scripts/train_kasa_fraction.py" \
                --model_name_or_path "$MODEL" \
                --task "$task" \
                --lora_r 8 --lora_alpha 16 --lora_dropout 0.0 \
                --num_epochs 100 --bs 32 --max_length 512 \
                --head_lr 4e-4 --module_lr 4e-4 \
                --beta "${BETA[$task]}" --gemma "${GEMMA[$task]}" \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/b1_kasa_${task}_s${seed}.log"
        fi
    done
done

# -----------------------------
# B.2: LR sensitivity on CoLA
# -----------------------------
echo ""
echo "=== Phase B.2: RoBERTa-large LR sensitivity (CoLA) ==="

LRS=("1e-4" "3e-4" "4e-4" "1e-3" "3e-3")

for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do

        out="$LOG_DIR/b2_lora_lr${lr}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] B.2 LoRA-large lr=$lr s=$seed"
            CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                --model_name_or_path "$MODEL" \
                --task cola --init_lora_weights True \
                --lora_r 8 --lora_alpha 16 \
                --head_lr "$lr" --module_lr "$lr" \
                --num_epochs 50 --bs 32 --max_length 512 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/b2_lora_lr${lr}_s${seed}.log"
        fi

        out="$LOG_DIR/b2_pissa_lr${lr}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] B.2 PiSSA-large lr=$lr s=$seed"
            CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/train_hf_peft.py" \
                --model_name_or_path "$MODEL" \
                --task cola --init_lora_weights pissa \
                --lora_r 8 --lora_alpha 16 \
                --head_lr "$lr" --module_lr "$lr" \
                --num_epochs 50 --bs 32 --max_length 512 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/b2_pissa_lr${lr}_s${seed}.log"
        fi

        out="$LOG_DIR/b2_kasa_lr${lr}_s${seed}"
        if [ ! -f "$out/results.json" ]; then
            echo "[$(date '+%H:%M:%S')] B.2 KaSA-large lr=$lr s=$seed"
            PYTHONPATH="$KASA_PYTHONPATH" CUDA_VISIBLE_DEVICES=0 python \
                "$PROJECT_DIR/scripts/train_kasa_fraction.py" \
                --model_name_or_path "$MODEL" \
                --task cola \
                --lora_r 8 --lora_alpha 16 --lora_dropout 0.0 \
                --num_epochs 50 --bs 32 --max_length 512 \
                --head_lr "$lr" --module_lr "$lr" \
                --beta 0.0001 --gemma 0.001 \
                --weight_decay 0.0 --warmup_ratio 0.06 \
                --seed "$seed" --output_dir "$out" \
                2>&1 | tee "$LOG_DIR/b2_kasa_lr${lr}_s${seed}.log"
        fi
    done
done

echo "=== Phase B complete: 72 runs ==="
