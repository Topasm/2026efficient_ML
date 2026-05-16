# Efficient ML Experiment Scripts

Source code for running PEFT adaptation experiments with LoRA, PiSSA, EVA, and KaSA-style variants.

This repository tracks code only. Generated outputs are written under `outputs/` and ignored by git.

## KaSA Mechanistic Experiments

The current mechanistic study asks whether KaSA's gain comes from knowledge-aware SVD or from rank-wise weighted LoRA.

Prepare GLUE data/model caches:

```bash
uv run python scripts/prepare_mechanistic_experiments.py \
  --model_name_or_path roberta-base \
  --tasks cola,rte,mrpc,sst2 \
  --output_dir outputs/kasa_mechanistic/prepare
```

Run the fast weighted-rank ablation grid:

```bash
TASKS="cola rte mrpc sst2" \
LRS="3e-4 3e-3" \
SEEDS="0" \
EPOCHS=10 \
bash scripts/run_mechanistic_fast_ablation.sh
```

Run the full report-data collection in the background:

```bash
setsid env TASKS="cola rte mrpc sst2" LRS="1e-4 3e-4 1e-3 3e-3" \
  SEEDS="0 42 123" EPOCHS=20 MAX_JOBS=4 OUT_ROOT="outputs/kasa_mechanistic/full" \
  bash scripts/run_mechanistic_full_experiments.sh \
  > outputs/kasa_mechanistic/full/launcher.log 2>&1 < /dev/null &
```

Check progress:

```bash
scripts/check_mechanistic_run_status.sh outputs/kasa_mechanistic/full
```

## Quick Setup

```bash
uv sync
source .venv/bin/activate
```

Or run `bash scripts/setup_uv.sh`.

For VLM dependencies:

```bash
uv sync --extra vlm
```

For VLM setup through the helper script, run `bash scripts/setup_uv.sh --vlm`.

See [SETUP.md](SETUP.md) for CUDA wheel notes, smoke tests, and KaSA setup details.

KaSA runs require a local KaSA checkout that includes the modified PEFT fork:

```bash
git clone https://github.com/juyongjiang/KaSA.git KaSA
```

Alternatively, set `KASA_DIR` before running scripts:

```bash
export KASA_DIR=/path/to/KaSA
```

## Examples

```bash
bash scripts/run_phase2_headtohead.sh
bash scripts/run_phase3_data_scaling.sh
bash scripts/run_lr_sensitivity.sh
```

You can override the virtualenv path with:

```bash
export VENV=/path/to/venv/bin/activate
```
