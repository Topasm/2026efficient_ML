# Efficient ML Experiment Scripts

Source code for running PEFT adaptation experiments with LoRA, PiSSA, EVA, and KaSA-style variants.

This repository tracks code only. Generated outputs are written under `outputs/` and ignored by git.

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
