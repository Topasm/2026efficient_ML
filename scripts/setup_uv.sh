#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

SYNC_ARGS=()
if [ "${1:-}" = "--vlm" ]; then
    SYNC_ARGS=(--extra vlm)
fi

uv sync "${SYNC_ARGS[@]}"

echo ""
echo "Environment ready."
echo "Activate with: source .venv/bin/activate"

if [ ! -d "$PROJECT_DIR/KaSA" ]; then
    echo ""
    echo "For KaSA runs, clone KaSA here or set KASA_DIR:"
    echo "  git clone https://github.com/juyongjiang/KaSA.git KaSA"
    echo "  export KASA_DIR=/path/to/KaSA"
fi
