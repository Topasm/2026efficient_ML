#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs/kasa_mechanistic/full}"
PID_FILE="$ROOT/full_run.pid"

echo "=== Run ==="
if [ -f "$PID_FILE" ]; then
    pid="$(cat "$PID_FILE")"
    ps -p "$pid" -o pid,ppid,sid,stat,etime,pcpu,pmem,args || echo "PID $pid is not running"
else
    echo "No PID file at $PID_FILE"
fi

echo ""
echo "=== Counts ==="
printf "ablation results: "
find "$ROOT/exp3_weighted_rank_ablation" -maxdepth 3 -name results.json 2>/dev/null | wc -l
printf "ablation statuses: "
find "$ROOT/exp3_weighted_rank_ablation" -maxdepth 3 -name status.txt 2>/dev/null | wc -l
printf "update-frame results: "
find "$ROOT/exp2_update_frame" -maxdepth 2 -name results.json 2>/dev/null | wc -l
printf "gradient-alignment results: "
find "$ROOT/exp1_gradient_alignment" -maxdepth 2 -name results.json 2>/dev/null | wc -l
printf "svd-stability results: "
find "$ROOT/exp4_svd_stability" -maxdepth 2 -name results.json 2>/dev/null | wc -l

echo ""
echo "=== Recent Launcher Log ==="
if [ -f "$ROOT/launcher.log" ]; then
    tail -40 "$ROOT/launcher.log"
else
    echo "No launcher.log yet"
fi

echo ""
echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
        --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits || true
else
    echo "nvidia-smi not found"
fi

echo ""
echo "=== Summary Files ==="
find "$ROOT/summaries" -maxdepth 1 -type f 2>/dev/null | sort || true
