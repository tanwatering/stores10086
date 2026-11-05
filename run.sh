#!/usr/bin/env bash
set -e
MODEL="${MODEL:-Qwen/Qwen3-8B-Instruct}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
DTYPE="${DTYPE:-auto}"
TP="${TP:-1}"
EXTRA="${EXTRA:-}"
echo "MODEL=$MODEL PORT=$PORT MAXLEN=$MAXLEN"
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype "$DTYPE" \
  --tensor-parallel-size "$TP" \
  $EXTRA
