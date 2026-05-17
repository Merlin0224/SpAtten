#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/SpAtten

export PATH=/root/.pixi/bin:${PATH}
export LD_LIBRARY_PATH=/root/autodl-tmp/SpAtten/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}
unset OMP_NUM_THREADS

COMMON_ARGS=(
  -m bench_kernel.model_seq_generate_compare_qwen3_spatten_exp_cachepath
  --seq-lens 2048,4096,8192
  --warmup 5
  --iters 10
  --decode-steps 16
  --model-name /root/autodl-tmp/models/Qwen3-0.6B-ms
  --enable-head-prune
  --head-prune-num 1
  --head-prune-interval 1
)

echo "===================="
echo "1) Baseline: no prefill fast path, KV window=2048, prefix=0"
echo "===================="
pixi run python "${COMMON_ARGS[@]}" --kv-window-size 2048 --kv-prefix-keep 0

echo "===================="
echo "2) Prefill SDPA fast path: max_seq_len=2048, KV window=2048, prefix=0"
echo "===================="
pixi run python "${COMMON_ARGS[@]}" \
  --enable-prefill-sdpa-fast-path \
  --prefill-sdpa-max-seq-len 2048 \
  --kv-window-size 2048 \
  --kv-prefix-keep 0

echo "===================="
echo "3) Aggressive KV: window=1024, prefix=0"
echo "===================="
pixi run python "${COMMON_ARGS[@]}" --kv-window-size 1024 --kv-prefix-keep 0

echo "===================="
echo "4) Conservative KV: window=2048, prefix=128"
echo "===================="
pixi run python "${COMMON_ARGS[@]}" --kv-window-size 2048 --kv-prefix-keep 128

echo "All four runs completed."
