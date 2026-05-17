#!/usr/bin/env bash
set -euo pipefail

export PATH=/root/.pixi/bin:${PATH}
export LD_LIBRARY_PATH=/root/autodl-tmp/SpAtten/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}
unset OMP_NUM_THREADS
cd /root/autodl-tmp/SpAtten

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="artifacts/full_experiment_${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"

echo "=============================================="
echo "Full SpAtten Experiment Suite"
echo "Experiment: Perplexity + Throughput Analysis"
echo "Timestamp: ${TIMESTAMP}"
echo "Output: ${RESULT_DIR}"
echo "=============================================="

# ========== Experiment 1: Perplexity Baseline ==========
echo ""
echo "============================================"
echo "EXP-1: Perplexity - Dense vs All Variants (no head pruning)"
echo "============================================"
pixi run python -m bench_kernel.perplexity_benchmark_qwen3   --model-name /root/autodl-tmp/models/Qwen3-0.6B-ms   --seq-lens 512,1024,2048,4096   --max-eval-chunks 200   --modes dense,quant_only,v_prune_only,full   --output-dir "${RESULT_DIR}/perplexity_baseline" 2>&1 | tee "${RESULT_DIR}/perplexity_baseline.log"
echo "EXP-1 completed."

# ========== Experiment 2: Perplexity - Threshold Sweep ==========
echo ""
echo "============================================"
echo "EXP-2: Perplexity - Threshold Sensitivity"
echo "============================================"
pixi run python -m bench_kernel.perplexity_benchmark_qwen3   --model-name /root/autodl-tmp/models/Qwen3-0.6B-ms   --seq-lens 1024,2048,4096   --max-eval-chunks 150   --modes quant_only,v_prune_only,full   --quant-thresholds 0.005,0.01,0.02,0.05   --v-thresholds 0.03,0.05,0.10,0.20   --output-dir "${RESULT_DIR}/perplexity_threshold_sweep" 2>&1 | tee "${RESULT_DIR}/perplexity_threshold_sweep.log"
echo "EXP-2 completed."

# ========== Experiment 3: Perplexity with Head Pruning ==========
echo ""
echo "============================================"
echo "EXP-3: Perplexity - Head Pruning Impact"
echo "============================================"
for hpn in 1 2 4; do
  for hpi in 1 2; do
    echo "--- head_prune_num=${hpn}, head_prune_interval=${hpi} ---"
    pixi run python -m bench_kernel.perplexity_benchmark_qwen3       --model-name /root/autodl-tmp/models/Qwen3-0.6B-ms       --seq-lens 1024,2048,4096       --max-eval-chunks 100       --modes dense,quant_only,v_prune_only,full       --enable-head-prune       --head-prune-nums ${hpn}       --head-prune-intervals ${hpi}       --output-dir "${RESULT_DIR}/perplexity_headprune" 2>&1 | tail -5
  done
done
echo "EXP-3 completed."

# ========== Experiment 4: Throughput - Full Comparison ==========
echo ""
echo "============================================"
echo "EXP-4: Throughput - Dense vs All Variants"
echo "============================================"
pixi run python -m bench_kernel.throughput_benchmark_qwen3   --model-name /root/autodl-tmp/models/Qwen3-0.6B-ms   --seq-lens 1024,2048,4096,8192   --decode-steps 16   --modes dense,quant_only,v_prune_only,full   --output-dir "${RESULT_DIR}/throughput_baseline" 2>&1 | tee "${RESULT_DIR}/throughput_baseline.log"
echo "EXP-4 completed."

# ========== Experiment 5: Throughput with Head Pruning ==========
echo ""
echo "============================================"
echo "EXP-5: Throughput - Head Pruning Impact"
echo "============================================"
pixi run python -m bench_kernel.throughput_benchmark_qwen3   --model-name /root/autodl-tmp/models/Qwen3-0.6B-ms   --seq-lens 1024,2048,4096,8192   --decode-steps 16   --modes quant_only,v_prune_only,full   --enable-head-prune   --head-prune-num 1   --head-prune-interval 1   --output-dir "${RESULT_DIR}/throughput_headprune" 2>&1 | tee "${RESULT_DIR}/throughput_headprune.log"
echo "EXP-5 completed."

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results: ${RESULT_DIR}"
echo "=============================================="

# Generate summary
echo ""
echo "=== FILE LISTING ==="
find "${RESULT_DIR}" -name "*.json" -exec echo {} \;
