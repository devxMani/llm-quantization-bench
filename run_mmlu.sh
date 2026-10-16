#!/usr/bin/env bash
# Evaluate MMLU (5-shot) with lm-evaluation-harness on the vLLM backend.
# Usage: ./run_mmlu.sh <model-path-or-hf-id> <label>
set -euo pipefail
MODEL="${1:?model path or HF id}"
LABEL="${2:?label, e.g. fp16 | fp8 | w4a16}"

lm_eval --model vllm \
  --model_args "pretrained=${MODEL},dtype=auto,max_model_len=4096,gpu_memory_utilization=0.85" \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path "results/mmlu_${LABEL}"
