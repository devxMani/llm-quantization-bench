# LLM Quantization Bench

**What do W4A16 and FP8 quantization actually save, and what do they cost, when you serve an LLM with vLLM?**

This repo quantizes one open model three ways (FP16 baseline, FP8, W4A16), serves each with [vLLM](https://github.com/vllm-project/vllm), and measures checkpoint size, throughput at several concurrency levels, and MMLU accuracy, all on the same GPU. The goal is a small, reproducible answer to a practical question: *which precision should I deploy?*

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![vLLM](https://img.shields.io/badge/serving-vLLM-green) ![llm-compressor](https://img.shields.io/badge/quantization-llm--compressor-orange) ![Status](https://img.shields.io/badge/results-pending%20first%20run-lightgrey)

---

## Results

> **Not run yet.** The table below is filled in by `make_table.py` after you run the steps in [Reproduce](#reproduce). Replace the dashes with your measured numbers, and record the GPU and library versions next to them. Do not publish numbers you did not measure.

| Precision | Checkpoint (GB) | MMLU 5-shot (%) | tok/s @ 1 | tok/s @ 4 | tok/s @ 16 | tok/s @ 64 |
|---|---|---|---|---|---|---|
| FP16 (baseline) | - | - | - | - | - | - |
| FP8 (W8A8 dynamic) | - | - | - | - | - | - |
| W4A16 (GPTQ) | - | - | - | - | - | - |

GPU: _fill in_ · vLLM: _fill in_ · llm-compressor: _fill in_ · model: `Qwen/Qwen2.5-7B-Instruct`

---

## What each precision means

| Scheme | Weights | Activations | Method | Needs calibration data |
|---|---|---|---|---|
| FP16 | 16-bit | 16-bit | none (baseline) | no |
| FP8 (`FP8_DYNAMIC`) | 8-bit float | 8-bit float, dynamic per token | post-training quantization | no |
| W4A16 | 4-bit int, group size 128 | 16-bit | GPTQ | yes (512 samples of UltraChat) |

Going from 16-bit to 8-bit halves weight storage and going to 4-bit cuts it to a quarter, so expect roughly 2x and 4x smaller checkpoints. The measured sizes in the table are what count, because embeddings and the output head stay in 16-bit.

---

## Method

1. **Quantize** with [llm-compressor](https://github.com/vllm-project/llm-compressor). Scripts follow the library's official examples. The `lm_head` stays in 16-bit.
2. **Serve and benchmark** each model with vLLM's offline engine. `bench_vllm.py` submits N identical-length prompts (default 512 input tokens, 128 output tokens, forced to full length) at concurrency 1, 4, 16, and 64, repeats each 3 times, and reports mean tokens per second. Prefix caching is off so identical prompts do not inflate throughput.
3. **Evaluate accuracy** with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on MMLU (5-shot) using the vLLM backend.
4. **Report** with `make_table.py`, which collects everything under `results/` into one markdown table.

---

## Reproduce

**Hardware:** one GPU with enough memory for a 7B model in FP16 (about 16 GB or more). FP8 compute needs an FP8-capable GPU (Ada or Hopper, for example L4, RTX 4090, H100); results on other GPUs may differ.

```bash
git clone https://github.com/devxMani/llm-quantization-bench.git
cd llm-quantization-bench
pip install -r requirements.txt      # use two environments if pins conflict (see file)

# 1. Quantize
python quantize_fp8.py   --model Qwen/Qwen2.5-7B-Instruct
python quantize_w4a16.py --model Qwen/Qwen2.5-7B-Instruct

# 2. Throughput and size, one run per precision
python bench_vllm.py --model Qwen/Qwen2.5-7B-Instruct        --label fp16
python bench_vllm.py --model Qwen2.5-7B-Instruct-FP8-Dynamic --label fp8
python bench_vllm.py --model Qwen2.5-7B-Instruct-W4A16-G128  --label w4a16

# 3. Accuracy
./run_mmlu.sh Qwen/Qwen2.5-7B-Instruct        fp16
./run_mmlu.sh Qwen2.5-7B-Instruct-FP8-Dynamic fp8
./run_mmlu.sh Qwen2.5-7B-Instruct-W4A16-G128  w4a16

# 4. Table for this README
python make_table.py
```

Free Colab GPUs are too small for a 7B FP16 run. Use a smaller model (for example `--model Qwen/Qwen2.5-1.5B-Instruct`) or rent a GPU for an hour.

---

## Repository layout

```
quantize_w4a16.py   GPTQ 4-bit weight quantization (llm-compressor)
quantize_fp8.py     FP8 dynamic quantization (llm-compressor)
bench_vllm.py       throughput at multiple concurrency levels + checkpoint size
run_mmlu.sh         MMLU 5-shot through lm-evaluation-harness (vLLM backend)
make_table.py       results/ -> markdown table
results/            JSON outputs from your runs
```

---

## Roadmap

- [ ] First full run and published results table with hardware and versions
- [ ] Second model family and a larger model size
- [ ] W8A8 INT8 (SmoothQuant + GPTQ) alongside FP8
- [ ] KV-cache quantization and its effect on long-context throughput
- [ ] More accuracy tasks than MMLU (for example GSM8K and a code benchmark)
- [ ] Cost per million tokens from measured throughput and GPU price
- [ ] Latency percentiles (p50 / p95) from a served endpoint, not only offline batches

---

## Acknowledgements

Built on [llm-compressor](https://github.com/vllm-project/llm-compressor) (Apache 2.0), [vLLM](https://github.com/vllm-project/vllm), and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Quantization scripts follow the examples in the llm-compressor repository.

## License

Add a license (MIT or Apache 2.0).
