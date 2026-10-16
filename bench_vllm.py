"""Measure vLLM throughput for one model at several concurrency levels.

Offline batch benchmark: N identical-length prompts are submitted at once and the
wall-clock time for all of them to finish is measured. Prefix caching is disabled
so identical prompts do not inflate the numbers.

Usage:
  python bench_vllm.py --model Qwen/Qwen2.5-7B-Instruct          --label fp16
  python bench_vllm.py --model Qwen2.5-7B-Instruct-FP8-Dynamic   --label fp8
  python bench_vllm.py --model Qwen2.5-7B-Instruct-W4A16-G128    --label w4a16
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def checkpoint_size_gb(path: str):
    """Size of the weights on disk (only for local directories)."""
    p = Path(path)
    if not p.is_dir():
        return None
    total = sum(f.stat().st_size for f in p.glob("*.safetensors"))
    return round(total / 1e9, 3) if total else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="e.g. fp16 | fp8 | w4a16")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=False,
    )
    tok = llm.get_tokenizer()
    ids = tok.encode("The history of computing spans many decades. " * 400)[: args.input_len]
    prompt = tok.decode(ids)
    params = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)

    llm.generate([prompt], params)  # warm-up

    rows = []
    for c in args.concurrency:
        tps, walls = [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            outs = llm.generate([prompt] * c, params)
            wall = time.perf_counter() - t0
            generated = sum(len(o.outputs[0].token_ids) for o in outs)
            tps.append(generated / wall)
            walls.append(wall)
        rows.append(
            {
                "concurrency": c,
                "tokens_per_s": round(statistics.mean(tps), 1),
                "batch_latency_s": round(statistics.mean(walls), 3),
            }
        )
        print(f"[{args.label}] concurrency={c:>3}  {rows[-1]['tokens_per_s']} tok/s  "
              f"{rows[-1]['batch_latency_s']} s per batch")

    try:
        import torch
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = None

    result = {
        "label": args.label,
        "model": args.model,
        "checkpoint_size_gb": checkpoint_size_gb(args.model),
        "gpu": gpu,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "repeats": args.repeats,
        "rows": rows,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = Path(args.out_dir) / f"bench_{args.label}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
