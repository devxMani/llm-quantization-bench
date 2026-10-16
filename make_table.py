"""Collect results/ into one markdown table you can paste into the README.

Run after bench_vllm.py and run_mmlu.sh for each label (fp16, fp8, w4a16).
Cells stay '-' for anything you have not measured yet.
"""
import glob
import json
from pathlib import Path

LABELS = ["fp16", "fp8", "w4a16"]
NAMES = {"fp16": "FP16 (baseline)", "fp8": "FP8 (W8A8 dynamic)", "w4a16": "W4A16 (GPTQ)"}
RES = Path("results")


def mmlu(label):
    files = sorted(glob.glob(str(RES / f"mmlu_{label}" / "**" / "results_*.json"), recursive=True))
    if not files:
        return None
    d = json.loads(Path(files[-1]).read_text())["results"]
    block = d.get("mmlu") or next((v for k, v in d.items() if k.startswith("mmlu")), None)
    if not block or "acc,none" not in block:
        return None
    return round(block["acc,none"] * 100, 1)


def main():
    bench = {}
    for label in LABELS:
        f = RES / f"bench_{label}.json"
        if f.exists():
            bench[label] = json.loads(f.read_text())
    levels = sorted({r["concurrency"] for b in bench.values() for r in b["rows"]})

    head = ["Precision", "Checkpoint (GB)", "MMLU 5-shot (%)"] + [f"tok/s @ {c}" for c in levels]
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] * len(head)) + "|")
    for label in LABELS:
        b = bench.get(label)
        size = b.get("checkpoint_size_gb") if b else None
        acc = mmlu(label)
        cells = [NAMES[label], size if size is not None else "-", acc if acc is not None else "-"]
        for c in levels:
            row = next((r for r in (b["rows"] if b else []) if r["concurrency"] == c), None)
            cells.append(row["tokens_per_s"] if row else "-")
        print("| " + " | ".join(str(x) for x in cells) + " |")
    gpus = {b.get("gpu") for b in bench.values() if b.get("gpu")}
    if gpus:
        print(f"\nGPU: {', '.join(sorted(gpus))}")


if __name__ == "__main__":
    main()
