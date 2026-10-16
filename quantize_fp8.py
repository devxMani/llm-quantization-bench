"""Quantize a model to FP8 (W8A8): FP8 weights + dynamic per-token FP8 activations.

No calibration data is needed for the dynamic scheme.
Usage:  python quantize_fp8.py --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default=None, help="output dir (default: <model>-FP8-Dynamic)")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe)

    out = args.out or args.model.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"Saved FP8 model to {out}")


if __name__ == "__main__":
    main()
