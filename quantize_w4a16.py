"""Quantize a model to W4A16 (4-bit weights, 16-bit activations) with GPTQ.

Follows the official llm-compressor W4A16 example.
Usage:  python quantize_w4a16.py --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default=None, help="output dir (default: <model>-W4A16-G128)")
    ap.add_argument("--samples", type=int, default=512, help="calibration samples")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Calibration data: a slice of UltraChat, rendered with the model's chat template.
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{args.samples}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)})
    ds = ds.map(
        lambda s: tokenizer(
            s["text"],
            padding=False,
            max_length=args.max_seq_len,
            truncation=True,
            add_special_tokens=False,
        ),
        remove_columns=ds.column_names,
    )

    # 4-bit weights via GPTQ (group size 128 is the W4A16 preset). Keep lm_head in 16-bit.
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.samples,
    )

    out = args.out or args.model.rstrip("/").split("/")[-1] + "-W4A16-G128"
    model.save_pretrained(out, save_compressed=True)
    tokenizer.save_pretrained(out)
    print(f"Saved W4A16 model to {out}")


if __name__ == "__main__":
    main()
