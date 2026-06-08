"""Run text continuation from a checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.model import build_llama_model

PROMPTS = [
    "The history of machine learning begins",
    "In order to train a large language model, you need",
    "The capital of France is Paris, and the capital of Germany is",
    "Scientists recently discovered that",
]


def generate(model, tokenizer, device, prompt: str, max_new_tokens: int = 100,
             temperature: float = 0.8, top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Path to .pt checkpoint file")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    print(f"Step       : {ckpt['step']}")
    print(f"Model      : {config['model']['name']}")
    print(f"Run name   : {config.get('run_name', '?')}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["dataset"].get("tokenizer_name", "gpt2")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_llama_model(config["model"], tokenizer)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}\n")

    for prompt in PROMPTS:
        continuation = generate(model, tokenizer, device, prompt,
                                args.max_new_tokens, args.temperature, args.top_p)
        print(f"PROMPT:       {prompt}")
        print(f"CONTINUATION: {continuation}")
        print()


if __name__ == "__main__":
    main()
