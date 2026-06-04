from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, default_data_collator, set_seed

from src.config import load_config, save_config
from src.data import load_lm_datasets
from src.model import build_llama_model
from src.schedules import build_lr_scheduler


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _configure_torch_for_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "none":
        return nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_run_id(config: dict[str, Any], override: str | None) -> str:
    base = override or config.get("run_name", "debug_run")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}_{timestamp}"


def _prepare_outputs(run_id: str) -> dict[str, Path]:
    paths = {
        "checkpoints": Path("checkpoints") / run_id,
        "logs": Path("logs") / run_id,
        "results": Path("results") / run_id,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
        },
        path,
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    context_length: int,
) -> dict[str, torch.Tensor]:
    sequence_length = int(batch["input_ids"].shape[1])
    if sequence_length > context_length:
        raise ValueError(
            f"Batch sequence length {sequence_length} exceeds model context_length {context_length}."
        )
    non_blocking = device.type == "cuda"
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    mixed_precision: str,
    context_length: int,
) -> float:
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = _batch_to_device(batch, device, context_length)
        with _autocast_context(device, mixed_precision):
            outputs = model(**batch)
        losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def _append_registry(run_id: str, config: dict[str, Any], final_metrics: dict[str, Any]) -> None:
    registry_path = Path("experiments") / "registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    schedule = config["schedule"]
    dataset = config["dataset"]
    model = config["model"]
    if schedule.get("type") in {"wsd", "wsd_s"}:
        if "decays" in schedule:
            decay_start = ";".join(str(decay["start_step"]) for decay in schedule["decays"])
            decay_length = ";".join(
                str(int(decay["end_step"]) - int(decay["start_step"]))
                for decay in schedule["decays"]
            )
        else:
            decay_start = int(schedule.get("warmup_steps", 0)) + int(schedule.get("stable_steps", 0))
            decay_length = schedule.get("decay_steps", "")
        final_lr_ratio = schedule.get("final_lr_ratio", "")
    else:
        decay_start = ""
        decay_length = ""
        final_lr_ratio = schedule.get("min_lr_ratio", "")

    row = {
        "run_id": run_id,
        "owner": config.get("owner", "TBD"),
        "status": "finished",
        "model": model.get("name", "small_llama"),
        "dataset": dataset.get("name", "HuggingFaceFW/fineweb"),
        "tokens": final_metrics.get("tokens_seen"),
        "schedule": schedule.get("type"),
        "decay_start": decay_start,
        "decay_length": decay_length,
        "final_lr_ratio": final_lr_ratio,
        "trigger": "fixed",
        "seed": config.get("seed", ""),
        "result": final_metrics.get("final_validation_loss"),
        "notes": training.get("notes", ""),
    }
    fieldnames = [
        "run_id",
        "owner",
        "status",
        "model",
        "dataset",
        "tokens",
        "schedule",
        "decay_start",
        "decay_length",
        "final_lr_ratio",
        "trigger",
        "seed",
        "result",
        "notes",
    ]
    with registry_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if registry_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def train(config_path: str, run_name: str | None = None) -> None:
    config = load_config(config_path)
    seed = int(config.get("seed", 1))
    set_seed(seed)

    run_id = _make_run_id(config, run_name)
    output_paths = _prepare_outputs(run_id)
    save_config(config, output_paths["results"] / "config.yaml")
    shutil.copy2(config_path, output_paths["results"] / "source_config.yaml")

    device = _device()
    _configure_torch_for_device(device)
    tokenizer = AutoTokenizer.from_pretrained(config["dataset"].get("tokenizer_name", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    context_length = int(config["model"].get("context_length", 512))
    datasets = load_lm_datasets(config["dataset"], tokenizer, context_length, seed)
    is_streaming = isinstance(datasets["train"], IterableDataset)
    if not is_streaming and (len(datasets["train"]) == 0 or len(datasets["validation"]) == 0):
        raise ValueError("Tokenized train and validation datasets must both contain at least one block.")
    model = build_llama_model(config["model"], tokenizer).to(device)

    training = config["training"]
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(training["batch_size"]),
        shuffle=not is_streaming,
        collate_fn=default_data_collator,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        datasets["validation"],
        batch_size=int(training.get("eval_batch_size", training["batch_size"])),
        shuffle=False,
        collate_fn=default_data_collator,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
        betas=tuple(training.get("betas", [0.9, 0.95])),
    )
    scheduler = build_lr_scheduler(optimizer, config["schedule"])

    max_steps = int(training["max_steps"])
    grad_accum = int(training.get("gradient_accumulation_steps", 1))
    log_interval = int(training.get("log_interval_steps", 1))
    eval_interval = int(training.get("eval_interval_steps", 50))
    save_interval = int(training.get("save_interval_steps", 100))
    max_eval_batches = int(training.get("max_eval_batches", 20))
    mixed_precision = training.get("mixed_precision", "auto")
    if mixed_precision == "auto":
        mixed_precision = "bf16" if device.type == "cuda" else "none"

    log_path = output_paths["logs"] / "metrics.jsonl"
    train_iter = iter(train_loader)
    tokens_seen = 0
    running_loss = 0.0
    start_time = time.time()

    model.train()
    progress = tqdm(range(1, max_steps + 1), desc=f"Training {run_id}")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = _batch_to_device(batch, device, context_length)
            tokens_seen += int(batch["input_ids"].numel())
            with _autocast_context(device, mixed_precision):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            loss.backward()
            step_loss += float(loss.detach().cpu())

        if training.get("max_grad_norm") is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
        optimizer.step()
        scheduler.step()

        running_loss += step_loss
        lr = scheduler.get_last_lr()[0]
        progress.set_postfix(loss=f"{step_loss:.4f}", lr=f"{lr:.2e}")

        if step % log_interval == 0:
            _append_jsonl(
                log_path,
                {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": step_loss,
                    "learning_rate": lr,
                    "elapsed_seconds": time.time() - start_time,
                },
            )

        if step % eval_interval == 0 or step == max_steps:
            valid_loss = evaluate(
                model,
                valid_loader,
                device,
                max_eval_batches,
                mixed_precision,
                context_length,
            )
            _append_jsonl(
                log_path,
                {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "validation_loss": valid_loss,
                    "learning_rate": lr,
                    "elapsed_seconds": time.time() - start_time,
                },
            )

        if step % save_interval == 0 or step == max_steps:
            _save_checkpoint(
                output_paths["checkpoints"] / f"checkpoint_step_{step}.pt",
                model,
                optimizer,
                scheduler,
                step,
                config,
            )

    final_validation_loss = evaluate(
        model,
        valid_loader,
        device,
        max_eval_batches,
        mixed_precision,
        context_length,
    )
    final_metrics = {
        "run_id": run_id,
        "final_train_loss": running_loss / max(1, max_steps),
        "final_validation_loss": final_validation_loss,
        "tokens_seen": tokens_seen,
        "final_learning_rate": scheduler.get_last_lr()[0],
        "total_seconds": time.time() - start_time,
        "device": str(device),
        "schedule": config["schedule"]["type"],
    }
    results_path = output_paths["results"] / "final_metrics.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    _append_registry(run_id, config, final_metrics)
    print(f"Finished run: {run_id}")
    print(f"Logs: {log_path}")
    print(f"Checkpoints: {output_paths['checkpoints']}")
    print(f"Results: {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LLaMA-style language model baseline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--run-name", default=None, help="Optional run-name override.")
    args = parser.parse_args()
    train(args.config, args.run_name)
    if os.environ.get("TRAIN_FORCE_OS_EXIT_ON_SUCCESS") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
