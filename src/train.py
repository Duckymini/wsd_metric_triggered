from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, default_data_collator, set_seed

from src.config import load_config, save_config
from src.data import load_lm_datasets
from src.model import build_llama_model
from src.policy import check_policy
from src.schedules import build_lr_scheduler, build_policy_decay_scheduler, build_probe_scheduler


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


def _compute_step_metrics(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_norm: float,
    prev_grad_vec: torch.Tensor | None,
) -> tuple[dict[str, float], torch.Tensor]:
    """Gradient and weight metrics computed after backward, before optimizer.step()."""
    grad_mags = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
    grad_snr = (grad_mags.mean() / (grad_mags.std() + 1e-8)).item()

    weight_norm = sum(p.data.norm() ** 2 for p in model.parameters()).sqrt().item()

    current_grad_vec = torch.cat([
        p.grad.detach().flatten() for p in model.parameters() if p.grad is not None
    ])
    grad_cosine_sim = (
        F.cosine_similarity(current_grad_vec.unsqueeze(0), prev_grad_vec.unsqueeze(0)).item()
        if prev_grad_vec is not None
        else 0.0
    )

    v_tensors = [s["exp_avg_sq"] for s in optimizer.state.values() if "exp_avg_sq" in s]
    adam_v_norm = sum(v.norm() ** 2 for v in v_tensors).sqrt().item() if v_tensors else 0.0

    metrics = {
        "grad_snr": grad_snr,
        "weight_norm": weight_norm,
        "grad_weight_ratio": grad_norm / (weight_norm + 1e-8),
        "grad_cosine_sim": grad_cosine_sim,
        "adam_v_norm": adam_v_norm,
    }
    return metrics, current_grad_vec


def _compute_post_step_metrics(
    model: torch.nn.Module,
    prev_params_vec: torch.Tensor | None,
) -> tuple[dict[str, float], torch.Tensor]:
    """Parameter update magnitude computed after optimizer.step()."""
    current_params_vec = torch.cat([p.data.detach().flatten() for p in model.parameters()])
    param_update_norm = (
        (current_params_vec - prev_params_vec).norm().item() if prev_params_vec is not None else 0.0
    )
    return {"param_update_norm": param_update_norm}, current_params_vec


def _run_probe_decay(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    sweep_cfg: dict[str, Any],
    training: dict[str, Any],
    device: torch.device,
    mixed_precision: str,
    context_length: int,
) -> float:
    """Snapshot model+optimizer, run a probe decay, evaluate, restore. Returns val loss."""
    decay_length = int(sweep_cfg["decay_length"])
    final_lr_ratio = float(sweep_cfg.get("final_lr_ratio", 0.1))
    decay_type = sweep_cfg.get("decay_type", "inverse_proportional")
    grad_accum = int(training.get("gradient_accumulation_steps", 1))
    max_grad_norm = training.get("max_grad_norm")
    max_eval_batches = int(training.get("max_eval_batches", 20))

    saved_model = deepcopy(model.state_dict())
    saved_optim = deepcopy(optimizer.state_dict())

    probe_scheduler = build_probe_scheduler(optimizer, decay_length, final_lr_ratio, decay_type)
    probe_iter = iter(train_loader)

    model.train()
    for _ in range(decay_length):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            try:
                batch = next(probe_iter)
            except StopIteration:
                probe_iter = iter(train_loader)
                batch = next(probe_iter)
            batch = _batch_to_device(batch, device, context_length)
            with _autocast_context(device, mixed_precision):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()
        probe_scheduler.step()

    val_loss = evaluate(model, valid_loader, device, max_eval_batches, mixed_precision, context_length)

    model.load_state_dict(saved_model)
    optimizer.load_state_dict(saved_optim)
    return val_loss


def _run_probe_decay_with_history(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    probe_cfg: dict[str, Any],
    training: dict[str, Any],
    device: torch.device,
    mixed_precision: str,
    context_length: int,
) -> tuple[float, list[dict[str, float]]]:
    """Snapshot state, run one probe decay, log validation through decay, then restore."""
    decay_length = int(probe_cfg["decay_length"])
    final_lr_ratio = float(probe_cfg.get("final_lr_ratio", 0.1))
    decay_type = probe_cfg.get("decay_type", "inverse_proportional")
    eval_interval = int(probe_cfg.get("probe_eval_interval_steps", 5))
    grad_accum = int(training.get("gradient_accumulation_steps", 1))
    max_grad_norm = training.get("max_grad_norm")
    max_eval_batches = int(probe_cfg.get("max_probe_eval_batches", training.get("max_eval_batches", 20)))

    saved_model = deepcopy(model.state_dict())
    saved_optim = deepcopy(optimizer.state_dict())

    probe_scheduler = build_probe_scheduler(optimizer, decay_length, final_lr_ratio, decay_type)
    probe_iter = iter(train_loader)
    history: list[dict[str, float]] = []

    model.train()
    for probe_step in range(1, decay_length + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(probe_iter)
            except StopIteration:
                probe_iter = iter(train_loader)
                batch = next(probe_iter)
            batch = _batch_to_device(batch, device, context_length)
            with _autocast_context(device, mixed_precision):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            loss.backward()
            step_loss += float(loss.detach().cpu())
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()
        probe_scheduler.step()

        if probe_step % eval_interval == 0 or probe_step == decay_length:
            val_loss = evaluate(
                model,
                valid_loader,
                device,
                max_eval_batches,
                mixed_precision,
                context_length,
            )
            history.append(
                {
                    "probe_decay_step": probe_step,
                    "probe_train_loss": step_loss,
                    "probe_validation_loss": val_loss,
                    "probe_learning_rate": probe_scheduler.get_last_lr()[0],
                }
            )
            model.train()

    final_val_loss = history[-1]["probe_validation_loss"]
    model.load_state_dict(saved_model)
    optimizer.load_state_dict(saved_optim)
    return final_val_loss, history


def _run_decay_amount_sweep(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    amount_cfg: dict[str, Any],
    training: dict[str, Any],
    device: torch.device,
    mixed_precision: str,
    context_length: int,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Run several short decay probes from the same snapshot, varying final LR ratio."""
    decay_length = int(amount_cfg["decay_length"])
    decay_type = amount_cfg.get("decay_type", "inverse_proportional")
    eval_interval = int(amount_cfg.get("probe_eval_interval_steps", 5))
    max_probe_eval_batches = int(amount_cfg.get("max_probe_eval_batches", training.get("max_eval_batches", 20)))
    ratios = [float(ratio) for ratio in amount_cfg["final_lr_ratios"]]

    rows = []
    history_rows = []
    for final_lr_ratio in ratios:
        probe_cfg = {
            "decay_length": decay_length,
            "final_lr_ratio": final_lr_ratio,
            "decay_type": decay_type,
            "probe_eval_interval_steps": eval_interval,
            "max_probe_eval_batches": max_probe_eval_batches,
        }
        val_loss, history = _run_probe_decay_with_history(
            model,
            optimizer,
            train_loader,
            valid_loader,
            probe_cfg,
            training,
            device,
            mixed_precision,
            context_length,
        )
        rows.append(
            {
                "decay_length": decay_length,
                "final_lr_ratio": final_lr_ratio,
                "probe_final_val_loss": val_loss,
            }
        )
        for history_row in history:
            history_rows.append(
                {
                    "decay_length": decay_length,
                    "final_lr_ratio": final_lr_ratio,
                    **history_row,
                }
            )
    return rows, history_rows


def _append_registry(run_id: str, config: dict[str, Any], final_metrics: dict[str, Any], trigger: str = "fixed") -> None:
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
        "trigger": trigger,
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
    beta_scheduler = None
    if config["schedule"]["type"].lower() == "wsd_beta":
        initial_betas = tuple(training.get("betas", [0.9, 0.95]))
        beta_scheduler = build_beta_scheduler(config["schedule"], float(initial_betas[0]), float(initial_betas[1]))

    max_steps = int(training["max_steps"])
    grad_accum = int(training.get("gradient_accumulation_steps", 1))
    log_interval = int(training.get("log_interval_steps", 1))
    eval_interval = int(training.get("eval_interval_steps", 50))
    save_interval = int(training.get("save_interval_steps", 100))
    max_eval_batches = int(training.get("max_eval_batches", 20))
    mixed_precision = training.get("mixed_precision", "auto")
    if mixed_precision == "auto":
        mixed_precision = "bf16" if device.type == "cuda" else "none"

    sweep_cfg = training.get("decay_sweep", {})
    sweep_enabled = bool(sweep_cfg.get("enabled", False))
    probe_steps = set(int(s) for s in sweep_cfg.get("probe_steps", []))
    amount_sweep_cfg = training.get("decay_amount_sweep", {})
    amount_sweep_enabled = bool(amount_sweep_cfg.get("enabled", False))
    amount_probe_steps = set(int(s) for s in amount_sweep_cfg.get("probe_steps", []))
    log_path = output_paths["logs"] / "metrics.jsonl"
    sweep_log_path = output_paths["logs"] / "decay_sweep.jsonl"
    amount_sweep_log_path = output_paths["logs"] / "decay_amount_sweep.jsonl"
    amount_trajectory_log_path = output_paths["logs"] / "decay_amount_trajectory.jsonl"
    # --- Policy decay setup ---
    policy_cfg = training.get("policy_decay", {})
    policy_enabled = bool(policy_cfg.get("enabled", False))
    policy_name = policy_cfg.get("policy", None)
    earliest_trigger = int(policy_cfg.get("earliest_trigger_step", 3500))
    policy_decay_length = int(policy_cfg.get("decay_length", 500))
    policy_final_lr_ratio = float(policy_cfg.get("final_lr_ratio", 0.25))
    policy_decay_type = policy_cfg.get("decay_type", "inverse_proportional")
    policy_warmup_steps = int(config["schedule"].get("warmup_steps", 0))
    policy_triggered = False
    policy_trigger_step: int | None = None
    metrics_history: deque[dict[str, float]] = deque(maxlen=100)
    val_loss_history: list[tuple[int, float]] = []
    policy_trigger_log_path = output_paths["logs"] / "policy_trigger.json"

    train_iter = iter(train_loader)
    tokens_seen = 0
    running_loss = 0.0
    start_time = time.time()

    # Rolling state for extended metrics
    recent_losses: deque[float] = deque(maxlen=10)
    prev_grad_vec: torch.Tensor | None = None
    prev_params_vec: torch.Tensor | None = None

    model.train()
    progress = tqdm(range(1, max_steps + 1), desc=f"Training {run_id}")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        all_per_example_losses: list[torch.Tensor] = []

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

            # Per-example loss variance (before backward while logits are live)
            with torch.no_grad():
                B, T, V = outputs.logits.shape
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = batch["labels"][:, 1:].contiguous()
                per_token = F.cross_entropy(
                    shift_logits.reshape(-1, V),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape(B, T - 1)
                mask = (shift_labels != -100).float()
                per_example = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)
                all_per_example_losses.append(per_example.cpu())

            loss.backward()
            step_loss += float(loss.detach().cpu())

        # Gradient metrics (gradients available, before optimizer.step)
        if training.get("max_grad_norm") is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["max_grad_norm"])
            ).item()
        else:
            grad_norm = sum(
                p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None
            ).sqrt().item()

        grad_metrics, current_grad_vec = _compute_step_metrics(
            model, optimizer, grad_norm, prev_grad_vec
        )
        prev_grad_vec = current_grad_vec

        optimizer.step()
        scheduler.step()
        if beta_scheduler is not None:
            b1, b2 = beta_scheduler(step)
            for pg in optimizer.param_groups:
                pg["betas"] = (b1, b2)

        post_metrics, current_params_vec = _compute_post_step_metrics(model, prev_params_vec)
        prev_params_vec = current_params_vec

        # Rolling loss stats
        running_loss += step_loss
        recent_losses.append(step_loss)
        loss_variance = (
            torch.cat(all_per_example_losses).var().item() if len(all_per_example_losses) > 1 else 0.0
        )
        loss_oscillation = (
            torch.tensor(list(recent_losses)).std().item() if len(recent_losses) > 1 else 0.0
        )
        loss_improvement_rate = (
            (recent_losses[0] - recent_losses[-1]) / len(recent_losses)
            if len(recent_losses) > 1
            else 0.0
        )

        if policy_enabled and not policy_triggered:
            metrics_history.append({
                "step": step,
                "loss_variance": loss_variance,
                "grad_snr": grad_metrics["grad_snr"],
                "loss_oscillation": loss_oscillation,
                "loss_improvement_rate": loss_improvement_rate,
            })

        lr = scheduler.get_last_lr()[0]
        progress.set_postfix(loss=f"{step_loss:.4f}", lr=f"{lr:.2e}")

        if step % log_interval == 0:
            _append_jsonl(
                log_path,
                {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": step_loss,
                    "loss_variance": loss_variance,
                    "loss_oscillation": loss_oscillation,
                    "loss_improvement_rate": loss_improvement_rate,
                    "grad_norm": grad_norm,
                    "grad_snr": grad_metrics["grad_snr"],
                    "grad_weight_ratio": grad_metrics["grad_weight_ratio"],
                    "grad_cosine_sim": grad_metrics["grad_cosine_sim"],
                    "adam_v_norm": grad_metrics["adam_v_norm"],
                    "weight_norm": grad_metrics["weight_norm"],
                    "param_update_norm": post_metrics["param_update_norm"],
                    "learning_rate": lr,
                    **({"beta1": beta_scheduler(step)[0], "beta2": beta_scheduler(step)[1]} if beta_scheduler is not None else {}),
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
            if policy_enabled and not policy_triggered:
                val_loss_history.append((step, valid_loss))

        if policy_enabled and not policy_triggered and step >= earliest_trigger and policy_name:
            if check_policy(policy_name, metrics_history, val_loss_history, policy_cfg):
                policy_triggered = True
                policy_trigger_step = step
                if step + policy_decay_length > max_steps:
                    import warnings
                    warnings.warn(
                        f"Policy triggered at step {step} but decay would end at "
                        f"{step + policy_decay_length} > max_steps {max_steps}. "
                        "Training will end mid-decay."
                    )
                scheduler = build_policy_decay_scheduler(
                    optimizer,
                    trigger_step=step,
                    decay_length=policy_decay_length,
                    final_lr_ratio=policy_final_lr_ratio,
                    warmup_steps=policy_warmup_steps,
                    decay_type=policy_decay_type,
                )
                with policy_trigger_log_path.open("w", encoding="utf-8") as _f:
                    json.dump(
                        {
                            "policy": policy_name,
                            "trigger_step": step,
                            "decay_end_step": step + policy_decay_length,
                            "final_lr_ratio": policy_final_lr_ratio,
                            "decay_length": policy_decay_length,
                            "metrics_at_trigger": dict(metrics_history[-1]) if metrics_history else {},
                        },
                        _f,
                        indent=2,
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

        if sweep_enabled and step in probe_steps:
            probe_val_loss = _run_probe_decay(
                model, optimizer, train_loader, valid_loader,
                sweep_cfg, training, device, mixed_precision, context_length,
            )
            _append_jsonl(
                sweep_log_path,
                {
                    "probe_start_step": step,
                    "probe_final_val_loss": probe_val_loss,
                    "train_loss": step_loss,
                    "loss_variance": loss_variance,
                    "loss_oscillation": loss_oscillation,
                    "loss_improvement_rate": loss_improvement_rate,
                    "grad_norm": grad_norm,
                    "grad_snr": grad_metrics["grad_snr"],
                    "grad_weight_ratio": grad_metrics["grad_weight_ratio"],
                    "grad_cosine_sim": grad_metrics["grad_cosine_sim"],
                    "adam_v_norm": grad_metrics["adam_v_norm"],
                    "weight_norm": grad_metrics["weight_norm"],
                    "param_update_norm": post_metrics["param_update_norm"],
                    "learning_rate": lr,
                },
            )

        if amount_sweep_enabled and step in amount_probe_steps:
            amount_rows, amount_history_rows = _run_decay_amount_sweep(
                model,
                optimizer,
                train_loader,
                valid_loader,
                amount_sweep_cfg,
                training,
                device,
                mixed_precision,
                context_length,
            )
            for amount_row in amount_rows:
                _append_jsonl(
                    amount_sweep_log_path,
                    {
                        "probe_start_step": step,
                        "train_loss": step_loss,
                        "loss_variance": loss_variance,
                        "loss_oscillation": loss_oscillation,
                        "loss_improvement_rate": loss_improvement_rate,
                        "grad_norm": grad_norm,
                        "grad_snr": grad_metrics["grad_snr"],
                        "grad_weight_ratio": grad_metrics["grad_weight_ratio"],
                        "grad_cosine_sim": grad_metrics["grad_cosine_sim"],
                        "adam_v_norm": grad_metrics["adam_v_norm"],
                        "weight_norm": grad_metrics["weight_norm"],
                        "param_update_norm": post_metrics["param_update_norm"],
                        "learning_rate": lr,
                        **amount_row,
                    },
                )
            for history_row in amount_history_rows:
                _append_jsonl(
                    amount_trajectory_log_path,
                    {
                        "probe_start_step": step,
                        **history_row,
                    },
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

    _append_registry(
        run_id,
        config,
        final_metrics,
        trigger=policy_name if (policy_enabled and policy_triggered) else "fixed",
    )
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
