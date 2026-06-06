from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _linear_warmup(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(warmup_steps))


def _inverse_proportional_decay(progress: float, min_lr_ratio: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return 1.0 / ((1.0 - progress) + progress / min_lr_ratio)


def _cosine_decay(progress: float, min_lr_ratio: float) -> float:
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _wsd_decay_intervals(schedule_config: dict[str, Any], warmup_steps: int) -> list[tuple[int, int]]:
    if "decays" in schedule_config:
        intervals = []
        for decay in schedule_config["decays"]:
            start_step = int(decay["start_step"])
            end_step = int(decay["end_step"])
            if end_step <= start_step:
                raise ValueError(
                    f"WSD decay end_step ({end_step}) must be greater than start_step ({start_step})."
                )
            intervals.append((start_step, end_step))
        return sorted(intervals)

    stable_steps = int(schedule_config["stable_steps"])
    decay_steps = int(schedule_config["decay_steps"])
    if decay_steps <= 0:
        raise ValueError("WSD decay_steps must be positive.")
    start_step = warmup_steps + stable_steps
    return [(start_step, start_step + decay_steps)]


def build_probe_scheduler(
    optimizer: Optimizer,
    decay_length: int,
    final_lr_ratio: float,
    decay_type: str = "inverse_proportional",
) -> LambdaLR:
    """Scheduler for probe decays: starts decaying immediately from step 0."""
    decay_type = decay_type.lower()

    def lr_lambda(step: int) -> float:
        if step >= decay_length:
            return final_lr_ratio
        progress = float(step) / float(decay_length)
        if decay_type == "inverse_proportional":
            return _inverse_proportional_decay(progress, final_lr_ratio)
        if decay_type == "cosine":
            return _cosine_decay(progress, final_lr_ratio)
        raise ValueError(f"Unknown decay_type '{decay_type}'.")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_lr_scheduler(optimizer: Optimizer, schedule_config: dict[str, Any]) -> LambdaLR:
    schedule_type = schedule_config["type"].lower()
    warmup_steps = int(schedule_config.get("warmup_steps", 0))

    if schedule_type == "warmup_stable":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return _linear_warmup(step, warmup_steps)
            return 1.0

    elif schedule_type in {"wsd", "wsd_s"}:
        final_lr_ratio = float(schedule_config.get("final_lr_ratio", 0.1))
        if not 0.0 < final_lr_ratio <= 1.0:
            raise ValueError("WSD final_lr_ratio must be in (0, 1].")
        decay_type = schedule_config.get("decay_type", "inverse_proportional").lower()
        decay_intervals = _wsd_decay_intervals(schedule_config, warmup_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return _linear_warmup(step, warmup_steps)
            for decay_idx, (start_step, end_step) in enumerate(decay_intervals):
                if step < start_step:
                    return 1.0
                if step <= end_step:
                    progress = float(step - start_step) / float(end_step - start_step)
                    if decay_type == "inverse_proportional":
                        return _inverse_proportional_decay(progress, final_lr_ratio)
                    if decay_type == "cosine":
                        return _cosine_decay(progress, final_lr_ratio)
                    raise ValueError(
                        f"Unknown WSD decay_type '{decay_type}'. "
                        "Expected 'inverse_proportional' or 'cosine'."
                    )
                if decay_idx < len(decay_intervals) - 1:
                    continue
                return final_lr_ratio
            return 1.0

    elif schedule_type == "cosine":
        total_steps = int(schedule_config["total_steps"])
        min_lr_ratio = float(schedule_config.get("min_lr_ratio", 0.1))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return _linear_warmup(step, warmup_steps)
            progress = float(step - warmup_steps + 1) / float(max(1, total_steps - warmup_steps))
            return _cosine_decay(progress, min_lr_ratio)

    else:
        raise ValueError(
            f"Unknown schedule type '{schedule_type}'. "
            "Expected 'warmup_stable', 'wsd', 'wsd_s', or 'cosine'."
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
