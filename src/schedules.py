from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _linear_warmup(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(warmup_steps))


def build_lr_scheduler(optimizer: Optimizer, schedule_config: dict[str, Any]) -> LambdaLR:
    schedule_type = schedule_config["type"].lower()
    warmup_steps = int(schedule_config.get("warmup_steps", 0))

    if schedule_type == "wsd":
        stable_steps = int(schedule_config["stable_steps"])
        decay_steps = int(schedule_config["decay_steps"])
        final_lr_ratio = float(schedule_config.get("final_lr_ratio", 0.1))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return _linear_warmup(step, warmup_steps)
            if step < warmup_steps + stable_steps:
                return 1.0
            decay_step = step - warmup_steps - stable_steps
            if decay_step >= decay_steps:
                return final_lr_ratio
            progress = float(decay_step + 1) / float(max(1, decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_lr_ratio + (1.0 - final_lr_ratio) * cosine

    elif schedule_type == "cosine":
        total_steps = int(schedule_config["total_steps"])
        min_lr_ratio = float(schedule_config.get("min_lr_ratio", 0.1))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return _linear_warmup(step, warmup_steps)
            progress = float(step - warmup_steps + 1) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    else:
        raise ValueError(f"Unknown schedule type '{schedule_type}'. Expected 'wsd' or 'cosine'.")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

