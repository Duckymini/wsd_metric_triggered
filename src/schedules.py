from __future__ import annotations

import math
from typing import Any, Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _linear_warmup(step: int, warmup_steps: int) -> float:
    """Compute a linear warmup multiplier.

    Args:
        step: Current scheduler step.
        warmup_steps: Number of warmup steps.

    Returns:
        LR multiplier in [0, 1].
    """
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(warmup_steps))


def _inverse_proportional_decay(progress: float, min_lr_ratio: float) -> float:
    """Compute inverse-proportional decay multiplier.

    Args:
        progress: Decay progress, clipped to [0, 1].
        min_lr_ratio: Final LR multiplier at progress 1.

    Returns:
        LR multiplier for the decay progress.
    """
    progress = min(1.0, max(0.0, progress))
    return 1.0 / ((1.0 - progress) + progress / min_lr_ratio)


def _cosine_decay(progress: float, min_lr_ratio: float) -> float:
    """Compute cosine decay multiplier.

    Args:
        progress: Decay progress, clipped to [0, 1].
        min_lr_ratio: Final LR multiplier at progress 1.

    Returns:
        LR multiplier for the decay progress.
    """
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _wsd_decay_intervals(schedule_config: dict[str, Any], warmup_steps: int) -> list[tuple[int, int]]:
    """Resolve WSD decay intervals from schedule config.

    Args:
        schedule_config: Schedule section of the experiment config.
        warmup_steps: Number of warmup steps before stable training.

    Returns:
        Sorted list of (start_step, end_step) decay intervals.
    """
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
    """Build a scheduler for temporary probe decays.

    Args:
        optimizer: Optimizer controlled by the scheduler.
        decay_length: Number of probe decay steps.
        final_lr_ratio: Final LR multiplier after decay.
        decay_type: Decay curve name.

    Returns:
        LambdaLR scheduler for the probe run.
    """
    decay_type = decay_type.lower()

    def lr_lambda(step: int) -> float:
        """Return the probe LR multiplier.

        Args:
            step: Probe scheduler step.

        Returns:
            LR multiplier for the probe step.
        """
        if step >= decay_length:
            return final_lr_ratio
        progress = float(step) / float(decay_length)
        if decay_type == "inverse_proportional":
            return _inverse_proportional_decay(progress, final_lr_ratio)
        if decay_type == "cosine":
            return _cosine_decay(progress, final_lr_ratio)
        raise ValueError(f"Unknown decay_type '{decay_type}'.")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_policy_decay_scheduler(
    optimizer: Optimizer,
    trigger_step: int,
    decay_length: int,
    final_lr_ratio: float,
    warmup_steps: int,
    decay_type: str = "inverse_proportional",
) -> LambdaLR:
    """Build a scheduler for a policy-triggered decay window.

    Args:
        optimizer: Optimizer controlled by the scheduler.
        trigger_step: Training step where the policy fired.
        decay_length: Number of steps from trigger to final LR.
        final_lr_ratio: Final LR multiplier after decay.
        warmup_steps: Warmup length from the base schedule.
        decay_type: Decay curve name.

    Returns:
        LambdaLR scheduler continuing from the trigger step.
    """
    decay_type = decay_type.lower()
    decay_end = trigger_step + decay_length

    def lr_lambda(step: int) -> float:
        """Return the policy-decay LR multiplier.

        Args:
            step: Global training step.

        Returns:
            LR multiplier for the global step.
        """
        if step < warmup_steps:
            return _linear_warmup(step, warmup_steps)
        if step <= trigger_step:
            return 1.0
        if step <= decay_end:
            progress = float(step - trigger_step) / float(decay_length)
            if decay_type == "inverse_proportional":
                return _inverse_proportional_decay(progress, final_lr_ratio)
            if decay_type == "cosine":
                return _cosine_decay(progress, final_lr_ratio)
            raise ValueError(f"Unknown decay_type '{decay_type}'.")
        return final_lr_ratio

    # last_epoch keeps the LR continuous when swapping from the base scheduler.
    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=trigger_step - 1)


def build_lr_scheduler(optimizer: Optimizer, schedule_config: dict[str, Any]) -> LambdaLR:
    """Build the main LR scheduler for a training run.

    Args:
        optimizer: Optimizer controlled by the scheduler.
        schedule_config: Schedule section of the experiment config.

    Returns:
        LambdaLR scheduler for the configured schedule.
    """
    schedule_type = schedule_config["type"].lower()
    warmup_steps = int(schedule_config.get("warmup_steps", 0))

    if schedule_type == "warmup_stable":

        def lr_lambda(step: int) -> float:
            """Return warmup-stable LR multiplier.

            Args:
                step: Global training step.

            Returns:
                LR multiplier for the global step.
            """
            if step < warmup_steps:
                return _linear_warmup(step, warmup_steps)
            return 1.0

    elif schedule_type in {"wsd", "wsd_s", "wsd_beta"}:
        final_lr_ratio = float(schedule_config.get("final_lr_ratio", 0.1))
        if not 0.0 < final_lr_ratio <= 1.0:
            raise ValueError("WSD final_lr_ratio must be in (0, 1].")
        decay_type = schedule_config.get("decay_type", "inverse_proportional").lower()
        decay_intervals = _wsd_decay_intervals(schedule_config, warmup_steps)

        def lr_lambda(step: int) -> float:
            """Return WSD LR multiplier.

            Args:
                step: Global training step.

            Returns:
                LR multiplier for the global step.
            """
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
            """Return cosine LR multiplier.

            Args:
                step: Global training step.

            Returns:
                LR multiplier for the global step.
            """
            if step < warmup_steps:
                return _linear_warmup(step, warmup_steps)
            progress = float(step - warmup_steps + 1) / float(max(1, total_steps - warmup_steps))
            return _cosine_decay(progress, min_lr_ratio)

    else:
        raise ValueError(
            f"Unknown schedule type '{schedule_type}'. "
            "Expected 'warmup_stable', 'wsd', 'wsd_s', 'wsd_beta', or 'cosine'."
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_beta_scheduler(
    schedule_config: dict[str, Any],
    initial_beta1: float,
    initial_beta2: float,
) -> Callable[[int], tuple[float, float]]:
    """Build a WSD-beta Adam beta scheduler.

    Args:
        schedule_config: Schedule section containing WSD decay windows.
        initial_beta1: Starting Adam beta1 value.
        initial_beta2: Starting Adam beta2 value.

    Returns:
        Function mapping training step to (beta1, beta2).
    """
    warmup_steps = int(schedule_config.get("warmup_steps", 0))
    decay_type = schedule_config.get("decay_type", "inverse_proportional").lower()
    decay_intervals = _wsd_decay_intervals(schedule_config, warmup_steps)
    _BETA_FINAL_RATIO = 1e-6

    def _shape(progress: float) -> float:
        """Compute beta interpolation shape.

        Args:
            progress: Decay progress in [0, 1].

        Returns:
            Multiplier used to interpolate Adam betas.
        """
        if decay_type == "inverse_proportional":
            return _inverse_proportional_decay(progress, _BETA_FINAL_RATIO)
        if decay_type == "cosine":
            return _cosine_decay(progress, _BETA_FINAL_RATIO)
        raise ValueError(f"Unknown decay_type '{decay_type}'.")

    final_beta1 = initial_beta1 * _BETA_FINAL_RATIO
    final_beta2 = 1.0 - (1.0 - initial_beta2) * _BETA_FINAL_RATIO

    def beta_fn(step: int) -> tuple[float, float]:
        """Return Adam betas for a training step.

        Args:
            step: Global training step.

        Returns:
            Pair (beta1, beta2).
        """
        for decay_idx, (start_step, end_step) in enumerate(decay_intervals):
            if step < start_step:
                return (initial_beta1, initial_beta2)
            if step <= end_step:
                progress = float(step - start_step) / float(end_step - start_step)
                s = _shape(progress)
                b1 = initial_beta1 * s
                b2 = 1.0 - (1.0 - initial_beta2) * s
                return (b1, b2)
            if decay_idx == len(decay_intervals) - 1:
                return (final_beta1, final_beta2)
        return (initial_beta1, initial_beta2)

    return beta_fn
