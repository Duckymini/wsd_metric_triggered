from __future__ import annotations

from collections import deque
from typing import Any


def check_policy(
    policy_name: str,
    metrics_history: deque[dict[str, float]],
    val_history: list[tuple[int, float]],
    policy_cfg: dict[str, Any],
) -> bool:
    """Return True if the policy condition is met and decay should begin.

    Universal pre-condition (all policies): the most recent validation loss
    must be below val_loss_gate. If no val loss has been recorded yet, or it
    hasn't dropped below the gate, returns False immediately.

    Args:
        policy_name: One of 'loss_variance_low', 'grad_snr_low', 'combined'.
        metrics_history: Deque of per-step train metric dicts (populated every
                         training step). Each dict has at minimum 'loss_variance',
                         'grad_snr', 'loss_oscillation', 'loss_improvement_rate'.
        val_history: List of (step, val_loss) pairs in chronological order.
        policy_cfg: The training.policy_decay config dict.
    """
    if not _check_val_loss_gate(val_history, policy_cfg):
        return False

    if policy_name == "loss_variance_low":
        return _check_loss_variance_low(metrics_history, policy_cfg)
    if policy_name == "grad_snr_low":
        return _check_grad_snr_low(metrics_history, policy_cfg)
    if policy_name == "combined":
        return _check_loss_variance_low(metrics_history, policy_cfg) and _check_grad_snr_low(
            metrics_history, policy_cfg
        )
    raise ValueError(f"Unknown policy '{policy_name}'. Expected one of: loss_variance_low, grad_snr_low, combined.")


def _check_val_loss_gate(
    val_history: list[tuple[int, float]],
    policy_cfg: dict[str, Any],
) -> bool:
    """Universal gate: require val_loss to have dropped below val_loss_gate."""
    gate = float(policy_cfg.get("val_loss_gate", float("inf")))
    if not val_history:
        return False
    return val_history[-1][1] < gate


def _check_loss_variance_low(
    metrics_history: deque[dict[str, float]],
    policy_cfg: dict[str, Any],
) -> bool:
    """Trigger when the most recent loss_variance is below threshold."""
    if not metrics_history:
        return False
    threshold = float(policy_cfg["loss_variance_threshold"])
    return metrics_history[-1].get("loss_variance", float("inf")) < threshold



def _check_grad_snr_low(
    metrics_history: deque[dict[str, float]],
    policy_cfg: dict[str, Any],
) -> bool:
    """Trigger when the most recent grad_snr is below threshold."""
    if not metrics_history:
        return False
    threshold = float(policy_cfg["grad_snr_threshold"])
    return metrics_history[-1].get("grad_snr", float("inf")) < threshold
