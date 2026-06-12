from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def check_policy(
    policy_name: str,
    metrics_history: deque[dict[str, float]],
    policy_cfg: dict[str, Any],
) -> bool:
    """Return True if the policy condition is met and decay should begin.

    Thresholds must have been pre-computed and injected into policy_cfg
    by compute_percentile_thresholds() before this is called.

    Args:
        policy_name: One of 'loss_variance_low', 'grad_snr_low', 'combined'.
        metrics_history: Deque of per-step train metric dicts.
        policy_cfg: The training.policy_decay config dict (with thresholds injected).
    """
    if policy_name == "loss_variance_low":
        return _check_loss_variance_low(metrics_history, policy_cfg)
    if policy_name == "grad_snr_low":
        return _check_grad_snr_low(metrics_history, policy_cfg)
    if policy_name == "combined":
        return _check_loss_variance_low(metrics_history, policy_cfg) and _check_grad_snr_low(
            metrics_history, policy_cfg
        )
    raise ValueError(f"Unknown policy '{policy_name}'. Expected one of: loss_variance_low, grad_snr_low, combined.")


def compute_percentile_thresholds(
    metrics_history: deque[dict[str, float]],
    policy_name: str,
    percentile: float = 5.0,
) -> dict[str, float]:
    """Compute percentile-based thresholds from the metrics accumulated so far.

    Called once at gate_step to auto-derive trigger thresholds from the
    distribution of metrics observed in the lookback window.

    Returns a dict with the relevant threshold keys (loss_variance_threshold,
    grad_snr_threshold) for the given policy.
    """
    result: dict[str, float] = {}
    if not metrics_history:
        return result

    if policy_name in ("loss_variance_low", "combined"):
        values = [m["loss_variance"] for m in metrics_history if "loss_variance" in m]
        if values:
            result["loss_variance_threshold"] = float(np.percentile(values, percentile))

    if policy_name in ("grad_snr_low", "combined"):
        values = [m["grad_snr"] for m in metrics_history if "grad_snr" in m]
        if values:
            result["grad_snr_threshold"] = float(np.percentile(values, percentile))

    return result


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
