from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def check_policy(
    policy_name: str,
    metrics_history: deque[dict[str, float]],
    policy_cfg: dict[str, Any],
) -> bool:
    """Evaluate whether a policy should start decay.

    Args:
        policy_name: Policy name from the config.
        metrics_history: Recent per-step metrics.
        policy_cfg: Policy config with thresholds.

    Returns:
        True when the policy condition is satisfied.
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
    """Compute percentile thresholds for a policy.

    Args:
        metrics_history: Recent per-step metrics.
        policy_name: Policy name determining which metrics are used.
        percentile: Percentile used as the low-metric threshold.

    Returns:
        Mapping of threshold names to values.
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
    """Check the loss-variance policy condition.

    Args:
        metrics_history: Recent per-step metrics.
        policy_cfg: Policy config containing loss_variance_threshold.

    Returns:
        True if the latest loss variance is below threshold.
    """
    if not metrics_history:
        return False
    threshold = float(policy_cfg["loss_variance_threshold"])
    return metrics_history[-1].get("loss_variance", float("inf")) < threshold


def _check_grad_snr_low(
    metrics_history: deque[dict[str, float]],
    policy_cfg: dict[str, Any],
) -> bool:
    """Check the gradient-SNR policy condition.

    Args:
        metrics_history: Recent per-step metrics.
        policy_cfg: Policy config containing grad_snr_threshold.

    Returns:
        True if the latest Grad SNR is below threshold.
    """
    if not metrics_history:
        return False
    threshold = float(policy_cfg["grad_snr_threshold"])
    return metrics_history[-1].get("grad_snr", float("inf")) < threshold
