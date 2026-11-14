"""Policy optimization and rollout-update utilities for S2T-PO."""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def clipped_policy_loss(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantage: Tensor,
    *,
    clip_epsilon: float = 0.20,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Token-level clipped likelihood-ratio objective."""
    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped_ratio = ratio.clamp(
        min=1.0 - float(clip_epsilon),
        max=1.0 + float(clip_epsilon),
    )
    objective = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    if mask is None:
        return -objective.mean()
    weights = mask.to(dtype=objective.dtype)
    return -(objective * weights).sum() / weights.sum().clamp_min(1.0)


def forward_kl(old_logits: Tensor, new_logits: Tensor) -> Tensor:
    """Compute D_KL(old || new) for categorical token distributions."""
    old_log_prob = F.log_softmax(old_logits, dim=-1)
    new_log_prob = F.log_softmax(new_logits, dim=-1)
    old_prob = old_log_prob.exp()
    return (old_prob * (old_log_prob - new_log_prob)).sum(dim=-1).mean()


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """EMA update: target <- (1 - tau) target + tau source."""
    tau = float(tau)
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    target_parameters = dict(target.named_parameters())
    source_parameters = dict(source.named_parameters())
    if target_parameters.keys() != source_parameters.keys():
        raise ValueError("source and target modules must have matching parameters")
    for name, target_parameter in target_parameters.items():
        target_parameter.lerp_(source_parameters[name], tau)

    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())
    if target_buffers.keys() != source_buffers.keys():
        raise ValueError("source and target modules must have matching buffers")
    for name, target_buffer in target_buffers.items():
        source_buffer = source_buffers[name]
        if target_buffer.is_floating_point() or target_buffer.is_complex():
            target_buffer.lerp_(source_buffer, tau)
        else:
            target_buffer.copy_(source_buffer)
