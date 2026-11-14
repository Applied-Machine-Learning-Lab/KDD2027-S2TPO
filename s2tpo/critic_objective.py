"""Paper-aligned critic calibration objective for S2T-PO."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class CriticLossOutput:
    total: Tensor
    page: Tensor
    item: Tensor
    prefix: Tensor
    rank: Tensor


def normalize_item_evidence(evidence: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert non-negative item evidence into the paper's alpha weights."""
    if evidence.ndim != 2:
        raise ValueError("item evidence must have shape [B,N]")
    evidence = evidence.clamp_min(0.0)
    total = evidence.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(evidence, 1.0 / float(evidence.shape[-1]))
    return torch.where(total > float(eps), evidence / total.clamp_min(eps), uniform)


def online_bootstrap_weights(
    batch_size: int,
    ensemble_size: int,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Poisson(1) online-bootstrap multiplicities for independent critics."""
    rates = torch.ones((batch_size, ensemble_size), device=device)
    weights = torch.poisson(rates, generator=generator)
    for member in range(ensemble_size):
        if not bool(weights[:, member].gt(0).any()):
            weights[member % batch_size, member] = 1.0
    return weights


def critic_calibration_objective(
    member_path_values: Tensor,
    session_returns: Tensor,
    item_evidence: Tensor,
    *,
    bootstrap_weights: Optional[Tensor] = None,
    item_coefficient: float = 0.75,
    prefix_coefficient: float = 0.50,
    rank_coefficient: float = 0.15,
    huber_beta: float = 1.0,
) -> CriticLossOutput:
    """Compute the complete critic objective disclosed in the appendix.

    Args:
        member_path_values: ``[B,N,L+1,C]`` values from item-masked to
            complete page states for every ensemble member.
        session_returns: discounted long-term return ``[B]``.
        item_evidence: non-negative observed interaction evidence ``[B,N]``.
        bootstrap_weights: optional online-bootstrap counts ``[B,C]``.
    """
    if member_path_values.ndim != 4:
        raise ValueError("member_path_values must have shape [B,N,L+1,C]")
    batch_size, page_size, path_length, ensemble_size = member_path_values.shape
    if path_length < 2:
        raise ValueError("each path needs a PAD state and at least one SID state")
    if session_returns.shape != (batch_size,):
        raise ValueError("session_returns must have shape [B]")
    if item_evidence.shape != (batch_size, page_size):
        raise ValueError("item_evidence must have shape [B,N]")
    if bootstrap_weights is None:
        bootstrap_weights = torch.ones(
            (batch_size, ensemble_size),
            dtype=member_path_values.dtype,
            device=member_path_values.device,
        )
    if bootstrap_weights.shape != (batch_size, ensemble_size):
        raise ValueError("bootstrap_weights must have shape [B,C]")

    sid_depth = path_length - 1
    complete = member_path_values[:, 0, -1, :]
    page_target = session_returns[:, None].expand_as(complete)
    page_loss = _weighted_mean(
        F.smooth_l1_loss(complete, page_target, beta=huber_beta, reduction="none"),
        bootstrap_weights,
    )

    alpha = normalize_item_evidence(item_evidence)
    item_target = alpha * session_returns[:, None]
    item_delta = member_path_values[:, :, -1, :] - member_path_values[:, :, 0, :]
    item_error = F.smooth_l1_loss(
        item_delta,
        item_target[:, :, None].expand_as(item_delta),
        beta=huber_beta,
        reduction="none",
    )
    item_loss = _weighted_mean(item_error, bootstrap_weights[:, None, :])

    prefix_delta = member_path_values[:, :, 1:, :] - member_path_values[:, :, :-1, :]
    prefix_target = item_target[:, :, None] / float(sid_depth)
    prefix_error = F.smooth_l1_loss(
        prefix_delta,
        prefix_target[:, :, :, None].expand_as(prefix_delta),
        beta=huber_beta,
        reduction="none",
    )
    prefix_loss = _weighted_mean(prefix_error, bootstrap_weights[:, None, None, :])

    target_gap = item_target[:, :, None] - item_target[:, None, :]
    prediction_gap = item_delta[:, :, None, :] - item_delta[:, None, :, :]
    pair_mask = torch.triu(
        torch.ones((page_size, page_size), dtype=torch.bool, device=item_delta.device),
        diagonal=1,
    )[None, :, :, None]
    pair_mask = pair_mask & target_gap[:, :, :, None].ne(0)
    rank_error = F.softplus(-target_gap.sign()[:, :, :, None] * prediction_gap)
    rank_weights = bootstrap_weights[:, None, None, :] * pair_mask
    rank_loss = _weighted_mean(rank_error, rank_weights)

    total = (
        page_loss
        + float(item_coefficient) * item_loss
        + float(prefix_coefficient) * prefix_loss
        + float(rank_coefficient) * rank_loss
    )
    return CriticLossOutput(total, page_loss, item_loss, prefix_loss, rank_loss)


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    weights = weights.to(dtype=values.dtype).expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)
