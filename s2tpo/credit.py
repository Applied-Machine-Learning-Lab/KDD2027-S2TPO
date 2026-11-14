"""Conservative value and conservation-aware complete-page credit assignment."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class AttributionOutput:
    """Item and SID signals for candidate pages with shape ``[...,N,L]``."""

    item_credit: Tensor
    sid_credit: Tensor
    credit_scale: Tensor
    centered_residual: Tensor
    token_advantage: Tensor


def conservative_ensemble_value(
    member_values: Tensor,
    beta: float = 0.5,
    *,
    dim: int = -1,
) -> Tensor:
    """Return ensemble mean minus ``beta`` population standard deviations."""
    if member_values.shape[dim] < 1:
        raise ValueError("the critic ensemble must contain at least one member")
    mean = member_values.mean(dim=dim)
    std = member_values.std(dim=dim, unbiased=False)
    return mean - float(beta) * std


def group_relative_advantage(
    scores: Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
    clip: Optional[float] = None,
) -> Tensor:
    """Standardize complete-page values within each request group."""
    if scores.shape[dim] < 2:
        raise ValueError("group-relative normalization requires at least two pages")
    mean = scores.mean(dim=dim, keepdim=True)
    std = scores.std(dim=dim, keepdim=True, unbiased=False)
    advantage = (scores - mean) / (std + float(eps))
    if clip is not None:
        advantage = advantage.clamp(min=-float(clip), max=float(clip))
    return advantage


def pathwise_sid_credit(path_values: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose each masked-to-complete item path into SID increments.

    The last axis is the path axis: PAD, then one state for every revealed SID
    token. Any preceding axes, including the page item axis, are preserved.
    """
    if path_values.ndim < 1 or path_values.shape[-1] < 2:
        raise ValueError("path_values must contain PAD and at least one SID state")
    sid_credit = path_values[..., 1:] - path_values[..., :-1]
    item_credit = path_values[..., -1] - path_values[..., 0]
    return item_credit, sid_credit


def hierarchical_token_advantage(
    group_advantage: Tensor,
    path_values: Tensor,
    *,
    eps: float = 1e-8,
) -> AttributionOutput:
    """Transport each page advantage to all ``N*L`` SID decisions.

    ``group_advantage`` has shape ``[...]`` and ``path_values`` has shape
    ``[...,N,L+1]``. Token increments are centered by the page-level mean
    item credit, making the residual zero-sum over the complete page while
    preserving differences between items.
    """
    if path_values.ndim < 3:
        raise ValueError("path_values must have shape [...,N,L+1]")
    item_credit, sid_credit = pathwise_sid_credit(path_values)
    if group_advantage.shape != item_credit.shape[:-1]:
        raise ValueError(
            "group_advantage must match path_values before item and path axes: "
            f"got {tuple(group_advantage.shape)} and {tuple(path_values.shape)}"
        )

    page_size = sid_credit.shape[-2]
    sid_depth = sid_credit.shape[-1]
    credit_scale = 2.0 * sid_credit.abs().mean().clamp_min(float(eps))
    mean_item_credit = item_credit.mean(dim=-1, keepdim=True)
    centered_residual = (
        sid_credit - mean_item_credit.unsqueeze(-1) / float(sid_depth)
    ) / credit_scale
    token_advantage = (
        group_advantage[..., None, None] / float(page_size * sid_depth)
        + centered_residual
    )
    return AttributionOutput(
        item_credit=item_credit,
        sid_credit=sid_credit,
        credit_scale=credit_scale,
        centered_residual=centered_residual,
        token_advantage=token_advantage,
    )
