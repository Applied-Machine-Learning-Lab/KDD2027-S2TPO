"""TIGER-compatible actor update implementing the paper objective directly."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .credit import (
    conservative_ensemble_value,
    group_relative_advantage,
    hierarchical_token_advantage,
)
from .objectives import clipped_policy_loss, forward_kl


def _flatten_candidates(batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    candidate_sids = batch["candidate_sids"]
    if candidate_sids.ndim != 4:
        raise ValueError(
            "candidate_sids must contain complete pages [B,W,N,L]; "
            f"got shape {tuple(candidate_sids.shape)}"
        )
    batch_size, candidate_count, page_size, sid_depth = candidate_sids.shape
    repeated_input = input_ids[:, None, None, :].expand(
        -1, candidate_count, page_size, -1
    )
    repeated_mask = attention_mask[:, None, None, :].expand(
        -1, candidate_count, page_size, -1
    )
    return (
        repeated_input.reshape(batch_size * candidate_count * page_size, -1),
        repeated_mask.reshape(batch_size * candidate_count * page_size, -1),
        candidate_sids.reshape(batch_size * candidate_count * page_size, sid_depth),
    )


def _token_outputs(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
) -> Tuple[Tensor, Tensor]:
    if not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        raise TypeError("the TIGER model must support seq2seq decoder label shifting")
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits
    log_prob = F.log_softmax(logits, dim=-1)
    target_log_prob = log_prob.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return target_log_prob, logits


def s2tpo_actor_loss(
    actor: nn.Module,
    old_policy: nn.Module,
    batch: Dict[str, Tensor],
    *,
    beta_q: float = 0.5,
    group_advantage_clip: Optional[float] = None,
    clip_epsilon: float = 0.20,
    kl_coefficient: float = 0.03,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute the complete attribution-weighted group-relative actor loss."""
    input_ids, attention_mask, labels = _flatten_candidates(batch)
    member_path_values = batch["critic_member_path_values"]
    path_values = conservative_ensemble_value(
        member_path_values,
        beta=float(beta_q),
        dim=-1,
    )
    if path_values.ndim != 4:
        raise ValueError(
            "critic paths must have shape [B,W,N,L+1] after ensemble reduction"
        )
    batch_size, candidate_count, page_size, path_length = path_values.shape
    sid_depth = path_length - 1

    # Every item path ends at the same complete page. Averaging tolerates tiny
    # numerical differences if those duplicate endpoint evaluations were batched.
    complete_scores = path_values[..., -1].mean(dim=-1)
    group_advantage = group_relative_advantage(
        complete_scores,
        clip=(
            float(group_advantage_clip)
            if group_advantage_clip is not None
            else None
        ),
    )
    attribution = hierarchical_token_advantage(
        group_advantage,
        path_values,
    )
    token_advantage = attribution.token_advantage.reshape(
        batch_size * candidate_count * page_size, sid_depth
    ).detach()

    with torch.no_grad():
        old_target_log_prob, old_logits = _token_outputs(
            old_policy, input_ids, attention_mask, labels
        )
    new_target_log_prob, new_logits = _token_outputs(
        actor, input_ids, attention_mask, labels
    )
    policy_loss = clipped_policy_loss(
        new_target_log_prob,
        old_target_log_prob,
        token_advantage,
        clip_epsilon=float(clip_epsilon),
    )

    kl_loss = forward_kl(old_logits, new_logits)
    total_loss = policy_loss + float(kl_coefficient) * kl_loss

    with torch.no_grad():
        credit_error = (
            attribution.sid_credit.sum(dim=-1) - attribution.item_credit
        ).abs().max()
        advantage_error = (
            attribution.token_advantage.sum(dim=(-2, -1)) - group_advantage
        ).abs().max()
        ratio = torch.exp(new_target_log_prob - old_target_log_prob)
    metrics = {
        "loss": float(total_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "forward_kl": float(kl_loss.item()),
        "ratio_mean": float(ratio.mean().item()),
        "group_advantage_abs_mean": float(group_advantage.abs().mean().item()),
        "token_advantage_abs_mean": float(token_advantage.abs().mean().item()),
        "conservative_value_mean": float(path_values.mean().item()),
        "ensemble_std_mean": float(
            member_path_values.std(dim=-1, unbiased=False).mean().item()
        ),
        "credit_conservation_error": float(credit_error.item()),
        "advantage_conservation_error": float(advantage_error.item()),
        "beta_q": float(beta_q),
    }
    return total_loss, metrics


def set_decoder_trainable(model: nn.Module) -> int:
    """Freeze the backbone except decoder-side parameters used by TIGER."""
    trainable = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "decoder" in name or "lm_head" in name
        if parameter.requires_grad:
            trainable += parameter.numel()
    if trainable == 0:
        raise ValueError("no decoder parameters matched; inspect the model parameter names")
    return trainable
