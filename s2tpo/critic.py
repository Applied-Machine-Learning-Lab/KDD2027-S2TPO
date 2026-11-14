"""Long-term page critics used by the public S2T-PO pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .credit import conservative_ensemble_value


class PageValueCritic(nn.Module):
    """Predict long-term page utility from an encoded context-page vector."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


class PageCriticEnsemble(nn.Module):
    """Independently initialized page critics with lower-confidence scoring."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, size: int = 8) -> None:
        super().__init__()
        if size < 1:
            raise ValueError("ensemble size must be positive")
        self.members = nn.ModuleList(
            PageValueCritic(input_dim, hidden_dim) for _ in range(size)
        )

    def member_values(self, features: Tensor) -> Tensor:
        return torch.stack([member(features) for member in self.members], dim=-1)

    def forward(self, features: Tensor, beta: float = 0.5) -> dict[str, Tensor]:
        values = self.member_values(features)
        return {
            "member_values": values,
            "mean": values.mean(dim=-1),
            "std": values.std(dim=-1, unbiased=False),
            "conservative": conservative_ensemble_value(values, beta=beta),
        }


class SIDPageValueCritic(nn.Module):
    """Predict a page return directly from context tokens and item SIDs.

    PAD tokens have ID zero. Context and page encoders are intentionally
    compact so the public implementation is independent of private simulator
    features while retaining the paper's page-value interface.
    """

    def __init__(
        self,
        vocab_size: int,
        sid_depth: int,
        hidden_dim: int = 256,
        max_page_size: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or sid_depth < 1 or max_page_size < 1:
            raise ValueError("invalid SID critic dimensions")
        self.vocab_size = int(vocab_size)
        self.sid_depth = int(sid_depth)
        self.hidden_dim = int(hidden_dim)
        self.max_page_size = int(max_page_size)
        self.dropout = float(dropout)

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.stage_embedding = nn.Embedding(sid_depth, hidden_dim)
        self.slot_embedding = nn.Embedding(max_page_size, hidden_dim)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.item_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=_compatible_head_count(hidden_dim),
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.page_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        context_input_ids: Tensor,
        context_attention_mask: Tensor,
        page_sids: Tensor,
    ) -> Tensor:
        if page_sids.ndim != 3:
            raise ValueError("page_sids must have shape [B,N,L]")
        if page_sids.shape[-1] != self.sid_depth:
            raise ValueError("page SID depth does not match critic configuration")
        if page_sids.shape[1] > self.max_page_size:
            raise ValueError("page size exceeds critic slot capacity")

        context_mask = context_attention_mask.to(dtype=torch.bool)
        context_tokens = self.token_embedding(context_input_ids)
        context = _masked_mean(context_tokens, context_mask, dim=1)
        context = self.context_projection(context)

        token_mask = page_sids.ne(0)
        stages = torch.arange(self.sid_depth, device=page_sids.device)
        token_hidden = self.token_embedding(page_sids) + self.stage_embedding(stages)
        items = _masked_mean(token_hidden, token_mask, dim=2)
        slots = torch.arange(page_sids.shape[1], device=page_sids.device)
        items = self.item_projection(items) + self.slot_embedding(slots)
        item_mask = token_mask.any(dim=-1)

        # Transformer attention cannot accept a fully masked row. Partial path
        # states only mask one item, but this guard keeps malformed inputs safe.
        safe_mask = item_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        if bool(empty.any()):
            safe_mask[empty, 0] = True
            items = items.clone()
            items[empty, 0] = 0.0
        encoded = self.page_encoder(items, src_key_padding_mask=~safe_mask)
        mean_page = _masked_mean(encoded, safe_mask, dim=1)
        max_page = encoded.masked_fill(~safe_mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        max_page = torch.where(torch.isfinite(max_page), max_page, torch.zeros_like(max_page))
        return self.value_head(torch.cat([context, mean_page, max_page], dim=-1)).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "sid_depth": self.sid_depth,
            "hidden_dim": self.hidden_dim,
            "max_page_size": self.max_page_size,
            "dropout": self.dropout,
        }


class SIDPageCriticEnsemble(nn.Module):
    """Independently initialized SID page critics."""

    def __init__(
        self,
        vocab_size: int,
        sid_depth: int,
        hidden_dim: int = 256,
        max_page_size: int = 32,
        dropout: float = 0.1,
        size: int = 8,
    ) -> None:
        super().__init__()
        if size < 1:
            raise ValueError("ensemble size must be positive")
        self.members = nn.ModuleList(
            SIDPageValueCritic(
                vocab_size=vocab_size,
                sid_depth=sid_depth,
                hidden_dim=hidden_dim,
                max_page_size=max_page_size,
                dropout=dropout,
            )
            for _ in range(size)
        )

    @property
    def size(self) -> int:
        return len(self.members)

    def member_values(
        self,
        context_input_ids: Tensor,
        context_attention_mask: Tensor,
        page_sids: Tensor,
    ) -> Tensor:
        return torch.stack(
            [
                member(context_input_ids, context_attention_mask, page_sids)
                for member in self.members
            ],
            dim=-1,
        )

    def forward(
        self,
        context_input_ids: Tensor,
        context_attention_mask: Tensor,
        page_sids: Tensor,
        beta: float = 0.5,
    ) -> Dict[str, Tensor]:
        values = self.member_values(context_input_ids, context_attention_mask, page_sids)
        return {
            "member_values": values,
            "mean": values.mean(dim=-1),
            "std": values.std(dim=-1, unbiased=False),
            "conservative": conservative_ensemble_value(values, beta=beta),
        }

    def config_dict(self) -> Dict[str, Any]:
        config = self.members[0].config_dict()
        config["size"] = self.size
        return config


def save_sid_page_critic(model: SIDPageCriticEnsemble, output_dir: str) -> None:
    """Save a portable critic checkpoint and its architecture metadata."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "critic.pt")
    (destination / "critic_config.json").write_text(
        json.dumps(model.config_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def load_sid_page_critic(
    checkpoint_dir: str,
    device: Optional[torch.device] = None,
) -> SIDPageCriticEnsemble:
    """Load a critic saved by :func:`save_sid_page_critic`."""
    source = Path(checkpoint_dir)
    config = json.loads((source / "critic_config.json").read_text(encoding="utf-8"))
    model = SIDPageCriticEnsemble(**config)
    state = torch.load(source / "critic.pt", map_location=device or "cpu")
    model.load_state_dict(state)
    return model.to(device or torch.device("cpu"))


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def _compatible_head_count(hidden_dim: int) -> int:
    for count in (8, 4, 2):
        if hidden_dim % count == 0:
            return count
    return 1
