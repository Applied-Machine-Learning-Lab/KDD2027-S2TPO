from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from s2tpo.tiger_actor import s2tpo_actor_loss, set_decoder_trainable


class TinyTiger(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8) -> None:
        super().__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoder_embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor) -> torch.Tensor:
        shifted = torch.zeros_like(labels)
        shifted[:, 1:] = labels[:, :-1]
        return shifted

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        **kwargs,
    ) -> SimpleNamespace:
        context = self.encoder_embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(context.dtype)
        context = (context * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        hidden = self.decoder_embedding(decoder_input_ids) + context.unsqueeze(1)
        return SimpleNamespace(logits=self.lm_head(hidden))


def actor_batch() -> dict:
    return {
        "input_ids": torch.tensor([[6, 7, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0]]),
        "candidate_sids": torch.tensor(
            [[[[2, 3], [4, 5]], [[6, 7], [8, 9]]]]
        ),
        "critic_member_path_values": torch.tensor(
            [
                [
                    [
                        [[0.0, 0.4], [0.8, 1.2], [1.4, 1.8]],
                        [[0.2, 0.6], [0.9, 1.3], [1.4, 1.8]],
                    ],
                    [
                        [[0.0, 0.2], [0.3, 0.5], [0.5, 0.7]],
                        [[0.1, 0.3], [0.4, 0.6], [0.5, 0.7]],
                    ],
                ]
            ]
        ),
    }


def test_actor_loss_backpropagates_and_conserves_advantage() -> None:
    torch.manual_seed(7)
    actor = TinyTiger()
    old_policy = deepcopy(actor).eval()
    trainable = set_decoder_trainable(actor)

    loss, metrics = s2tpo_actor_loss(actor, old_policy, actor_batch(), beta_q=0.5)
    loss.backward()

    assert trainable > 0
    assert actor.lm_head.weight.grad is not None
    assert actor.encoder_embedding.weight.grad is None
    assert metrics["credit_conservation_error"] < 1e-6
    assert metrics["advantage_conservation_error"] < 1e-6
    assert metrics["ensemble_std_mean"] > 0


def test_beta_q_changes_conservative_path_values() -> None:
    actor = TinyTiger()
    old_policy = deepcopy(actor).eval()
    _, no_penalty = s2tpo_actor_loss(actor, old_policy, actor_batch(), beta_q=0.0)
    _, penalized = s2tpo_actor_loss(actor, old_policy, actor_batch(), beta_q=1.0)
    assert penalized["conservative_value_mean"] < no_penalty["conservative_value_mean"]
