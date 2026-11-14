from types import SimpleNamespace

import torch
from torch import nn

from s2tpo.critic import SIDPageCriticEnsemble
from s2tpo.page_rollout import (
    build_sid_trie,
    build_unique_path_states,
    decode_ranked_sids,
    evaluate_candidate_paths,
    sample_complete_pages,
)


class TinyPolicy(nn.Module):
    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.arange(vocab_size, dtype=torch.float32))

    def forward(self, *, decoder_input_ids: torch.Tensor, **kwargs) -> SimpleNamespace:
        batch, length = decoder_input_ids.shape
        logits = self.bias.view(1, 1, -1).expand(batch, length, -1)
        return SimpleNamespace(logits=logits)


def test_complete_page_rollout_and_parallel_paths() -> None:
    valid_sids = [[1, 5], [1, 6], [2, 7], [2, 8], [3, 9], [3, 10]]
    trie = build_sid_trie(valid_sids)
    context = torch.tensor([[4, 5]])
    attention = torch.ones_like(context)
    ranked = decode_ranked_sids(
        TinyPolicy(), context, attention, trie, sid_depth=2, proposal_count=6
    )
    pages = sample_complete_pages(
        ranked,
        page_size=3,
        group_size=4,
        temperature=1.0,
        generator=torch.Generator().manual_seed(9),
    )
    assert pages.shape == (4, 3, 2)
    assert all(len({tuple(item) for item in page.tolist()}) == 3 for page in pages)
    assert len({tuple(tuple(item) for item in page.tolist()) for page in pages}) == 4

    states = build_unique_path_states(pages)
    assert states.shape == (4, 7, 3, 2)
    critic = SIDPageCriticEnsemble(
        vocab_size=16, sid_depth=2, hidden_dim=16, max_page_size=4, size=2
    )
    paths = evaluate_candidate_paths(critic, context, attention, pages, batch_size=8)
    assert paths.shape == (4, 3, 3, 2)
    assert torch.allclose(paths[:, 0, -1], paths[:, 1, -1])
