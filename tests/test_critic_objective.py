import torch

from s2tpo.critic import SIDPageCriticEnsemble
from s2tpo.critic_objective import (
    critic_calibration_objective,
    normalize_item_evidence,
    online_bootstrap_weights,
)
from s2tpo.page_rollout import build_unique_path_states, restore_item_paths


def test_item_evidence_is_normalized_with_uniform_fallback() -> None:
    evidence = torch.tensor([[1.0, 3.0], [0.0, 0.0]])
    alpha = normalize_item_evidence(evidence)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(2))
    assert torch.allclose(alpha[1], torch.tensor([0.5, 0.5]))


def test_complete_critic_objective_backpropagates() -> None:
    torch.manual_seed(3)
    model = SIDPageCriticEnsemble(
        vocab_size=24,
        sid_depth=2,
        hidden_dim=16,
        max_page_size=4,
        dropout=0.0,
        size=3,
    )
    context = torch.tensor([[3, 4, 0], [5, 6, 7]])
    attention = context.ne(0).long()
    pages = torch.tensor([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
    paths = []
    for index in range(len(pages)):
        states = build_unique_path_states(pages[index : index + 1]).squeeze(0)
        values = model.member_values(
            context[index : index + 1].expand(len(states), -1),
            attention[index : index + 1].expand(len(states), -1),
            states,
        )
        paths.append(restore_item_paths(values[None], page_size=2, sid_depth=2)[0])
    member_paths = torch.stack(paths)
    bootstrap = online_bootstrap_weights(2, 3, device=torch.device("cpu"))
    losses = critic_calibration_objective(
        member_paths,
        torch.tensor([1.5, 0.8]),
        torch.tensor([[1.0, 2.0], [3.0, 1.0]]),
        bootstrap_weights=bootstrap,
    )
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert model.members[0].value_head[-1].weight.grad is not None
    assert losses.page.item() >= 0.0
    assert losses.rank.item() >= 0.0
