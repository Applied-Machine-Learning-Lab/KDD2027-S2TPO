import torch
from torch import nn

from s2tpo import (
    clipped_policy_loss,
    conservative_ensemble_value,
    group_relative_advantage,
    hierarchical_token_advantage,
    pathwise_sid_credit,
    soft_update,
)


def test_conservative_ensemble_value() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0]])
    expected = values.mean(dim=-1) - 0.5 * values.std(dim=-1, unbiased=False)
    torch.testing.assert_close(conservative_ensemble_value(values), expected)

    beta_one = values.mean(dim=-1) - values.std(dim=-1, unbiased=False)
    torch.testing.assert_close(
        conservative_ensemble_value(values, beta=1.0), beta_one
    )


def test_group_advantage_is_centered_and_scaled() -> None:
    scores = torch.tensor([[1.0, 2.0, 4.0], [4.0, 4.5, 6.0]])
    advantage = group_relative_advantage(scores)
    torch.testing.assert_close(advantage.mean(dim=-1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        advantage.std(dim=-1, unbiased=False), torch.ones(2), atol=1e-6, rtol=0
    )


def test_path_credit_telescopes_to_item_credit() -> None:
    path = torch.tensor([[0.2, 0.7, 0.4, 1.1, 1.0]])
    item_credit, sid_credit = pathwise_sid_credit(path)
    torch.testing.assert_close(sid_credit.sum(dim=-1), item_credit)


def test_dual_conservation() -> None:
    group_advantage = torch.tensor([1.2, -0.8])
    path = torch.tensor(
        [
            [[0.0, 0.4, 0.3, 0.7, 0.9], [0.2, 0.1, 0.5, 0.7, 0.9]],
            [[0.2, -0.1, 0.5, 0.1, 0.0], [0.4, 0.2, 0.3, 0.1, 0.0]],
        ]
    )
    output = hierarchical_token_advantage(group_advantage, path)
    torch.testing.assert_close(
        output.sid_credit.sum(dim=-1), output.item_credit, atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        output.centered_residual.sum(dim=(-2, -1)),
        torch.zeros(2),
        atol=1e-6,
        rtol=0,
    )
    expected_item_residual = (
        output.item_credit
        - output.item_credit.mean(dim=-1, keepdim=True)
    ) / output.credit_scale
    torch.testing.assert_close(
        output.centered_residual.sum(dim=-1),
        expected_item_residual,
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        output.token_advantage.sum(dim=(-2, -1)), group_advantage, atol=1e-6, rtol=0
    )


def test_local_credit_can_override_negative_candidate_direction() -> None:
    group_advantage = torch.tensor([-0.4])
    path = torch.tensor([[[0.0, 1.0, 0.8, 0.6, 0.4]]])
    output = hierarchical_token_advantage(group_advantage, path)
    assert output.token_advantage[0, 0, 0] > 0
    torch.testing.assert_close(
        output.token_advantage.sum(dim=(-2, -1)), group_advantage, atol=1e-6, rtol=0
    )


def test_clipped_policy_loss_matches_manual_objective() -> None:
    old_log_prob = torch.zeros(2)
    new_log_prob = torch.log(torch.tensor([1.2, 0.8]))
    advantage = torch.tensor([1.0, -1.0])
    loss = clipped_policy_loss(
        new_log_prob,
        old_log_prob,
        advantage,
        clip_epsilon=0.1,
    )
    ratio = torch.tensor([1.2, 0.8])
    clipped = ratio.clamp(0.9, 1.1)
    expected = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    torch.testing.assert_close(loss, expected)


def test_soft_update() -> None:
    target = nn.Linear(2, 1, bias=False)
    source = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        target.weight.zero_()
        source.weight.fill_(2.0)
    soft_update(target, source, tau=0.25)
    torch.testing.assert_close(target.weight, torch.full_like(target.weight, 0.5))


def test_soft_update_includes_float_and_integer_buffers() -> None:
    target = nn.BatchNorm1d(2)
    source = nn.BatchNorm1d(2)
    with torch.no_grad():
        target.running_mean.zero_()
        source.running_mean.fill_(2.0)
        target.num_batches_tracked.zero_()
        source.num_batches_tracked.fill_(4)
    soft_update(target, source, tau=0.25)
    torch.testing.assert_close(target.running_mean, torch.ones(2) * 0.5)
    assert target.num_batches_tracked.item() == 4
