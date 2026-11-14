"""Core components of Session-to-Token Policy Optimization."""

from .agent4rec import (
    build_next_sid_samples,
    discounted_returns,
    encode_history,
    load_behavior_sessions,
    load_cf_sequences,
    load_sid_mapping,
    page_feedback,
)
from .credit import (
    AttributionOutput,
    conservative_ensemble_value,
    group_relative_advantage,
    hierarchical_token_advantage,
    pathwise_sid_credit,
)
from .critic import (
    PageCriticEnsemble,
    PageValueCritic,
    SIDPageCriticEnsemble,
    SIDPageValueCritic,
    load_sid_page_critic,
    save_sid_page_critic,
)
from .critic_objective import (
    CriticLossOutput,
    critic_calibration_objective,
    normalize_item_evidence,
    online_bootstrap_weights,
)
from .export import (
    assemble_complete_page_group,
    group_page_records,
    write_groups_jsonl,
)
from .objectives import clipped_policy_loss, forward_kl, soft_update
from .groups import SCHEMA_VERSION

__all__ = [
    "AttributionOutput",
    "build_next_sid_samples",
    "CriticLossOutput",
    "PageCriticEnsemble",
    "PageValueCritic",
    "SIDPageCriticEnsemble",
    "SIDPageValueCritic",
    "SCHEMA_VERSION",
    "assemble_complete_page_group",
    "clipped_policy_loss",
    "conservative_ensemble_value",
    "critic_calibration_objective",
    "discounted_returns",
    "encode_history",
    "forward_kl",
    "group_relative_advantage",
    "group_page_records",
    "hierarchical_token_advantage",
    "load_sid_page_critic",
    "load_behavior_sessions",
    "load_cf_sequences",
    "load_sid_mapping",
    "normalize_item_evidence",
    "online_bootstrap_weights",
    "page_feedback",
    "pathwise_sid_credit",
    "soft_update",
    "save_sid_page_critic",
    "write_groups_jsonl",
]
