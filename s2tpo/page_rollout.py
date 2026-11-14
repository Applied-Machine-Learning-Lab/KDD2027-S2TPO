"""Complete-page proposal sampling and PAD-to-complete path evaluation."""

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn

from .groups import CandidateGroup


SIDTrie = Dict[Tuple[int, ...], List[int]]


def build_sid_trie(valid_sids: Sequence[Sequence[int]]) -> SIDTrie:
    """Build a prefix-to-next-token map for trie-constrained SID decoding."""
    trie: SIDTrie = {}
    depth = None
    for raw_sid in valid_sids:
        sid = tuple(int(token) for token in raw_sid)
        if not sid or any(token <= 0 for token in sid):
            raise ValueError("valid SIDs must contain positive non-PAD token IDs")
        if depth is None:
            depth = len(sid)
        elif len(sid) != depth:
            raise ValueError("all valid SIDs must have the same depth")
        for index, token in enumerate(sid):
            trie.setdefault(sid[:index], []).append(token)
    for prefix, children in trie.items():
        trie[prefix] = sorted(set(children))
    if not trie:
        raise ValueError("at least one valid SID is required")
    return trie


@torch.inference_mode()
def decode_ranked_sids(
    policy: nn.Module,
    context_input_ids: Tensor,
    context_attention_mask: Tensor,
    trie: SIDTrie,
    *,
    sid_depth: int,
    proposal_count: int,
    decoder_start_token_id: int = 0,
) -> List[Tuple[List[int], float]]:
    """Run trie-constrained NTP beam search for one request context."""
    if context_input_ids.shape[0] != 1:
        raise ValueError("decode_ranked_sids expects one context")
    beams: List[Tuple[List[int], float]] = [([], 0.0)]
    for _ in range(sid_depth):
        decoder_rows = [
            [int(decoder_start_token_id)] + sequence for sequence, _ in beams
        ]
        decoder_input_ids = torch.tensor(
            decoder_rows, dtype=torch.long, device=context_input_ids.device
        )
        outputs = policy(
            input_ids=context_input_ids.expand(len(beams), -1),
            attention_mask=context_attention_mask.expand(len(beams), -1),
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        log_prob = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
        candidates: List[Tuple[List[int], float]] = []
        for index, (sequence, base_score) in enumerate(beams):
            allowed = trie.get(tuple(sequence), [])
            if not allowed:
                continue
            allowed_tensor = torch.tensor(allowed, device=log_prob.device)
            scores = log_prob[index, allowed_tensor]
            count = min(proposal_count, len(allowed))
            values, positions = torch.topk(scores, k=count)
            for value, position in zip(values.tolist(), positions.tolist()):
                token = allowed[position]
                candidates.append((sequence + [token], base_score + float(value)))
        candidates.sort(key=lambda row: row[1], reverse=True)
        beams = []
        seen = set()
        for sequence, score in candidates:
            key = tuple(sequence)
            if key in seen:
                continue
            seen.add(key)
            beams.append((sequence, score))
            if len(beams) >= proposal_count:
                break
        if not beams:
            raise RuntimeError("trie-constrained decoding produced no valid SID")
    return beams


def sample_complete_pages(
    ranked_sids: Sequence[Tuple[Sequence[int], float]],
    *,
    page_size: int,
    group_size: int,
    temperature: float,
    generator: torch.Generator,
) -> Tensor:
    """Sample every item of every page from the shared NTP proposal pool.

    Sampling is without replacement inside each page. Different pages are
    sampled independently and duplicate complete pages are rejected.
    """
    unique: Dict[Tuple[int, ...], float] = {}
    for sid, score in ranked_sids:
        unique.setdefault(tuple(int(token) for token in sid), float(score))
    if len(unique) < page_size:
        raise ValueError("proposal pool is smaller than the requested page size")
    sids = list(unique)
    scores = torch.tensor([unique[sid] for sid in sids], dtype=torch.float64)
    probabilities = torch.softmax(scores / max(abs(float(temperature)), 1e-6), dim=0)
    pages: List[Tensor] = []
    seen = set()
    attempts = max(200, group_size * 100)
    sid_tensor = torch.tensor(sids, dtype=torch.long)
    for _ in range(attempts):
        indices = torch.multinomial(
            probabilities,
            num_samples=page_size,
            replacement=False,
            generator=generator,
        )
        page = sid_tensor[indices]
        key = tuple(tuple(int(token) for token in item) for item in page.tolist())
        if key in seen:
            continue
        seen.add(key)
        pages.append(page)
        if len(pages) == group_size:
            break
    if len(pages) != group_size:
        raise RuntimeError("could not sample enough distinct complete pages")
    return torch.stack(pages)


def build_unique_path_states(candidate_pages: Tensor) -> Tensor:
    """Create ``[W,N*L+1,N,L]`` states with one shared complete endpoint."""
    if candidate_pages.ndim != 3:
        raise ValueError("candidate_pages must have shape [W,N,L]")
    group_size, page_size, sid_depth = candidate_pages.shape
    groups: List[Tensor] = []
    for page_index in range(group_size):
        page = candidate_pages[page_index]
        states: List[Tensor] = []
        for item_index in range(page_size):
            for revealed in range(sid_depth):
                state = page.clone()
                state[item_index].zero_()
                if revealed:
                    state[item_index, :revealed] = page[item_index, :revealed]
                states.append(state)
        states.append(page.clone())
        groups.append(torch.stack(states))
    return torch.stack(groups)


def restore_item_paths(
    unique_values: Tensor,
    *,
    page_size: int,
    sid_depth: int,
) -> Tensor:
    """Restore raw critic paths ``[W,N,L+1,C]`` from unique evaluations."""
    if unique_values.ndim != 3:
        raise ValueError("unique_values must have shape [W,N*L+1,C]")
    group_size, state_count, ensemble_size = unique_values.shape
    if state_count != page_size * sid_depth + 1:
        raise ValueError("unique path-state count is inconsistent")
    partial = unique_values[:, :-1].reshape(
        group_size, page_size, sid_depth, ensemble_size
    )
    endpoint = unique_values[:, -1:, :].reshape(
        group_size, 1, 1, ensemble_size
    )
    endpoint = endpoint.expand(-1, page_size, -1, -1)
    return torch.cat([partial, endpoint], dim=2)


@torch.inference_mode()
def evaluate_candidate_paths(
    critic: nn.Module,
    context_input_ids: Tensor,
    context_attention_mask: Tensor,
    candidate_pages: Tensor,
    *,
    batch_size: int = 256,
) -> Tensor:
    """Evaluate all counterfactual states in batched parallel critic calls."""
    states = build_unique_path_states(candidate_pages)
    group_size, state_count, page_size, sid_depth = states.shape
    flat_states = states.reshape(group_size * state_count, page_size, sid_depth)
    chunks: List[Tensor] = []
    for start in range(0, len(flat_states), batch_size):
        pages = flat_states[start : start + batch_size].to(context_input_ids.device)
        context = context_input_ids.expand(len(pages), -1)
        mask = context_attention_mask.expand(len(pages), -1)
        values = critic.member_values(context, mask, pages)
        chunks.append(values.detach().cpu())
    unique_values = torch.cat(chunks).reshape(group_size, state_count, -1)
    return restore_item_paths(
        unique_values, page_size=page_size, sid_depth=sid_depth
    )


def make_candidate_group(
    context_input_ids: Tensor,
    context_attention_mask: Tensor,
    candidate_pages: Tensor,
    member_path_values: Tensor,
) -> CandidateGroup:
    """Build and validate the actor's canonical complete-page record."""
    group = CandidateGroup(
        context_input_ids.squeeze(0).detach().cpu().tolist(),
        context_attention_mask.squeeze(0).detach().cpu().tolist(),
        candidate_pages.detach().cpu().tolist(),
        member_path_values.detach().cpu().tolist(),
    )
    group.validate()
    return group
