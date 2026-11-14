# S2T-PO

Official implementation of **Session-to-Token Policy Optimization
(S2T-PO)** for long-term generative recommendation.

S2T-PO aligns autoregressive Semantic Identifier (SID) generation with
delayed session utility. It first compares multiple complete recommendation
pages, then transports each page-level signal through its items and SID
tokens along PAD-to-complete counterfactual paths.

## Overview

The public pipeline follows four paper-aligned stages:

1. **Conservative Long-Term Page Critic** trains a bootstrap ensemble from
   discounted session returns and penalizes critic disagreement.
2. **Group-Relative Page Advantage** constructs multiple distinct complete
   pages for one context and normalizes their conservative values.
3. **Pathwise Credit Transport** masks one item at a time and progressively
   restores its SID tokens while keeping the rest of the page fixed.
4. **Credit-Aware Token Policy Optimization** combines the page advantage
   with zero-sum local residuals and updates the SID decoder with PPO and KL
   regularization.

## Core Formulation

For ensemble page values $Q_c(x,\mathbf a)$, the conservative value is

$$
Q(x,\mathbf a)
=\mathrm{Mean}_c\, Q_c(x,\mathbf a)
-\beta\,\mathrm{Std}_c\, Q_c(x,\mathbf a).
$$

For page $w$ and item $k$, the critic evaluates a path from the
PAD-filled SID to the complete SID. Adjacent values define item and token
credits:

$$
\Delta_{w,k}=q_{w,k,L}-q_{w,k,0},
\qquad
\Delta_{w,k,\ell}=q_{w,k,\ell}-q_{w,k,\ell-1}.
$$

The finite differences telescope exactly:

$$
\sum_{\ell=1}^{L}\Delta_{w,k,\ell}=\Delta_{w,k}.
$$

After page-wise centering and shared-batch normalization, the zero-sum
residual $R_{w,k,\ell}$ refines the group-relative page advantage:

$$
\widetilde A_{w,k,\ell}
=\frac{A_w}{NL}+R_{w,k,\ell},
\qquad
\sum_{k=1}^{N}\sum_{\ell=1}^{L}\widetilde A_{w,k,\ell}=A_w.
$$

Thus, local SID decisions receive differentiated updates without changing
the aggregate preference assigned to their complete page.

## Repository Layout

| Path | Purpose |
|---|---|
| `s2tpo/critic.py` | SID page-critic ensemble and portable checkpoints |
| `s2tpo/critic_objective.py` | Page, item, prefix, and ranking objectives |
| `s2tpo/page_rollout.py` | Trie decoding, complete-page sampling, and path evaluation |
| `s2tpo/credit.py` | Conservative values and conservation-aware credit transport |
| `s2tpo/tiger_actor.py` | TIGER-compatible token-level actor objective |
| `s2tpo/groups.py` | Validated complete-page candidate-group schema |
| `scripts/run_smoke_pipeline.py` | Offline end-to-end smoke pipeline |
| `scripts/run_kuairand_pipeline.py` | One-round KuaiRand/KuaiSim pipeline |
| `scripts/run_agent4rec_pipeline.py` | One-round MovieLens/Beauty pipeline |
| `scripts/evaluate_agent4rec.py` | External Agent4Rec Arena evaluation |
| `configs/kuairand_paper.json` | Paper-default hyperparameters |
| `configs/agent4rec_paper.json` | Agent4Rec paper-default hyperparameters |
| `tests/` | Formula, schema, rollout, critic, and actor tests |

## Installation

Python 3.9 or later is required.

```bash
git clone https://github.com/Applied-Machine-Learning-Lab/KDD2027-S2TPO.git S2T-PO
cd S2T-PO

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

GPU execution is recommended for the paper-scale configuration. The smoke
pipeline runs entirely on CPU.

## Quick Start

### Run the Tests

```bash
pytest -q
```

The suite verifies ensemble scoring, bootstrap critic training,
complete-page sampling, item-to-SID telescoping, zero-sum residuals,
page-advantage conservation, PPO clipping, KL regularization, and EMA
updates.

### Run the End-to-End Smoke Pipeline

```bash
python3 -m scripts.run_smoke_pipeline \
  --output-dir outputs/smoke
```

This command requires no dataset download or pretrained checkpoint. It
creates a tiny local TIGER model and synthetic sessions, then executes:

```text
critic training
  -> complete-page proposal and sampling
  -> PAD-to-complete path evaluation
  -> candidate-group validation
  -> actor optimization
  -> learner and EMA checkpoint saving
```

Successful completion ends with:

```text
smoke pipeline complete: .../outputs/smoke
```

## KuaiRand / KuaiSim

Given a complete-session rollout trace exported by KuaiSim, a SID mapping,
and a TIGER checkpoint exported in Hugging Face seq2seq format, one
paper-default post-training round runs with:

```bash
python3 -m scripts.run_kuairand_pipeline \
  --trace /path/to/rollout_trace.jsonl \
  --sid-mapping /path/to/video_sid_mapping.csv \
  --model /path/to/tiger_hf_checkpoint \
  --vocab-size 33 \
  --output-dir outputs/kuairand_pure \
  --device cuda:0
```

The command performs the following stages automatically:

1. converts complete simulator sessions into discounted critic targets;
2. trains the bootstrap long-term page-critic ensemble;
3. decodes a shared trie-constrained NTP proposal pool;
4. samples `W` distinct complete pages, with `N` items sampled without
   replacement inside every page;
5. evaluates all item paths in parallel and writes canonical actor groups;
6. updates the learner with token-level PPO and forward KL; and
7. saves both the learner and EMA rollout checkpoints.

Discounted return with `gamma=0.9` is used only for critic training. Reported
evaluation Total Reward remains the undiscounted sum over the session.

### Run Individual Stages

The one-command runner is composed of the following auditable commands.

Prepare simulator rollouts:

```bash
python3 -m scripts.prepare_kuairand_rollouts \
  --trace /path/to/rollout_trace.jsonl \
  --sid-mapping /path/to/video_sid_mapping.csv \
  --critic-sessions outputs/prepared/critic_sessions.jsonl \
  --contexts outputs/prepared/contexts.jsonl \
  --valid-sids outputs/prepared/valid_sids.json
```

Train the critic ensemble:

```bash
python3 -m scripts.train_s2tpo_critic \
  --sessions outputs/prepared/critic_sessions.jsonl \
  --output-dir outputs/critic \
  --vocab-size 33 \
  --sid-depth 4 \
  --ensemble-size 8 \
  --device cuda:0
```

Generate complete-page candidate groups:

```bash
python3 -m scripts.build_s2tpo_groups \
  --model /path/to/tiger_hf_checkpoint \
  --critic outputs/critic \
  --contexts outputs/prepared/contexts.jsonl \
  --valid-sids outputs/prepared/valid_sids.json \
  --output outputs/candidate_groups.jsonl \
  --group-size 8 \
  --page-size 6 \
  --proposal-count 192 \
  --device cuda:0
```

Validate groups and update the actor:

```bash
python3 -m scripts.validate_groups outputs/candidate_groups.jsonl

python3 -m scripts.train_s2tpo_actor \
  --model /path/to/tiger_hf_checkpoint \
  --groups outputs/candidate_groups.jsonl \
  --output-dir outputs/actor \
  --device cuda:0
```

## MovieLens / Beauty with Agent4Rec

Agent4Rec remains an external dependency. This repository provides the
paper-aligned TIGER training, session conversion, S2T-PO post-training, and
evaluation adapter without redistributing Agent4Rec source code, prompts, or
datasets.

The expected external dataset layout is:

```text
<agent4rec-root>/datasets/<dataset>/
|-- cf_data/
|   |-- train.txt
|   |-- valid.txt
|   `-- test.txt
`-- simulation/
    |-- all_personas_like_modify.csv
    |-- movie_detail.csv
    `-- user_statistic.csv
```

For MovieLens, `<dataset>` is `ml-1m`. The Beauty extension uses
`all-beauty` with the same Agent4Rec-compatible layout.

### 1. Train the Initial TIGER Policy

```bash
python3 -m scripts.train_agent4rec_tiger \
  --train-sequences <agent4rec-root>/datasets/ml-1m/cf_data/train.txt \
  --sid-mapping /path/to/ml1m_sid_mapping.csv \
  --output-dir outputs/agent4rec_ml1m/tiger_base \
  --model-size mini \
  --device cuda:0
```

The same command supports Beauty by changing the sequence and SID-mapping
paths. It trains next-SID prediction only and writes a Hugging Face seq2seq
checkpoint accepted directly by the public S2T-PO actor.

### 2. Collect Initial Interactive Sessions

Set the LLM credentials required by the external Agent4Rec installation,
then evaluate the initial TIGER policy:

```bash
export OPENAI_API_KEY=<your-api-key>

python3 -m scripts.evaluate_agent4rec \
  --agent4rec-root <agent4rec-root> \
  --dataset ml-1m \
  --model outputs/agent4rec_ml1m/tiger_base \
  --sid-mapping /path/to/ml1m_sid_mapping.csv \
  --eval-name tiger_base_rollout \
  --seeds 101 202 303 \
  --n-avatars 50 \
  --max-pages 20 \
  --page-size 6
```

The adapter delegates Recall, Precision, Click, Likes, and Exit Page to the
official Agent4Rec Arena. Each run also produces per-user behavior files used
as complete-session post-training data.

### 3. Run S2T-PO Post-Training

```bash
python3 -m scripts.run_agent4rec_pipeline \
  --run-dirs \
    <agent4rec-root>/storage/ml-1m/S2TPO/tiger_base_rollout_seed101 \
    <agent4rec-root>/storage/ml-1m/S2TPO/tiger_base_rollout_seed202 \
    <agent4rec-root>/storage/ml-1m/S2TPO/tiger_base_rollout_seed303 \
  --train-sequences <agent4rec-root>/datasets/ml-1m/cf_data/train.txt \
  --sid-mapping /path/to/ml1m_sid_mapping.csv \
  --model outputs/agent4rec_ml1m/tiger_base \
  --vocab-size 33 \
  --output-dir outputs/agent4rec_ml1m/s2tpo \
  --device cuda:0
```

Only six-item pages are accepted. The converter builds discounted session
returns, the same ensemble critic and complete-page groups used by KuaiRand,
and the same PAD-to-complete token advantages used by the paper objective.
For an item, calibration evidence is the sum of its alignment indicator,
click indicator, and rating divided by five. Immediate page utility is the
sum of page-level alignment rate, click rate, mean normalized rating, and a
continuation indicator. The critic target is the discounted sum of this
utility over the remaining session.

### 4. Evaluate the Trained Policy

```bash
python3 -m scripts.evaluate_agent4rec \
  --agent4rec-root <agent4rec-root> \
  --dataset ml-1m \
  --model outputs/agent4rec_ml1m/s2tpo/actor/rollout_ema \
  --sid-mapping /path/to/ml1m_sid_mapping.csv \
  --eval-name s2tpo \
  --seeds 101 202 303 \
  --n-avatars 50 \
  --max-pages 20 \
  --page-size 6 \
  --output outputs/agent4rec_ml1m/evaluation.json
```

Use the same Agent4Rec profiles, prompts, seeds, page budget, and termination
rules for every compared method. `--dry-run` validates adapter installation
and prints commands without issuing LLM requests.

## Data Contracts

### Critic Sessions

`scripts/train_s2tpo_critic.py` reads one
`s2tpo_critic_session_v1` JSON object per line.

| Field | Shape | Meaning |
|---|---:|---|
| `context_input_ids` | `[H]` | Tokenized user interaction history |
| `context_attention_mask` | `[H]` | Optional context mask |
| `page_sids` | `[N,L]` | Complete displayed recommendation page |
| `session_return` | scalar | Discounted return from the current page onward |
| `item_evidence` | `[N]` | Non-negative observed interaction evidence |

The item evidence is normalized into the paper's $\alpha_{t,k}$ weights.
An all-zero evidence row falls back to a uniform decomposition.

### Candidate Groups

`scripts/train_s2tpo_actor.py` reads one
`s2tpo_complete_page_group_v1` object per context.

| Field | Shape | Meaning |
|---|---:|---|
| `context_input_ids` | `[H]` | Shared request context |
| `context_attention_mask` | `[H]` | Context mask |
| `candidate_sids` | `[W,N,L]` | `W` distinct complete candidate pages |
| `critic_member_path_values` | `[W,N,L+1,C]` | Raw member values along every item path |

Precomputed advantages are intentionally excluded. The actor reconstructs
the conservative values, group-relative page advantages, pathwise credits,
and zero-sum residuals directly from raw critic-member predictions.

See [`docs/group_data_contract.md`](docs/group_data_contract.md) for complete
validation rules and legacy-record conversion.

## Paper Configuration

The defaults used by `configs/kuairand_paper.json` are:

| Component | Parameter | Value |
|---|---|---:|
| Page | Page size `N` | 6 |
| SID | SID depth `L` | 4 |
| Proposal | Proposal-pool size `P` | 192 |
| Group | Candidate pages `W` | 8 |
| Critic | Ensemble members `C` | 8 |
| Critic | Pessimism coefficient `beta` | 0.5 |
| Critic | Session discount `gamma` | 0.9 |
| Critic | Item calibration coefficient | 0.75 |
| Critic | Prefix calibration coefficient | 0.50 |
| Critic | Ranking coefficient | 0.15 |
| Actor | PPO clipping radius | 0.20 |
| Actor | Forward-KL coefficient | 0.03 |
| Rollout | EMA retention | 0.98 |

Group-relative advantages use population standard deviation and are not
clipped by default. All critic-derived quantities are detached during actor
optimization.

## Outputs

A KuaiRand run writes:

```text
outputs/kuairand_pure/
|-- prepared/
|   |-- critic_sessions.jsonl
|   |-- contexts.jsonl
|   `-- valid_sids.json
|-- critic/
|   |-- critic.pt
|   |-- critic_config.json
|   |-- run_config.json
|   `-- training_metrics.json
|-- candidate_groups.jsonl
`-- actor/
    |-- learner/
    |-- rollout_ema/
    |-- run_config.json
    `-- training_metrics.json
```

Datasets, rollout traces, model checkpoints, and generated outputs are
ignored by Git and should not be committed.

## Reproducibility Scope

This repository contains the paper-aligned S2T-PO method core together with
KuaiRand/KuaiSim and Agent4Rec post-training interfaces. KuaiSim, KuaiRand,
Agent4Rec, datasets, LLM services, and generated rollout traces remain
external assets governed by their respective projects and licenses.

The release intentionally excludes historical sign/monotonic objectives,
private production infrastructure, and the legacy slot-replacement schema,
because none is part of the reported method.

## License

S2T-PO is released under the [MIT License](LICENSE). External environments,
datasets, models, and dependencies retain their original licenses. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution details.
