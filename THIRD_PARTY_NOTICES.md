# Third-party notices

The S2T-PO source code in this repository is released under the MIT License.
The current source tree does not vendor source code, datasets, model weights,
or generated artifacts from the projects below. Their original license texts
are retained under `third_party/licenses/`. S2T-PO interoperates with their
public interfaces or uses them as experimental environments.

## TIGER

S2T-PO uses a TIGER-compatible Semantic Identifier generator. TIGER was
introduced in "Recommender Systems with Generative Retrieval". The current
repository provides an integration layer and does not redistribute an
external TIGER implementation or checkpoint.

- Paper: https://arxiv.org/abs/2305.05065

## KRLBenchmark / KuaiSim

The KuaiSim experiments use the public KRLBenchmark simulator codebase, which
is distributed under the Apache License 2.0. No KRLBenchmark source or data is
included in this repository.

- Repository: https://github.com/CharlieMat/KRLBenchmark
- License: Apache-2.0
- License text: [third_party/licenses/KRLBenchmark-LICENSE](third_party/licenses/KRLBenchmark-LICENSE)

## Agent4Rec

The Agent4Rec experiments use the public Agent4Rec environment, which is
distributed under the MIT License. No Agent4Rec source, prompts, data, or
generated responses are included in this repository.

- Repository: https://github.com/LehengTHU/Agent4Rec
- License: MIT
- License text: [third_party/licenses/Agent4Rec-LICENSE](third_party/licenses/Agent4Rec-LICENSE)

## Python dependencies

PyTorch and Hugging Face Transformers remain subject to their respective
upstream licenses. Installing a dependency does not relicense it under the
S2T-PO license.
