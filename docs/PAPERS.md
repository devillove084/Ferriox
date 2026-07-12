# Ferriox Local Paper Index

All papers are stored as both the original PDF and a `pdftotext -layout` text
copy for local search. Design claims and adopted decisions are summarized in
[`CODESIGN.md`](./CODESIGN.md); this file is only the source index.

## Foundation

| Paper | arXiv | Local PDF | Searchable text | Use in Ferriox |
|---|---|---|---|---|
| DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | [2606.19348](https://arxiv.org/abs/2606.19348) | [PDF](./deepseek_v4.pdf) | [Text](./deepseek_v4.txt) | Compression, lightning indexer, shared compressed KV, window, sink, model dimensions |
| StreamIndex: Memory-Bounded Compressed Sparse Attention via Streaming Top-k | [2605.02568](https://arxiv.org/abs/2605.02568) | [PDF](./streamindex_csa.pdf) | [Text](./streamindex_csa.txt) | Bounded score tiles, causal partition-merge selection |
| FlashAttention | [2205.14135](https://arxiv.org/abs/2205.14135) | [PDF](./flash_attention_1.pdf) | [Text](./flash_attention_1.txt) | IO-aware exact online softmax |
| FlashAttention-2 | [2307.08691](https://arxiv.org/abs/2307.08691) | [PDF](./flash_attention_2.pdf) | [Text](./flash_attention_2.txt) | Work partitioning and backward recomputation |
| FlashAttention-3 | [2407.08608](https://arxiv.org/abs/2407.08608) | [PDF](./flash_attention_3.pdf) | [Text](./flash_attention_3.txt) | Asynchronous pipelines and low precision |
| FlashAttention-4 | [2603.05451](https://arxiv.org/abs/2603.05451) | [PDF](./flash_attention_4.pdf) | [Text](./flash_attention_4.txt) | Blackwell UMMA/TMEM pipeline and scalar/Tensor-Core co-design |
| Online Normalizer Calculation for Softmax | [1805.02867](https://arxiv.org/abs/1805.02867) | [PDF](./online_softmax.pdf) | [Text](./online_softmax.txt) | Online softmax recurrence used by sink-aware fixed-support attention |

## Block Selection and Sparse Kernels

| Paper | arXiv | Local PDF | Searchable text | Adopted lesson / boundary |
|---|---|---|---|---|
| MiniMax Sparse Attention | [2606.13392](https://arxiv.org/abs/2606.13392) | [PDF](./minimax_sparse_attention.pdf) | [Text](./minimax_sparse_attention.txt) | Closest precedent: post-score block max, final block top-k, reverse index, KV-outer gather, hot-block scheduling, detached KL. Ferriox does not copy its forced local block or full partial-output layout. |
| Native Sparse Attention | [2502.11089](https://arxiv.org/abs/2502.11089) | [PDF](./native_sparse_attention.pdf) | [Text](./native_sparse_attention.txt) | Hardware-aligned block selection and head-group-shared support. Its three gated attention branches differ from Ferriox's one sink-aware softmax. |
| FSA: An Alternative Efficient Implementation of Native Sparse Attention Kernel | [2508.18224](https://arxiv.org/abs/2508.18224) | [PDF](./flash_sparse_attention.pdf) | [Text](./flash_sparse_attention.txt) | KV-outer loop inversion, compact valid-query mapping, atomic-free partial/reduction. Full partial `dQ` is too large for Ferriox. |
| MoBA: Mixture of Block Attention for Long-Context LLMs | [2502.13189](https://arxiv.org/abs/2502.13189) | [PDF](./moba.pdf) | [Text](./moba.txt) | Learned block routing and full/sparse transition; model semantics differ from V4/Ferriox max pooling. |
| Optimizing Mixture of Block Attention / FlashMoBA | [2511.11571](https://arxiv.org/abs/2511.11571) | [PDF](./flash_moba.pdf) | [Text](./flash_moba.txt) | Streaming block top-k, histogram/scan/scatter reverse layout, gather-and-densify, KV-major atomic backward, block-size/SNR trade-off. |
| SeerAttention | [2410.13276](https://arxiv.org/abs/2410.13276) | [PDF](./seer_attention.pdf) | [Text](./seer_attention.txt) | Block-level self-distillation and teacher-statistic reuse; optional training reference, not the canonical router. |
| StreamKL | [2606.20005](https://arxiv.org/abs/2606.20005) | [PDF](./streamkl.pdf) | [Text](./streamkl.txt) | Online KL and tile-recomputed backward without materializing attention distributions. |

## Native and Distributed Scheduling

| Paper | arXiv | Local PDF | Searchable text | Use in Ferriox |
|---|---|---|---|---|
| Zero Bubble Pipeline Parallelism | [2401.10241](https://arxiv.org/abs/2401.10241) | [PDF](./zero_bubble_pipeline.pdf) | [Text](./zero_bubble_pipeline.txt) | Split activation-gradient `B` from parameter-gradient `W`; pipeline-only gains are not projected onto one GPU. |
| MG-WFBP | [1912.09268](https://arxiv.org/abs/1912.09268) | [PDF](./mg_wfbp.pdf) | [Text](./mg_wfbp.txt) | Merge small gradient messages to improve compute/communication overlap. |
| Priority-Based Parameter Propagation (P3) | [1905.03960](https://arxiv.org/abs/1905.03960) | [PDF](./p3.pdf) | [Text](./p3.txt) | Fine-grained synchronization ordered by next-use slack. |
| TicTac | [1803.03288](https://arxiv.org/abs/1803.03288) | [PDF](./tictac.pdf) | [Text](./tictac.txt) | Deterministic communication priorities and iteration-variance reduction. |

## Adjacent Inference and Approximation Work

| Paper | arXiv | Local PDF | Searchable text | Ferriox status |
|---|---|---|---|---|
| MInference 1.0 | [2407.02490](https://arxiv.org/abs/2407.02490) | [PDF](./minference.pdf) | [Text](./minference.txt) | Backend-dispatch reference for sparse prefill; not a training replacement. |
| Quest | [2406.10774](https://arxiv.org/abs/2406.10774) | [PDF](./quest.pdf) | [Text](./quest.txt) | Page metadata for future decode pruning. |
| SpargeAttention | [2502.18137](https://arxiv.org/abs/2502.18137) | [PDF](./sparge_attention.pdf) | [Text](./sparge_attention.txt) | Optional approximate online filtering profile. |
| Squeezed Attention | [2411.09688](https://arxiv.org/abs/2411.09688) | [PDF](./squeezed_attention.pdf) | [Text](./squeezed_attention.txt) | Centroid selection followed by exact selected attention; relevant to an alternative router. |
| Predict, Reuse, and Repair | [2606.30389](https://arxiv.org/abs/2606.30389) | [PDF](./prr.pdf) | [Text](./prr.txt) | Future speculative decode only; predicted extra blocks mean it is not strict exact top-k. |

## Reproducibility Note

Performance numbers in these papers use different models, head dimensions,
selection units, sequence lengths, precision formats, and GPUs. Ferriox cites
them as algorithmic and systems precedents. No paper speedup is treated as a
Ferriox prediction without an equal-shape local benchmark.
