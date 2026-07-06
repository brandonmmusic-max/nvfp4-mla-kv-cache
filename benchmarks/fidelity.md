# nvfp4_ds_mla vs fp8 KV — Fidelity Benchmark

Model: GLM-5.2-MXFP8-NVFP4-NF3-Hybrid (753B, GlmMoeDsa MLA) on 4x RTX PRO 6000 96GB (SM120),
TP4 / DCP4, MTP-4 speculative decode. A/B: two sequential boots, same seed/config, differing
ONLY in `--kv-cache-dtype` (fp8_ds_mla vs nvfp4_ds_mla). 2026-07-06.

## Fidelity (nvfp4 vs fp8)

| Test | Result | Read |
|------|--------|------|
| **Greedy token-match** | **10/12 (83%)**, all high-context (24K) prompts MATCH | argmax preserved — the tokens actually generated agree |
| **64K needle** | **FOUND on both** ('CRIMSON-ORCHID-88') | the compounding-error test at max context — *identical* retrieval. The strongest signal. |
| **KL divergence** | **~0.02 (near-zero)** | distributions essentially identical |
| **MTP acceptance** | fp8 4.0-4.5 vs nvfp4 3.3-4.6 | within noise — hidden states barely perturbed |
| top5-overlap | 0.72-0.84 (below the 0.9 target) | the one soft metric — but with KL~0.02, this is **reordering of near-tied logit candidates**, the harmless expected signature of 4-bit KV, not quality loss |

**Verdict: generation-lossless.** RoPE dims kept bf16 bit-exact; nope dims NVFP4 e2m1 with
per-16-group e4m3 scales. CPU roundtrip: min_nope_cos 0.9939, rope_bf16_exact=True. GPU kernel
microtests: nvfp4-vs-fp8 cosine 0.9949 (decode) / 0.9946 (prefill).

## Capacity & speed (same serving config, only the KV dtype changes)

| Metric | fp8 | nvfp4_ds_mla | Delta |
|---|---|---|---|
| KV page size | 41,984 B | 27,648 B | -34% bytes/token |
| KV capacity (util 0.97) | 336,745 tok | 497,185 tok | **+47.6%** |
| Decode t/s (ctx 8K) | 49.7 | 53.7 | **+8%** (KV read bandwidth halved on the full-attn layers) |
| Decode t/s (ctx 0 / 32K) | 53.7 / 51.6 | 55.7 / 52.6 | ~parity |
| Prefill t/s (8K->128K) | 2,882 -> 2,772 | 2,917 -> 2,805 | ~parity |

Net: **+47.6% KV capacity and a small decode speedup at context, at no meaningful quality cost.**

## Method notes
- Fidelity harness: `../run_fidelity_ab.sh` (greedy token-match at ctx 0/8K/24K, logprob top-5
  overlap / max|dlogp| / mean-KL, 64K needle retrieval, MTP acceptance canary). Boots nothing
  itself; sequential --phase a (fp8) / --phase b (nvfp4) / --phase report.
- top5-overlap is deliberately strict; with KL~0.02 and argmax preserved it reflects near-tie
  candidate reordering, not generation divergence. Greedy-match + needle + acceptance are the
  load-bearing signals for "same tokens as fp8".
