# Speed & Capacity Benchmarks — nvfp4_ds_mla vs fp8, DCP sweep

Model: GLM-5.2-MXFP8-NVFP4-NF3-Hybrid (753B, GlmMoeDsa MLA). 4x RTX PRO 6000 96GB
(SM120), **PCIe Gen5, no NVLink**. TP4, MTP-4, `--max-num-batched-tokens=4096`,
util 0.96, `B12X_MLA_SPARSE`. Bench: `llm_decode_bench.py` (unmodified), concurrency 1,
30s/cell, max_tokens 8192. "Real-prose" = a 1500-token temp-1.0 thinking generation
(the representative single-user rate).

## TL;DR
- **nvfp4_ds_mla is a CAPACITY win, not a speed win.** It gives **+47% KV tokens** vs fp8,
  at a **~7% speed COST** — the 4-bit KV must be dequantized on every attention read, and on
  this hardware that dequant costs slightly more than the read-bandwidth it saves.
- **The real speed lever is DCP (decode-context-parallelism), because there is no NVLink.**
  DCP=4 shards KV across GPUs and does an all-gather/reduce-scatter every step; on a PCIe-only
  rig that cross-GPU traffic dominates. **DCP=1 removes it entirely: ~+40% decode, ~+45% prefill.**
- The tradeoff: DCP=1 does not shard KV, so KV capacity drops to ~1/4. It's a genuine
  **speed-vs-context** choice.

## DCP sweep (nvfp4_ds_mla, identical config, only DCP changes)
| DCP | Real-prose t/s | Prefill 8K t/s | KV pool (tok) | Max context |
|-----|----------------|----------------|---------------|-------------|
| 1   | **75.0**       | **4,638**      | 115,674       | ~110K |
| 2   | 53.6           | 3,930          | 225,703       | ~220K |
| 4   | ~53            | 3,185          | 387,154       | 250-350K |

DCP=2 is a poor middle ground — it keeps most of DCP=4's slowness while only recovering
~2x the KV. The useful choices are the endpoints.

## fp8 vs nvfp4 KV — clean isolation (DCP=1, only `--kv-cache-dtype` changes)
| Metric | fp8 | nvfp4_ds_mla | fp8 advantage |
|--------|-----|--------------|---------------|
| Prefill 8K / 16K / 32K (t/s) | 4,964 / 4,902 / 4,669 | 4,638 / 4,533 / 4,322 | **+7-8%** |
| Decode ctx 0 / 8K / 32K (t/s) | 76.5 / 67.5 / 68.6 | 70.6 / 66.6 / 62.5 | **+5-8%** |
| Real-prose (t/s) | 78.9 | 75.0 | +5% |
| KV pool | 78,464 tok (maxlen 64K*) | 115,674 tok (maxlen 100K) | nvfp4 +47% |
| MTP acceptance | 2.97 | 2.92 | ~equal |

*fp8 couldn't fit maxlen 100K in its (larger-page) pool at DCP=1, so it ran at 64K; the speed
comparison is unaffected (rate depends on the context processed, not the maxlen ceiling).

**Read:** nvfp4 trades ~7% attention speed for +47% KV capacity. If you need the capacity
(long context, high concurrency), that's an excellent trade. If you're context-light and want
raw speed, fp8 is marginally faster.

## Prefill lever ranking (measured, same nvfp4 image, 8K prefill t/s)
| Change | Prefill t/s | Effect |
|--------|-------------|--------|
| DCP4, batched 2048 | 2,789 | baseline |
| DCP4, batched 4096 | 3,185 | batched 2048->4096 = **+14%** |
| DCP1, batched 4096 | 4,638 | + DCP4->DCP1 = **+46%** |

**DCP is the dominant prefill lever, not `max-num-batched-tokens`** (which is only ~+14% here).

## Raw decode bench (synthetic, conc 1) — for completeness
| Config | ctx 0 | ctx 8K | ctx 32K |
|--------|-------|--------|---------|
| nvfp4 DCP4 b4096 | 88.5† | 87.3† | 53.5 |
| nvfp4 DCP2 b4096 | 53.6 | 48.7 | 47.7 |
| nvfp4 DCP1 b4096 | 70.6 | 66.6 | 62.5 |
| fp8   DCP1 b4096 | 76.5 | 67.5 | 68.6 |

†The nvfp4 DCP4 short-context synthetic cells (88.5/87.3) were an outlier that did not
reproduce; real-prose and the 32K cell put DCP=4 decode at ~53 t/s. Treat the DCP=4 real
decode rate as ~53, not ~88. Synthetic short-context spec-decode cells can spike; the
real-prose number is the honest single-user metric.
