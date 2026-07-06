# Speed & Capacity Benchmarks — nvfp4_ds_mla vs fp8, DCP sweep

Model: GLM-5.2-MXFP8-NVFP4-NF3-Hybrid (753B, GlmMoeDsa MLA). 4x RTX PRO 6000 96GB
(SM120), **PCIe Gen5, no NVLink**. TP4, MTP-4 unless a row says otherwise (the shipped `:v2` is
**MTP-3** since 2026-07-06 — see the MTP sweep below), `--max-num-batched-tokens=4096`,
util 0.96, `B12X_MLA_SPARSE`. Bench: `llm_decode_bench.py` (unmodified), concurrency 1,
30s/cell, max_tokens 8192. "Real-prose" = a 1500-token temp-1.0 thinking generation
(the representative single-user rate).

## Correction (2026-07-06) — measurement basis, read first
The **authoritative, reproducible** numbers for the published `verdictai/glm52-nvfp4-kv:v2`
image are the table right below. Where anything further down disagrees, trust this table.

- Some prefill figures I previously cited (~2,900 t/s) came from **inline boot-probe scripts,
  not `llm_decode_bench`.** They should never have been presented as bench results, and they do
  not reproduce on the shipped image.
- The DCP-sweep prefill absolutes below (3,185 / 3,930 / 4,638) came from an **older, different
  `llm_decode_bench` script** (no version field). Proven: running that old script against the
  shipped `:v2` server today still yields **~3,015 @ 8K**, while the current script (v0.4.24)
  yields **1,804** on the identical box — same hardware, different ruler. The old script (1)
  subtracts the ctx-0 baseline (`ctx / (ttft − baseline)`) and (2) times a lean dedicated prefill
  probe; the current one reports raw `tokens ÷ TTFT` from an integrated decode-scout. The sweep
  rows stay valid as a *relative* DCP comparison, but their absolute magnitude is on the old
  ruler. Server config is not the cause: an A/B (all-reduce OFF vs ON/cpp) gave the same 1,804 vs
  1,797.

### Published `:v2` (MTP-3, since 2026-07-06) — unmodified `llm_decode_bench.py` (conc 1, all-reduce off, maxlen 196,608)
| ctx  | Decode t/s | Prefill t/s (`tokens ÷ TTFT`, N=1) |
|------|-----------|------------------------------------|
| 0    | **67.0**  | —     |
| 8k   | **62.9**  | 1,796 |
| 32k  | **65.7**  | 1,793 |
| 64k  | —         | 1,852 |
| 128k | **63.2**  | 1,746 |

Real-prose (temp 1.0 chat): **~64 t/s**. KV pool **~407K tokens** (+47% vs fp8 ~262K).
(The same image at its original MTP-4 measured 51.9/51.7/50.1/51.5 · 1,804 — see the sweep below.)

### The MTP-depth sweep — why `:v2` ships MTP-3 (single-variable, same image/config)
| num_speculative_tokens | Decode 0 / 8k / 32k / 128k |
|---|---|
| **3 (shipped)** | **68.9 / 64.9 / 61.6 / 60.2** (confirm run: 67.0/62.9/65.7/63.2) |
| 4 | 51.9 / 51.7 / 50.1 / 51.5 |
| 5 | 49.4 / 45.9 / 45.7 / 47.3 |

Monotonic. Mechanism: each draft token is a serial MTP-head pass on a latency-bound rig, and
acceptance decays hard with depth (measured per-position at depth 3: 0.85 / 0.57-0.65 / 0.37-0.52;
at depth 4 the 4th position accepted only ~23%). Depth 3 stops paying a full pass for a token that
almost never survives verification. Prefill is untouched by MTP depth. Verification is exact, so
output quality is unchanged.

Also swept on this image (none helped): draft greedy-vs-probabilistic (noise; probabilistic kept),
`B12X_MLA_SM120_NUM_SPLITS` 1/2/4 (heuristic already optimal), `VLLM_RTX6K_FUSED_ALLREDUCE_ADD=1`
(worse at 128K), `--dcp-comm-backend a2a` (**broken** on this stack — boots but generates nothing).
Prefill was flat ~1,750-1,860 across every variant — on a no-NVLink rig it is pinned by the DCP4
PCIe collectives, not by any of these knobs.

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
| DCP4, batched 2048 | 2,789 | baseline (older basis; also graph64/util0.965 — see note) |
| DCP4, batched 4096 | 3,185 | **confounded** — this row also changed graph-cap + util, so the +14% is NOT batched-only |
| DCP1, batched 4096 | 4,638 | + DCP4->DCP1 = **+46%** |

**DCP is the dominant prefill lever, not `max-num-batched-tokens`.** The clean batched-only
isolation (pinpoint A→B in `configs.md`, all else fixed) is **only +2%** (2,887→2,940); the
"+14%" above is confounded by simultaneous graph-cap/util changes. All absolutes here are on the
older basis (see the Correction box up top) — the shipped `:v2` DCP4 measures ~1,804 @ 8K.

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
