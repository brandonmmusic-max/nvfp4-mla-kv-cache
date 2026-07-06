# Benchmark Config Traceability

Every number in this repo, mapped to the **exact** serving config that produced it. The
published `serving/docker-compose.yml` is ONE specific config; most benchmark runs below
deliberately deviate from it (DCP sweep, fp8-vs-nvfp4 isolation, batched-tokens pinpoint,
CPU-throttle proof). **Do not attribute a benchmark number to the published compose unless
the "== published compose" column says YES.**

Hardware for all runs: 4x RTX PRO 6000 96GB (SM120), PCIe Gen5 no NVLink,
**AMD Threadripper PRO 9965WX (24c/48t, 5.49 GHz)**. Bench: `llm_decode_bench.py`
(unmodified), concurrency 1, 30s/cell, max_tokens 8192, contexts 0/8K/32K unless noted.

Published config (`verdictai/glm52-nvfp4-kv:v2`, `serving/docker-compose.yml`):
`nvfp4_ds_mla · DCP4 · util0.96 · maxlen196608 · seqs4 · batched4096 · graph16 · MTP3 · PCIe all-reduce OFF`

> **Correction (2026-07-06).** This file previously (a) described the published config as
> `:v1 / maxlen 250000` — that config OOM'd at load and never shipped; the shipped image is
> `:v2 / maxlen 196608`; and (b) listed the DCP4 "published" prefill as **3,185 t/s**. That 3,185
> is NOT the published number: it came from an **older, different `llm_decode_bench` script** (no
> version field) that subtracts the ctx-0 baseline and times a lean prefill probe. Proven: that old
> script run against the shipped `:v2` server today still yields **~3,015 @ 8K**, while the current
> script (v0.4.24) yields **1,804** on the identical box — same hardware, different ruler. The
> current script reports raw `tokens ÷ TTFT` from an integrated decode-scout. Server config is not
> the cause: an A/B with the PCIe all-reduce off vs on (cpp) gave the same 1,804 vs 1,797 @ 8K. The
> DCP-sweep rows below remain valid as a *relative* comparison but their prefill absolutes are on
> the old ruler, not the shipped image.

| Run | Image | KV dtype | DCP | maxlen | batched | util | graph | MTP | == published | Result (decode ctx0/8K/32K · prefill 8K · real-prose) |
|-----|-------|----------|-----|--------|---------|------|-------|-----|--------------|-------------------------------------------------------|
| **nvfp4 DCP4 MTP-3 (published v2, 2026-07-06 update)** | verdictai:v2 | nvfp4_ds_mla | 4 | 196608 | 4096 | 0.96 | 16 | **3** | **YES (exact)** | **67.0/62.9/65.7 (128K: 63.2) · 1,796 · 63.6** |
| nvfp4 DCP4 MTP-4 (v2 as first published) | verdictai:v2 (pre-update) | nvfp4_ds_mla | 4 | 196608 | 4096 | 0.96 | 16 | 4 | no (superseded by MTP-3) | 51.9/51.7/50.1/51.5 · 1,804 · ~54 |
| nvfp4 DCP4 (older run, superseded) | base v2 | nvfp4_ds_mla | 4 | 250000 | 4096 | 0.96 | 16 | 4 | no (older basis, not shipped) | 88.5†/87.3†/53.5 · 3,185‡ · ~53 |
| nvfp4 DCP1              | verdictai:v1 | nvfp4_ds_mla | 1 | 100000 | 4096 | 0.96  | 16 | 4 | no (DCP1, 100K) | 70.6/66.6/62.5 · 4,638 · 75.0 |
| nvfp4 DCP2              | verdictai:v1 | nvfp4_ds_mla | 2 | 100000 | 4096 | 0.96  | 16 | 4 | no (DCP2, 100K) | 53.6/48.7/47.7 · 3,930 · 53.6 |
| fp8 DCP1 (isolation)    | verdictai:v1 | fp8          | 1 | 64000  | 4096 | 0.96  | 16 | 4 | no (fp8, DCP1)  | 76.5/67.5/68.6 · 4,964 · 78.9 |
| pinpoint A              | base v2      | fp8          | 4 | 120000 | 2048 | 0.96  | 16 | 4 | no (base, fp8, b2048) | 53.7/49.8/48.7 · 2,887 · — |
| pinpoint B              | base v2      | fp8          | 4 | 120000 | 4096 | 0.96  | 16 | 4 | no (base, fp8)  | 52.6/50.7/48.7 · 2,940 · — |
| pinpoint C             | verdictai:v1 | fp8          | 4 | 120000 | 4096 | 0.96  | 16 | 4 | no (fp8 not nvfp4) | 54.6/50.7/51.6 · 2,932 · — |
| graph-cap 64            | verdictai:v1 | nvfp4_ds_mla | 4 | 250000 | 2048 | 0.965 | 64 | 4 | no (util/graph/b2048) | 52.7/52.6/52.7 · 2,789 · — |

†The nvfp4-DCP4 short-context synthetic cells (88.5/87.3) did not reproduce; real-prose and the
32K cell put DCP=4 decode at ~53 t/s. Treat DCP=4 real decode as ~53. See `speed.md`.

‡3,185 came from an older, different bench script (baseline-subtracted formula + lean prefill
probe). That old script on this same `:v2` server today still gives **~3,015 @ 8K**, vs **1,804**
for the current v0.4.24 script (raw `tokens ÷ TTFT`) — same hardware, different ruler. Robust to
PCIe all-reduce on/off (A/B: 1,804 vs 1,797). See the Correction box above.

## Tier-1 single-variable A/B sweep (2026-07-06, all on verdictai:v2, 0.4.24 bench)
One variable changed per run vs the MTP-4 baseline (51.9/51.7/50.1/51.5 · 1,804):
- **MTP depth (the winner): 3 → 68.9/64.9/61.6/60.2 · 4 → baseline · 5 → 49.4/45.9/45.7/47.3.**
  Monotonic 3 > 4 > 5: each draft step is a serial pass on a latency-bound rig and the 4th/5th
  tokens accept at only ~23%/worse (confirmed per-position acceptance 0.85/0.57-0.65/0.37-0.52 at
  depth 3, mean length 2.77-3.07). Confirm run reproduced: 67.0/62.9/65.7/63.2, prose 63.6 t/s.
- draft greedy vs probabilistic: 52.9 vs 51.9 ctx0 = noise; probabilistic kept (production sampler).
- `B12X_MLA_SM120_NUM_SPLITS` 1/2/4: no win (wave-balanced heuristic already optimal; splits=1
  hurt 8K prefill: 1,099).
- `VLLM_RTX6K_FUSED_ALLREDUCE_ADD=1`: 55.0 ctx0 but 45.9 @128K = net worse; stays 0.
- `--dcp-comm-backend a2a`: **broken** on this stack (boots, generates 0 tokens); use ag_rs.
- Prefill: flat 1,750-1,860 across every config — pinned by DCP4 PCIe collectives on no-NVLink.

## What each run isolated
- **DCP sweep (DCP1/2/4)**: DCP=1 is the big speed lever on a no-NVLink rig (drops the cross-GPU
  KV all-reduce): ~75 vs ~53 t/s decode, 4,638 vs 3,185 prefill — at the cost of KV capacity.
- **fp8-vs-nvfp4 at DCP1**: fp8 ~7% faster (nvfp4's 4-bit dequant costs more than the bandwidth it
  saves). nvfp4 is a **capacity** win (+47% KV), not a speed win.
- **Pinpoint A/B/C (fp8, DCP4)**: `max-num-batched-tokens` 2048->4096 = +2% only; the nvfp4 overlay
  is transparent to fp8 (2,932 ≈ 2,940). Neither is a prefill lever.
- **CPU note (RETRACTED — unverified).** An earlier version of this file claimed the Threadripper
  PRO CPU explained a ~2,900-vs-~1,600 prefill edge over other rigs. That was built on the ~2,900
  boot-probe / older-basis numbers that do **not** reproduce on the shipped image (which measures
  ~1,804 with the unmodified script), and the CPU effect was **never actually isolated** on GLM —
  every run here was on the same machine, so nothing in this data attributes prefill to the CPU.
  Treat any CPU-driven prefill claim as unproven.
