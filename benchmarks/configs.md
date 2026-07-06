# Benchmark Config Traceability

Every number in this repo, mapped to the **exact** serving config that produced it. The
published `serving/docker-compose.yml` is ONE specific config; most benchmark runs below
deliberately deviate from it (DCP sweep, fp8-vs-nvfp4 isolation, batched-tokens pinpoint,
CPU-throttle proof). **Do not attribute a benchmark number to the published compose unless
the "== published compose" column says YES.**

Hardware for all runs: 4x RTX PRO 6000 96GB (SM120), PCIe Gen5 no NVLink,
**AMD Threadripper PRO 9965WX (24c/48t, 5.49 GHz)**. Bench: `llm_decode_bench.py`
(unmodified), concurrency 1, 30s/cell, max_tokens 8192, contexts 0/8K/32K unless noted.

Published `serving/docker-compose.yml`:
`image=verdictai/glm52-nvfp4-kv:v1 · nvfp4_ds_mla · DCP4 · util0.96 · maxlen250000 · seqs4 · batched4096 · graph16 · MTP4`

| Run | Image | KV dtype | DCP | maxlen | batched | util | graph | MTP | == published | Result (decode ctx0/8K/32K · prefill 8K · real-prose) |
|-----|-------|----------|-----|--------|---------|------|-------|-----|--------------|-------------------------------------------------------|
| nvfp4 DCP4 (baseline)   | verdictai:v1 | nvfp4_ds_mla | 4 | 250000 | 4096 | 0.96  | 16 | 4 | **YES (exact)** | 88.5†/87.3†/53.5 · 3,185 · ~53 |
| nvfp4 DCP1              | verdictai:v1 | nvfp4_ds_mla | 1 | 100000 | 4096 | 0.96  | 16 | 4 | no (DCP1, 100K) | 70.6/66.6/62.5 · 4,638 · 75.0 |
| nvfp4 DCP2              | verdictai:v1 | nvfp4_ds_mla | 2 | 100000 | 4096 | 0.96  | 16 | 4 | no (DCP2, 100K) | 53.6/48.7/47.7 · 3,930 · 53.6 |
| fp8 DCP1 (isolation)    | verdictai:v1 | fp8          | 1 | 64000  | 4096 | 0.96  | 16 | 4 | no (fp8, DCP1)  | 76.5/67.5/68.6 · 4,964 · 78.9 |
| pinpoint A              | base v2      | fp8          | 4 | 120000 | 2048 | 0.96  | 16 | 4 | no (base, fp8, b2048) | 53.7/49.8/48.7 · 2,887 · — |
| pinpoint B              | base v2      | fp8          | 4 | 120000 | 4096 | 0.96  | 16 | 4 | no (base, fp8)  | 52.6/50.7/48.7 · 2,940 · — |
| pinpoint C             | verdictai:v1 | fp8          | 4 | 120000 | 4096 | 0.96  | 16 | 4 | no (fp8 not nvfp4) | 54.6/50.7/51.6 · 2,932 · — |
| graph-cap 64            | verdictai:v1 | nvfp4_ds_mla | 4 | 250000 | 2048 | 0.965 | 64 | 4 | no (util/graph/b2048) | 52.7/52.6/52.7 · 2,789 · — |

†The nvfp4-DCP4 short-context synthetic cells (88.5/87.3) did not reproduce; real-prose and the
32K cell put DCP=4 decode at ~53 t/s. Treat DCP=4 real decode as ~53. See `speed.md`.

## What each run isolated
- **DCP sweep (DCP1/2/4)**: DCP=1 is the big speed lever on a no-NVLink rig (drops the cross-GPU
  KV all-reduce): ~75 vs ~53 t/s decode, 4,638 vs 3,185 prefill — at the cost of KV capacity.
- **fp8-vs-nvfp4 at DCP1**: fp8 ~7% faster (nvfp4's 4-bit dequant costs more than the bandwidth it
  saves). nvfp4 is a **capacity** win (+47% KV), not a speed win.
- **Pinpoint A/B/C (fp8, DCP4)**: `max-num-batched-tokens` 2048->4096 = +2% only; the nvfp4 overlay
  is transparent to fp8 (2,932 ≈ 2,940). Neither is a prefill lever.
- **CPU note**: the base image gives ~2,900 prefill on this Threadripper PRO rig vs ~1,600 others
  report on the same image; prefill is a CPU<->GPU per-chunk ping-pong that `--async-scheduling`
  only hides when the CPU keeps the GPUs fed.
