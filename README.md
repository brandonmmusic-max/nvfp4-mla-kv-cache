# nvfp4_ds_mla — 4-bit (NVFP4) MLA latent KV cache for vLLM on SM120

A CUDA extension + integration patch that stores the MLA latent KV cache in **NVFP4 (4-bit)**
instead of fp8, for DeepSeek-family MLA models (DeepSeek-V4-Flash, GLM-5.2 / GlmMoeDsa) served
with vLLM-family forks on consumer/workstation Blackwell (SM120, e.g. RTX PRO 6000).

## Measured results (4x RTX PRO 6000 96GB, TP4/DCP4, GLM-5.2, maxlen 64K profile)
- **KV capacity: 554,666 tokens (fp8) -> 817,777 tokens (nvfp4) = +47%** at identical
  serving config (`GPU_UTIL=0.96 DCP=4 MTP num_spec=2`).
- **KV record: 432 bytes/token** (nope groups NVFP4-quantized with per-group scales;
  **RoPE dims kept bf16 bit-exact** — rope_bf16_exact=True in tests).
- CPU numerics harness: min cosine (nope) 0.9939, full-record 0.9941, decode-path 0.9941;
  max group MSE 0.295, mean 0.032.
- End-to-end: boots and serves with DCP4 + MTP speculative decode; output quality faithful
  to fp8 KV under recommended sampling `temperature=1.0, top_p=0.95, repetition_penalty=1.05`.

## Layout
- `kernel/nvfp4_mla_ext.cu` — the quantize/dequantize extension (prefill write path +
  decode gather path) for the MLA latent cache, NVFP4 e2m1 with per-16-group e4m3 scales.
- `kernel/setup_nvfp4_ext.py` — build script (targets `sm_120a`).
- `kernel/patch_custom_ops.py` — wires the ext into the fork's `_custom_ops` and registers
  the `nvfp4_ds_mla` kv-cache dtype.
- `docker/` — image build recipe used for the tested artifact.
- `tests/` — CPU numerics harness + GPU decode/prefill microtests + the GPU test runner.
- `docs/` — full build/validation logs: `FP4_MLA_PROGRESS.md` (implementation + numerics),
  `FP4_MLA_VERDICT.md` (A/B results, KV capacity numbers, and a bonus root-cause: a DCP
  replicated-draft KV-grouping default that silently costs ~94K tokens of fp8 KV capacity).

## Usage sketch
1. Build the ext inside your vLLM image: `python kernel/setup_nvfp4_ext.py build_ext --inplace`
   (see `docker/Dockerfile`).
2. Apply `kernel/patch_custom_ops.py` to register the op + dtype.
3. Serve with `--kv-cache-dtype nvfp4_ds_mla` (MLA attention backend required; tested with
   the B12X_MLA_SPARSE backend on SM120, TP4, DCP4, CUDA graphs on).

## Caveats
- Built and validated against a vLLM fork lineage (voipmonitor/vllm "eldritch" b12x family,
  June 2026); the `_custom_ops` patch points may need adaptation on other trees.
- MLA-latent-cache models only (DeepSeek V3/V4 family, GLM-5.2 DSA). Not GQA.
- Quality bar used: "same tokens as fp8 under production sampling", not bit-exact logits.
- SM120 (consumer/workstation Blackwell). SM100/Hopper untested.

Author: Brandon M. Music, 2026. Shared for research use; mit license — open an
issue if you want to use this commercially.

## Serving (Docker Hub image) — copy-paste

The ready-to-run image is **`verdictai/glm52-nvfp4-kv:v2`**. It bakes the *entire* proven-good
env block **and** the full `vllm serve` launch into the image itself, so there is nothing to
drop and no wrong flag to pass — just mount the checkpoint at `/model`:

```bash
hf download madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid --local-dir ./GLM-5.2-MXFP8-NVFP4-NF3-Hybrid
docker pull verdictai/glm52-nvfp4-kv:v2

docker run --rm --name glm52-nvfp4-kv \
  --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ./GLM-5.2-MXFP8-NVFP4-NF3-Hybrid:/model:ro -v glm52-nvfp4-cache:/cache \
  verdictai/glm52-nvfp4-kv:v2
```

Equivalents: `MODEL_DIR=./GLM-5.2-MXFP8-NVFP4-NF3-Hybrid ./serving/run.sh`, or
`MODEL_DIR=... docker compose -f serving/docker-compose.yml up -d` (the compose spells out the
same env + launch explicitly if you want to tune it). The line that turns on the 4-bit MLA KV
cache is `--kv-cache-dtype=nvfp4_ds_mla` (requires `--attention-backend=B12X_MLA_SPARSE`).

### If you got an OOM at model load with an earlier config, read this
On a 4x96GB rig the TP4 weights already occupy ~90GB/GPU, so load-time headroom is thin. Two
settings decide whether it boots — both are now correct in `:v2` / the compose here:
- **`VLLM_ENABLE_PCIE_ALLREDUCE=0` (backend `cpp`).** Turning the b12x PCIe all-reduce *on*
  allocates extra GPU workspace during load and is what caused the `Tried to allocate 1.77 GiB`
  OOM on GPU2/3. (nvfp4 uses *less* KV memory than fp8, so the OOM was never the KV cache.)
- **`--max-model-len 196608`** (~192K), the value this config was actually benched at. Some
  earlier notes said 250000; that was never the booted number.

### Measured on the published `:v2` — unmodified `llm_decode_bench.py`
`--port 5001 --concurrency 1 --contexts 0,8k,32k,128k` · 4× RTX PRO 6000 96GB · TP4/DCP4 ·
MTP-4 · util 0.96 · maxlen 196,608 · all-reduce off. **These are what you get pulling `:v2`
and running the script as-is.**

| Metric | nvfp4_ds_mla (published `:v2`) |
|---|---|
| KV pool | **407,808 tok** (+47% vs fp8 ~262K) |
| Decode t/s (conc 1) @ ctx 0 / 8k / 32k / 128k | **51.9 / 51.7 / 50.1 / 51.5** (flat) |
| Prefill t/s (`prompt_tokens ÷ TTFT`, N=1) @ 8k / 32k / 64k / 128k | **1,804 / 1,807 / 1,860 / 1,754** |
| Fidelity vs fp8 | greedy 10/12, 64K needle identical, KL~0.02 (see benchmarks/fidelity.md) |

> **Correction (2026-07-06) — read this.** An earlier version of this table listed prefill
> around **~2,900 t/s**. Those figures were NOT produced by `llm_decode_bench` — they came from
> inline boot-probe scripts I ran during bring-up, and they do **not** reproduce with the actual
> script on the shipped image. I shouldn't have presented them as benchmark results; the numbers
> above are the honest `llm_decode_bench` output on the published `:v2`. **The old ~3,000 figure
> was a different bench script, not a faster server** — proven: running the older
> `llm_decode_bench` (no version field) against this same `:v2` server still yields **~3,015 @ 8K**,
> while the current script (v0.4.24) yields **1,804** on the identical box. The old script (1)
> subtracts the ctx-0 baseline (`ctx / (ttft − baseline)`; it prints `baseline TTFT subtracted`)
> and (2) times a lean dedicated prefill probe (raw TTFT 2.84s), whereas the current script reports
> raw `tokens ÷ TTFT` from an integrated decode-scout that includes full generation setup (raw TTFT
> 4.55s). Both are legitimate — old ≈ GPU prefill-*compute* rate, new ≈ client-facing
> time-to-first-token — but they are **different rulers on the same hardware.** Server config is
> not the cause: an A/B with the PCIe all-reduce OFF vs ON (cpp) gave the same 1,804 vs 1,797.
> Decode (~51 t/s, flat 0→128k) is unaffected. See `benchmarks/speed.md`.

Rebuild the v2 image from source: `docker build -t verdictai/glm52-nvfp4-kv:v2 -f serving/Dockerfile.v2 .`
(the `:v2` layer is just baked env + CMD on top of `:v1`, which carries the kernel + readers).

Tuning notes: `--max-cudagraph-capture-size` 16 is conservative for memory headroom; 64 can be
faster if the GPUs have capture headroom. `--max-num-seqs` can drop to 1-2 for single-user to
free KV for longer `--max-model-len`. **DCP=1 (not the KV dtype) is the real prefill/decode
speed lever on a no-NVLink rig** — see `benchmarks/speed.md`.

## Benchmarks
- [`benchmarks/fidelity.md`](benchmarks/fidelity.md) — nvfp4 vs fp8 output fidelity (generation-lossless)
- [`benchmarks/speed.md`](benchmarks/speed.md) — full DCP sweep + fp8-vs-nvfp4 speed isolation.
  **Key finding: nvfp4 = +47% KV capacity at ~7% speed cost; DCP=1 (not the KV dtype) is the
  real speed lever on a no-NVLink rig (~75 vs ~53 t/s).**
