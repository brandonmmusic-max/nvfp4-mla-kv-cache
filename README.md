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

## Serving (Docker Hub image)

A ready-to-run image is published at **`verdictai/glm52-nvfp4-kv:v1`** (base layers
cross-mounted from `madeby561/vllm-glm52-nvfp4-nf3-hybrid:v2`; this image adds the NVFP4
MLA KV writer `.so` + the ported b12x MLA nvfp4 readers).

```bash
hf download madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid --local-dir ./GLM-5.2-MXFP8-NVFP4-NF3-Hybrid
docker pull verdictai/glm52-nvfp4-kv:v1
MODEL_DIR=./GLM-5.2-MXFP8-NVFP4-NF3-Hybrid docker compose -f serving/docker-compose.yml up -d
```

The only line that turns on the 4-bit MLA KV cache is:
```
--kv-cache-dtype=nvfp4_ds_mla   (with --attention-backend=B12X_MLA_SPARSE)
```

### Measured on this config (4x RTX PRO 6000 96GB, TP4/DCP4, MTP-4, util 0.96)
| Metric | fp8 | nvfp4_ds_mla |
|---|---|---|
| KV pool | ~262K tok | **387K tok** (+47%) |
| Decode t/s (real-prose, single-user) | ~53 | ~53 (DCP4) / **75 (DCP1)** |
| Prefill t/s (8K->128K) | ~2,880 -> 2,770 | 3,185 -> 2,842 |
| Real-prose t/s (temp 1.0) | ~53 | ~55 |
| Fidelity vs fp8 | — | greedy 10/12, 64K needle identical, KL~0.02 (see benchmarks/fidelity.md) |

Rebuild from source: `docker build -t glm52-nvfp4-kv -f serving/Dockerfile .`

Tuning notes: `--max-cudagraph-capture-size` 16 is conservative for memory headroom; **64
is faster** if the GPUs have capture headroom (nvfp4's smaller KV pages usually free enough).
`--max-num-seqs` can drop to 1-2 for single-user to free KV for longer `--max-model-len`.

## Benchmarks
- [`benchmarks/fidelity.md`](benchmarks/fidelity.md) — nvfp4 vs fp8 output fidelity (generation-lossless)
- [`benchmarks/speed.md`](benchmarks/speed.md) — full DCP sweep + fp8-vs-nvfp4 speed isolation.
  **Key finding: nvfp4 = +47% KV capacity at ~7% speed cost; DCP=1 (not the KV dtype) is the
  real speed lever on a no-NVLink rig (~75 vs ~53 t/s).**
