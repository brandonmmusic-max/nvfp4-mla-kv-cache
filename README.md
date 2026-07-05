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

Author: Brandon M. Music, 2026. Shared for research use; no license granted yet — open an
issue if you want to use this commercially.
