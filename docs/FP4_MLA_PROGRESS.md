# FP4 MLA Progress

## 2026-06-20 GPU prefill fixed and serving verified

- Respected hard fences: did not stop/rm/run/restart `glm52-9300-test` or `dsv4-9200-prod`; did not touch ports `9300` or `9200`. Only `fp4mla-nvfp4-9401` and port `9401` were restarted/used.
- Added `nvfp4_prefill_microtest.py`, a real GPU prefill/extend comparison test using vLLM `concat_and_cache_mla`, B12X `sparse_mla_extend_forward`, and identical BF16 synthetic Q/KV across `fp8_ds_mla`, `nvfp4_ds_mla`, and a PyTorch reference.
- Tightened the micro-test to fail on any non-finite output/cosine instead of falsely passing on NaN.
- Localized the prefill bug:
  - Tile 0 could use NVFP4 geometry, but the steady-state MG prefetch for tile 1+ did not pass `scale_format=t.scale_format`.
  - As a result, later NVFP4 prefill tiles fell back to GLM defaults and copied `528` bytes from `656`-byte records into `288`-byte NVFP4 shared-memory rows, corrupting MG attention state on long prompts.
  - The MG QK/PV path also needed the native NVFP4 E2M1/E4M3 dequant branch that decode already used.
- Fixed B12X sparse-MLA prefill/MG:
  - threaded `scale_format` through `api.py` and `prefill.py`,
  - forced `nvfp4_ds_mla` GLM-family prefill into `ComputeMode.BF16`,
  - added NVFP4 MG layout support and assertions (`kv_smem_stride=288`, Q-NoPE BF16 stride `520`),
  - added native NVFP4 MG QK-NoPE, RoPE load geometry for 432-byte records, BF16 probability staging, and native NVFP4 BF16 PV,
  - passed `scale_format=t.scale_format` on all GLM-family MG gather call sites.
- Removed the NVFP4 default decode-routing workaround in `b12x_mla_sparse.py`; live `fp4mla-nvfp4-9401` env has no `VLLM_B12X_MLA_NVFP4_DECODE_PREFILL_MAX_Q` and no `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE`.
- Rebuilt `klc/fp4mla-test:dev`:
  - Image id: `sha256:6c28c0fe7bdcec3fd0738e86c7a960b3c4cce5a370a81bfef1d26e8e70749563`
  - Build py-compile passed; build import check produced the expected no-GPU `libcuda.so.1` warning.
- Rebuilt-image GPU prefill micro-test passed at 1024 tokens:
  - `fp8_min=0.999656`, `nvfp4_min=0.995335`, `nvfp4_vs_fp8_min=0.994629`, `delta_nvfp4=3.46875`
  - Result: `PASS: nvfp4 prefill tracks fp8/reference and changes with Q`
- Restarted only `fp4mla-nvfp4-9401` on port `9401` with the rebuilt image. Server booted, loaded weights, captured CUDA graphs, and `/health` returned `ok`.
- Live API verification on `:9401`:
  - Raw `/v1/completions`, 494 prompt tokens ending in the Kentucky question: `finish=stop`, output `Answer: Frankfort.`
  - Reasoning-on chat with `reasoning_effort: "medium"` plus short-thinking instruction: parsed reasoning `The user is asking for the capital of Kentucky, which is Frankfort.` and content `Frankfort`.
  - Multi-fact long-context chat, 4,936 prompt tokens, thinking disabled for content answer: output `Meriden`.
- Wrote fixed report: `/home/brandonmusic/klc-linux/FP4_MLA_PREFILL_FIXED.md`.

## 2026-06-20 GPU decode fixed and serving verified

- Confirmed `/home/brandonmusic/klc-linux/FP4_GPU_WINDOW_OPEN` exists and used the GPU window.
- Respected hard fences: did not stop/rm/run/restart `glm52-9300-test` or `dsv4-9200-prod`; did not touch ports `9300` or `9200`. Only `fp4mla-nvfp4-9401` and port `9401` were used.
- Added `nvfp4_decode_microtest.py`, a real GPU one-step decode localization test using:
  - vLLM `concat_and_cache_mla(..., "fp8_ds_mla" / "nvfp4_ds_mla", ...)`,
  - B12X `sparse_mla_decode_forward`,
  - identical synthetic BF16 Q + latent KV for fp8/nvfp4/reference comparison.
- Localized the bug:
  - Cache writer/gather bytes were valid: CPU decode of the CUDA-written 432-byte records reconstructed the latent vector at about 0.995 cosine and preserved RoPE BF16 bytes.
  - The failing stage was the NVFP4 in-register dequant as wired into QK/PV: before the fix, top-1 nvfp4 gather/PV produced a zero vector while fp8 matched reference.
- Fixed `_nvfp4_pair_bfloat2` in `decode_math.py`:
  - replaced the old packed dequant helper path with native `fp4_decode_2` + `f16x2_to_f32x2`,
  - loaded the E4M3 group-16 scale byte,
  - converted scale with `cvt_e4m3_to_f32_via_f16`,
  - multiplied in fp32 and packed the pair to BF16 for the BF16 MMA path.
- Fixed serving route for short nvfp4 prompts:
  - `nvfp4_ds_mla` now defaults short extend/prompt batches through the validated decode kernel instead of the GLM/FP8-shaped MG prefill path.
  - Added bounded defaults in `b12x_mla_sparse.py`: `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1` and `VLLM_B12X_MLA_NVFP4_DECODE_PREFILL_MAX_Q=128` behavior for `nvfp4_ds_mla`.
- Rebuilt `klc/fp4mla-test:dev`:
  - Image id: `sha256:e3332cd4c5d05e887a4b104fa72cdcfc34240a4612a046da6b5f1e70a9d80d08`
  - Build py-compile passed; build import check produced the expected no-GPU `libcuda.so.1` warning.
- GPU micro-test passed inside the rebuilt image:
  - `top1_gather_pv`: `nvfp4_min=0.995275`, `nvfp4_vs_fp8_min=0.994672`
  - `focus_token_5`: `nvfp4_min=0.995494`, `delta_nvfp4=3.51562`
  - `focus_token_42`: `nvfp4_min=0.995218`, `delta_nvfp4=4.0625`
  - Result: `PASS: nvfp4 one-step decode tracks fp8/reference and changes with Q`
- Restarted only `fp4mla-nvfp4-9401` on port `9401` with the rebuilt image. Server booted, loaded weights, captured CUDA graphs, and `/v1/models` returned `glm-5.2-nvfp4`.
- Required raw completion now returns coherent text containing Frankfort:
  - Prompt: `The capital of Kentucky is`
  - Output text: ` Frankfort. It is a small town in the mountains of eastern Kentucky. It is known for its famous cheese, which is called "cheese". It is`
- Chat endpoint now returns final content containing Frankfort when `reasoning_effort` is set to `none`:
  - User: `what is the capital of kentucky?`
  - Output content includes: `its **capital city** is **Frankfort**`
- Wrote fixed report: `/home/brandonmusic/klc-linux/FP4_MLA_DECODE_FIXED.md`.

## 2026-06-20 Layer-4 decode implementation

- Confirmed `/home/brandonmusic/klc-linux/FP4_GPU_WINDOW_OPEN` is absent. No GPU work was run in this phase.
- Implemented the `nvfp4_ds_mla` Layer-4 Strategy-A decode branch in the local b12x tree and synced it into the docker build context.
  - Added decode-local BF16 Q-NoPE staging for NVFP4.
  - Added in-register NVFP4 NoPE dequant helpers using packed E2M1 data plus E4M3 group-16 scales.
  - Added BF16 QK-NoPE for NVFP4 through `mma_m16n8k16_f32_bf16`.
  - Added BF16 PV/NoPE for NVFP4, reading BF16 probabilities from `sm_p_full` and dequantizing V in registers.
  - Kept RoPE as direct BF16.
  - Removed the `nvfp4_ds_mla` fail-closed guards from the b12x sparse-MLA API and SM120 launcher.
- Updated NVFP4 decode constants:
  - gmem record stride: 432 bytes/token.
  - staged KV row: 288 bytes/token.
  - BF16 Q-NoPE stride: 520 elements (`512 + 8`).
  - logical FP4 NoPE K steps: 8 (`512/64`).
  - NVFP4 smem layout check: total=79232, q_bytes=16640, kv_buf=18432, bulk_tx=26624.
- Extended `test_nvfp4_mla_cpu.py` to decode-dequant the 432-byte record by FP4 nibbles and E4M3 scales, then assert per-64-dim step cosine against the BF16 latent reference.

### 2026-06-20 CPU verification

- Python syntax/import checks passed with `CUDA_VISIBLE_DEVICES=''` for edited local and docker-context b12x files plus the CPU numeric test.
- CPU numeric round-trip plus decode-style dequant passed:
  - Command: `CUDA_VISIBLE_DEVICES='' python3 test_nvfp4_mla_cpu.py`
  - Result: `PASS nvfp4_ds_mla CPU numerics: tokens=257 record_bytes=432 max_group_mse=0.294820 mean_group_mse=0.031773 min_nope_cos=0.993945 min_full_cos=0.994147 min_step_cos=0.991351 min_decode_full_cos=0.994147 rope_bf16_exact=True decode_rope_bf16_exact=True`
- Rebuilt distinct non-prod image tag: `klc/fp4mla-test:dev`
  - Image id: `sha256:44d0220d4c22e29e33637c01adce76c3287e8aa5628afe1fb70e144f3c9115a1`
  - Build used `CMAKE_CUDA_FLAGS="-gencode arch=compute_120f,code=sm_120"` and did not start a GPU container.
  - Build import check warned that `libcuda.so.1` was unavailable inside the no-GPU build environment; the custom-op wrapper attribute was present.
- Wrote readiness handoff: `/home/brandonmusic/klc-linux/FP4_MLA_DECODE_READY.md`.
- GPU retrieval/decode gates remain deferred until `/home/brandonmusic/klc-linux/FP4_GPU_WINDOW_OPEN` exists.

## 2026-06-19 CPU-only pass

- Confirmed `/home/brandonmusic/klc-linux/FP4_GPU_WINDOW_OPEN` is absent. No GPU work is allowed in this phase.
- Confirmed protected container state without touching it: `glm52-9300-test` is running; `dsv4-9200-prod` is exited. No protected container or port was modified.
- Created requested workspace: `/home/brandonmusic/klc-linux/fp4_mla_build`.
- Extracted source from stopped temporary container `fp4src` using image `voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618`.
- Removed temporary container `fp4src`.
- Local source roots:
  - vLLM: `/home/brandonmusic/klc-linux/fp4_mla_build/vllm`
  - b12x CuteDSL package: `/home/brandonmusic/klc-linux/fp4_mla_build/site-packages/b12x`
  - DeepGEMM reference package: `/home/brandonmusic/klc-linux/fp4_mla_build/site-packages/deep_gemm`

### CPU-only implementation status

- Layer 1/2 Python dtype plumbing is complete in the extracted local tree.
  - Added `nvfp4_ds_mla` as a cache dtype.
  - Classified it as quantized and `torch.uint8`.
  - Preserved `nvfp4_ds_mla` only for `B12X_MLA_SPARSE`; `FLASHMLA_SPARSE` remains on `fp8_ds_mla`.
  - Added 432-byte cache shape/page-size accounting for the packed MLA latent.
- Layer 3 CUDA store-kernel work is complete enough to build.
  - Added a standalone `concat_and_cache_nvfp4_mla` CUDA op in the extracted tree.
  - The new store writes the 432-byte record: 256B E2M1 NoPE data, 32B E4M3 group-16 scales, 16B pad, 128B BF16 RoPE.
  - Registered the op in stable torch bindings and added the Python wrapper/dispatch.
- Layer 4 b12x CuteDSL work is a fail-closed scaffold, not a complete deployable decode.
  - Added `ScaleFormat.NVFP4_E4M3`, 432-byte IO gather constants, scratch/caps threading, and vLLM b12x dispatch of `scale_format=2`.
  - The SM120 b12x launcher deliberately raises for `nvfp4_ds_mla` until the FP4 E2M1/E4M3 in-register dequant plus BF16 QK/PV math branch is implemented.
  - This guard prevents silently running the existing GLM fp8 decode math on FP4 bytes.
  - Follow-up decode inspection: the current decode path stages Q NoPE as fp8 in `q_fp8` smem and has no decode-local BF16-QK NoPE path. The BF16-QK primitive exists in prefill/MG code, but a safe decode Strategy A needs either new BF16 Q staging/register loading or a different FP4-native QK path. That is the remaining no-GPU code blocker before GPU smoke tests can pass.

### CPU verification

- Python syntax checks passed with `CUDA_VISIBLE_DEVICES=''` for the modified vLLM, b12x, and CPU numeric-test files.
- CPU numeric round-trip test passed against DeepGEMM reference:
  - Command: `CUDA_VISIBLE_DEVICES='' python3 test_nvfp4_mla_cpu.py`
  - Result: `PASS nvfp4_ds_mla CPU numerics: tokens=257 record_bytes=432 max_group_mse=0.294820 mean_group_mse=0.031773 min_nope_cos=0.993945 min_full_cos=0.994147 rope_bf16_exact=True`
- Built distinct non-prod image tag: `klc/fp4mla-test:dev`
  - Image id: `sha256:9443097bde4acf141b4f1db6dda3957061aef17eea48bc2f9b760d658382b26f`
  - Build used `CMAKE_CUDA_FLAGS="-gencode arch=compute_120f,code=sm_120"` and did not start a GPU container.
  - Build import check warned that `libcuda.so.1` was unavailable inside the no-GPU build environment; the custom-op wrapper attribute was present.

### Deferred GPU work

- Wrote guarded later-run script: `/home/brandonmusic/klc-linux/fp4_mla_build/FP4_MLA_GPU_TESTS.sh`.
  - It exits unless `/home/brandonmusic/klc-linux/FP4_GPU_WINDOW_OPEN` exists.
  - It refuses protected container names `glm52-9300-test` and `dsv4-9200-prod`.
  - It refuses protected ports `9200` and `9300`; defaults are `9401` and `9402`.
  - It includes commands for build/start/decode-equivalence/retrieval/RULER/speed stages.
- Current gate state after the CPU pass: `/home/brandonmusic/klc-linux/FP4_GPU_WINDOW_OPEN` is still absent.
- Protected state after the CPU pass: `glm52-9300-test` is still running; `dsv4-9200-prod` is still exited. No protected container or port was modified.
- `FP4_MLA_DEPLOYABILITY_VERDICT.md` was not written because the section 5 GPU retrieval gate has not run.
