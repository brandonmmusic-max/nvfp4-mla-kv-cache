# FP4 MLA verdict

Date: 2026-06-20

Image built/tested: `klc/glm52-nvfp4-dcpmtp:v2-trim`

Image id: `sha256:b3508a1d8e6e43495836fa062e81b1223b3b62f63d1467c92e7bef19a9a02710`

Test fence honored: only `glm52-nvfp4-*` / `fp4mla-*` containers, port `9401`; no stop/rm/run/restart of `glm52-9300-test` or `dsv4-9200-prod`, no use of ports `9300` or `9200`.

## Config used

Common production config:

`GPU_UTIL=0.96 MAXLEN=64000 DCP=4 MTP=1 NUM_SPEC=2 MAX_SEQS=8 GRAPH_CAP=32`

Important launch hygiene:

- `CUTE_DSL_ARCH=sm_120a`
- `NCCL_GRAPH_FILE`, `NCCL_GRAPH_DUMP_FILE`, and `VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` unset inside the container
- `--attention-backend B12X_MLA_SPARSE`
- `--moe-backend b12x`
- `--linear-backend auto`
- `--speculative-config {"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","moe_backend":"b12x","use_local_argmax_reduction":true}`

## Memory culprit

The large v2 KV-capacity regression was not the model weights and not primarily `nvfp4_ds_mla`.

The culprit was PR#30's DCP replicated-draft KV grouping being default-on for MTP/draft layers:

- MTP/draft layers with `layer_id >= num_hidden_layers` were marked `dcp_replicated=True` when `VLLM_DCP_SHARD_DRAFT` was unset.
- That split the scheduler-visible KV layout for the draft group from the target DCP-sharded group.
- The available KV bytes were roughly the same, but the per-token/page accounting got heavier, dropping fp8 KV capacity from the base class to about 469K tokens.

Evidence:

- First patched A/B with `VLLM_DCP_GLOBAL_TOPK=0` but replicated draft still default-on:
  - `Available KV cache memory: 7.19 GiB`
  - `GPU KV cache size: 468,884 tokens`
  - OOM during KV allocation
  - log: `fp4_mla_build/gpu_results/20260620T155444Z_trim_fp8_boot_sm120a/boot.log`
- After making replicated draft opt-in/default-off:
  - `Available KV cache memory: 7.19 GiB`
  - `GPU KV cache size: 554,666 tokens`
  - booted
  - log: `fp4_mla_build/gpu_results/20260620T160217Z_trim2_fp8_boot_sm120a/boot.log`
- Local base image under the same manual command also reported:
  - `Available KV cache memory: 7.19 GiB`
  - `GPU KV cache size: 554,666 tokens`
  - booted
  - log: `fp4_mla_build/gpu_results/20260620T161328Z_base_fp8_gate_sm120a/boot.log`

The user's reported base number was `564,825` tokens at `7.27 GiB`. My local base run had `7.19 GiB` available, so the absolute token count is lower, but v2-trim matches the local base exactly at `554,666`.

`VLLM_DCP_GLOBAL_TOPK=1` is not the main 94K-token regression. With replicated draft still off, exact global top-k booted at:

- `Available KV cache memory: 7.05 GiB`
- `GPU KV cache size: 543,492 tokens`
- log: `fp4_mla_build/gpu_results/20260620T161016Z_trim2_fp8_globaltopk_boot_sm120a/boot.log`

That is a smaller scratch/runtime cost, not the large per-token regression.

## What changed

Trimmed defaults in the hotfix image:

- `VLLM_DCP_GLOBAL_TOPK` now defaults off in:
  - `vllm/model_executor/layers/sparse_attn_indexer.py`
  - `vllm/v1/attention/backends/mla/indexer.py`
- DCP replicated-draft KV grouping is now opt-in via `VLLM_DCP_REPLICATE_DRAFT=1`; default unset follows the base footprint.
- `VLLM_DCP_SHARD_DRAFT=1` remains available for explicitly sharding the draft path.
- MTP draft config inherits `B12X_MLA_SPARSE` when the parent target uses that backend, avoiding the generic auto-selector failure for sparse MLA draft layers.
- `Dockerfile.hotfix` now copies and py-compiles the changed files into both `/opt/vllm` and `/opt/venv/lib/python3.12/site-packages`.

## Boot results

| Run | KV dtype | Image/config variant | Result | Available KV | GPU KV cache |
| --- | --- | --- | --- | ---: | ---: |
| v2 pre-trim replication path | `fp8_ds_mla` | top-k off, replicated draft still default-on | OOM | 7.19 GiB | 468,884 |
| v2-trim | `fp8_ds_mla` | replicated draft off/default, global top-k off/default | BOOTED | 7.19 GiB | 554,666 |
| base local control | `fp8_ds_mla` | base image, same manual command | BOOTED | 7.19 GiB | 554,666 |
| v2-trim exact global top-k | `fp8_ds_mla` | `VLLM_DCP_GLOBAL_TOPK=1`, replicated draft off | BOOTED | 7.05 GiB | 543,492 |
| v2-trim | `nvfp4_ds_mla` | replicated draft off/default, global top-k off/default | BOOTED | 7.19 GiB | 817,777 |

## Gate and compare

fp8 gate requested:

`what is the capital of kentucky?`, thinking on, `temperature=1.0`, `top_p=0.95`, `repetition_penalty=1.05`

Result: not passed.

Important: this failure reproduced on the base image under the same local launcher settings, so it is not introduced by the v2 memory trim.

- v2-trim fp8:
  - `reasoning_effort=high`: `finish=length`, empty answer, looped in reasoning
  - `reasoning_effort=none`, temp 0 probe: answered `Lexington`
  - `/tmp/cmp_test.py` status: `0`, but Kentucky and Marbury failed
  - log/output: `fp4_mla_build/gpu_results/20260620T160217Z_trim2_fp8_boot_sm120a/`
- base fp8 control:
  - `reasoning_effort=high`: `finish=length`, empty answer, looped in reasoning
  - `reasoning_effort=none`, temp 0 probe: answered `Lexington`
  - log/output: `fp4_mla_build/gpu_results/20260620T161328Z_base_fp8_gate_sm120a/`
- v2-trim fp8 with `VLLM_DCP_GLOBAL_TOPK=1`:
  - `reasoning_effort=high`: `finish=length`, empty answer, looped in reasoning
  - log/output: `fp4_mla_build/gpu_results/20260620T161016Z_trim2_fp8_globaltopk_boot_sm120a/`

Side-by-side `/tmp/cmp_test.py`:

| Prompt | fp8 result | nvfp4 result |
| --- | --- | --- |
| car 10 meters | both answer walk | both answer walk |
| four children/four oranges/knife | fp8 gives mixed/bad off answer, high answer usable; nvfp4 high answer usable | similar |
| Kentucky | fp8 says Lexington off, high loops to length | nvfp4 says Lexington off, high loops to length |
| Marbury | fp8 hallucinates/denies case | nvfp4 hallucinates/denies case |

nvfp4 + MTP=1 serve confirmation:

- `nvfp4_ds_mla` booted and served.
- `/tmp/cmp_test.py` completed with `STATUS=0`.
- KV pool: `817,777 tokens`.
- log/output: `fp4_mla_build/gpu_results/20260620T162317Z_trim2_nvfp4_boot_sm120a/`

## Final verdict

The memory regression is fixed in `klc/glm52-nvfp4-dcpmtp:v2-trim`.

The exact culprit was default-on DCP replicated-draft KV grouping from PR#30. Making that path opt-in/default-off restores fp8 KV capacity to the local base footprint (`554,666` tokens at `7.19 GiB`) and allows v2 to boot at the requested util/maxlen/DCP/MTP settings. The nvfp4 path also boots and serves with a larger KV pool (`817,777` tokens).

The requested Kentucky semantic gate is still failing, but it also fails on the base image with the same local launcher settings. That is a separate MTP/launcher/model-quality issue, not the v2 memory regression.
