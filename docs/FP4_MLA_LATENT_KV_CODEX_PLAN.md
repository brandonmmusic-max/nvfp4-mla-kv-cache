# Codex Build Plan — `nvfp4_ds_mla`: 4-bit NVFP4 KV cache for the GLM-5.2 / DeepSeek-V4 MLA *latent* on sm_120 (b12x vLLM)

**Owner:** codex (autonomous build → smoke-test → iterate → deployability verdict)
**Status:** research/design COMPLETE (this doc). Implementation NOT started.
**Image:** `voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618`
**Hardware:** 4× RTX PRO 6000 (Blackwell sm_120, native NVFP4 tensor cores). GPUs are SHARED with the prod model — request a GPU window before each on-GPU smoke test (see §6).

---

## 1. Mission
Halve the MLA *latent* KV record from `fp8_ds_mla` (656 B/tok) to a 4-bit `nvfp4_ds_mla` (~432 B/tok) to gain ~1.5× context length, **without** unacceptable long-context-retrieval loss. Implement, smoke-test accuracy vs the `fp8_ds_mla` baseline, and **iterate until you reach a defensible deployability verdict** (deployable, or not-deployable-with-evidence). Do NOT promote without passing the §5 retrieval gate.

## 2. The design (DO NOT deviate without cause)
`nvfp4_ds_mla`, 432 B/token contiguous record (mirror the GLM footer-free convention):

| Sub-vector | Format | Bytes |
|---|---|---|
| NoPE (512 dims, kv_lora_rank) | **NVFP4 (E2M1)**, 2 nibbles/byte | 256 |
| NoPE block scales | **E4M3 FP8**, 1 per group-16 | 32 |
| pad → 16B-align RoPE | — | 16 |
| RoPE (64 dims) | **bf16, UNQUANTIZED** | 128 |
| **Total** | | **432** (0.66× → ~1.52× context) |

**Non-negotiable:** RoPE stays bf16. Unanimous precedent (DeepSeek V3.2/V4, vLLM, NVIDIA, KVQuant/RotateKV) — RoPE channels are outlier-heavy/position-sensitive and quantizing them collapses retrieval. Use **NVFP4 (E4M3 group-16), NOT MXFP4/E8M0** — NVIDIA reports ~5 pt KV-accuracy edge from the smaller block + E4M3 scale. (Aggressive 324 B/tok variant that also FP4s RoPE = SGLang's choice = gated stage-2 ONLY, behind a passing eval.)

## 3. Prior art (learn from, don't copy blindly)
- **SGLang PR #10078** (merged 2025-11-02, https://github.com/sgl-project/sglang/pull/10078) — the only production 4-bit MLA latent. MXFP4 (E8M0), **quantizes RoPE too**, dequant→bf16, **no long-context eval**. Accuracy: gsm8k −0.3pp, gpqa_diamond −4.3pp, aime25 −10.7pp; +17.8% tput. Reviewer flagged untested long-ctx. **Our design is deliberately more conservative** (RoPE bf16, NVFP4 scales). Diff at `/tmp/pr10078.diff` if present.
- **TRT-LLM #8142** — NVFP4-on-MLA storage exists but disables block-reuse + missing kernels. Unfinished.
- **minimax_m3 nvfp4 indexer** (`vllm/models/minimax_m3/common/indexer.py:552`, `vllm/config/attention.py:13,55`) — declared enum + `NotImplementedError` stub. Use as the **dispatch/naming pattern**, not an impl.
- **No one has done NVFP4 NoPE-only + bf16 RoPE on the MLA latent.** You are characterizing it first → the §5 retrieval eval is mandatory, not optional.

## 4. Implementation — exact change map (all file:line in the audited image)
**Layer 1 — config/enum (~0.5 day, Python):**
- `vllm/config/cache.py:19` — add `"nvfp4_ds_mla"` to `CacheDType` Literal; ensure `is_quantized_kv_cache()` → True for it.
- `vllm/model_executor/layers/attention/mla_attention.py:416-430` — **THE chokepoint**: it force-coerces any quantized dtype → `fp8_ds_mla` for `{"FLASHMLA_SPARSE","B12X_MLA_SPARSE"}`. Add a branch keeping `nvfp4_ds_mla` for `B12X_MLA_SPARSE`. Also `:721` (`fp8_attention and kv_cache_dtype != "fp8_ds_mla"`) — extend so the nvfp4 record isn't bit-cast to fp8.

**Layer 2 — backend dtype + shape (~0.5 day, Python):**
- `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`: add `"nvfp4_ds_mla"` to `supported_kv_cache_dtypes` (~:190-196); `get_kv_cache_shape` (:260-273) → return 432-byte record; thread `self.kv_cache_dtype` to the decode plan (`B12xMLASparseImpl.__init__` :484+). Head-size 576 logical contract unchanged.
- Leave `flashmla_sparse.py` untouched (not needed for sm_120).

**Layer 3 — store/quant CUDA kernel (~1–1.5 day, the ONLY rebuild):**
- Clone `concat_and_cache_mla_kernel` in `/opt/vllm/csrc/libtorch_stable/cache_kernels.cu:398-590` → `concat_and_cache_nvfp4_mla_kernel`. Splice the NVFP4 quant device fn from `/opt/vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:49` + `quantization/fp4/nvfp4_utils.cuh` (E2M1 nibble pack + E4M3 group-16 scale). Layout: NoPE→[0,256), scales→[256,288), pad, RoPE bf16→[304,432).
- Register in `csrc/libtorch_stable/torch_bindings.cpp`; add Python wrapper beside `concat_and_cache_mla` (`vllm/_custom_ops.py:2690,2703`); dispatch on dtype.
- Validate bit-exact vs DeepGEMM reference `per_token_cast_to_fp4`/`cast_back_from_fp4` (`deep_gemm/utils/math.py:85,131`).

**Layer 4 — DECODE kernel (~1.5–2.5 day, PURE CuteDSL Python — NO .so, JIT, no image rebuild to iterate):**
- The b12x sparse-MLA decode is 100% Python and already parameterized by a `scale_format` constexpr (`b12x/attention/mla/traits.py:47`, `io.py:97`, `decode_math.py:311`): `UE8M0_BYTE=0` (DSV4), `ARBITRARY_FP32=1` (GLM). **Add `NVFP4_E4M3=2`**, mirroring the GLM branch:
  - `b12x/attention/mla/io.py:57-71` + `io_mg.py:29-34` — NVFP4 byte constants (stride 432, NoPE+sf 288, RoPE 128); branch at `io.py:142-151`.
  - `b12x/attention/mla/decode_math.py:311-370` — **the heart**. Swap the GLM dequant `dequant_kv_e4m3_pair_to_bf16x2` (`cute/fp4.py:3440`) → `packed_dequant_e2m1x4_to_bfloat2x2` (`cute/fp4.py:2150`) + `packed_dequant_e4m3x4_to_bfloat2x2` (`:2228`) — **both primitives already exist, unused by this path.** Update K-step count: NoPE contraction is `512/64=8` fp4 steps (FP4_MMA_K=64, `cute/fp4.py:101`) vs `512/32=16` fp8 (FP8_MMA_K=32, `:10`).
  - `traits.py:47-85` (add enum + validator), `smem.py` (NoPE smem stride 288).
- **STRATEGY A (do this first):** dequant NVFP4→bf16 in-register, feed the EXISTING bf16-QK MMA (exactly what the fp8 path does today). Smallest, safest change; full memory win; proven compute path. **STRATEGY B** (native FP4 QK MMA, FP4_MMA_K=64) = stage-2 perf, only after A passes accuracy.
- DeepGEMM `sm120_fp8_fp4_gemm_1d1d` is a GEMM, **not** reusable for the sparse attention decode — do NOT try; reuse only its host quant reference for tests.

## 5. Deployability gate (smoke-tests — promotion criteria)
Run in order; each gates the next. Accuracy is vs the `fp8_ds_mla` baseline. Bench: `/home/brandonmusic/llm-inference-bench/llm_decode_bench.py`, temp 1.0/top_p 0.95 (temp 0 → runaway on this stack).
1. **Unit numerics (no GPU/model):** round-trip random `[T,576]` latent through the store kernel + DeepGEMM `cast_back_from_fp4`; assert per-group MSE < 1.0 + cosine-sim. Bit-exact CUDA-vs-torch reference.
2. **Decode equivalence (GPU):** single-seq greedy, `nvfp4_ds_mla` vs `fp8_ds_mla`, same prompt; report token divergence point + per-step logit cosine-sim.
3. **Long-context RETRIEVAL (the real gate — NOT perplexity):** KLC probes **Estonia 10-hop, LAVD ledger, hotel-lights** (baselines: Estonia 10/10, LAVD 5/5) + synthetic RULER multi-needle @ 32K/64K/128K/256K. **PROMOTE iff: no Estonia/LAVD regression AND ≤1–2 pt RULER drop @ 64–128K.**
4. **Capacity + speed:** confirm KV tokens scale ~1.5× at fixed util (656→432) and decode t/s within noise of fp8_ds_mla (Strategy A dequants to the same bf16 MMA).
5. **Iterate if gate #3 fails (in order, each has precedent):** (a) sink+recent fp8_ds_mla residual tokens (KVSink — first-64 + recent-N high-precision, per-block dtype tag); (b) FP8 NoPE for first-512 tokens; (c) Hadamard/smooth rotation on the latent. Re-run gate #3 after each. **If still failing after (a)+(b): conclude NOT deployable, document the failure curve (which context length breaks), recommend FP8 floor.**

## 6. Environment / coordination
- Layers 1–2 + 4 iterate WITHOUT an image rebuild (Python/CuteDSL JIT) → fast loop. Only Layer 3 (CUDA store) needs one rebuild.
- On-GPU tests (gates #2–4) need the sm_120 cards, shared with prod. Request a GPU window; do NOT run the model while the prod/eval battery holds the GPUs.
- Bisect-friendly: build Layer 3 + 4 behind the `nvfp4_ds_mla` dtype flag so the baseline `fp8_ds_mla` is always one env var away.

## 7. Effort
~3–6 focused days for Strategy-A decode + the accuracy gate. Layer 1–2: 0.5d. Layer 3: 1–1.5d. Layer 4(A): 1.5–2.5d. Residual mitigation: +1d if needed. Strategy B: separate follow-up.

*(Full research provenance + URLs in the agent transcript; key sources: SGLang #10078, NVIDIA NVFP4-KV RULER-64K=94.6% blog, Vultr "FP8 is the floor" kernel, KVQuant/RotateKV/KVSink papers.)*
