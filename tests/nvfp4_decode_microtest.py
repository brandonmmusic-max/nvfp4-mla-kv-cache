#!/usr/bin/env python3
"""GPU micro-test for fp8_ds_mla vs nvfp4_ds_mla one-step decode.

The test uses the real vLLM cache write ops and the real B12X SM120 sparse MLA
decode op.  It compares both cache formats against a BF16 PyTorch reference over
the same synthetic Q and latent KV.  The top-1 case isolates gather + PV; the
64-token focus cases exercise QK-dependent attention.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from b12x.attention.mla.api import sparse_mla_decode_forward
from b12x.attention.mla.traits import ScaleFormat
from b12x.integration.sparse_mla_scratch import (
    B12XSparseMLAScratchCaps,
    plan_sparse_mla_scratch,
)


HEADS = 16
D_NOPE = 512
D_ROPE = 64
D_Q = D_NOPE + D_ROPE
PAGE_SIZE = 64


@dataclass(frozen=True)
class DecodeResult:
    name: str
    fp8_min_cos: float
    nvfp4_min_cos: float
    fp8_mean_cos: float
    nvfp4_mean_cos: float
    nvfp4_vs_fp8_min_cos: float
    nvfp4_delta: float
    fp8_delta: float


def _make_plan(fmt: str, *, topk: int, device: torch.device):
    scale_format = (
        ScaleFormat.NVFP4_E4M3
        if fmt == "nvfp4_ds_mla"
        else ScaleFormat.ARBITRARY_FP32
    )
    caps = B12XSparseMLAScratchCaps(
        device=device,
        num_q_heads=HEADS,
        max_q_rows=1,
        max_width=topk,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        head_dim=D_Q,
        v_head_dim=D_NOPE,
        mode="decode",
        max_batch=1,
        max_chunks_per_row=max(1, math.ceil(topk / PAGE_SIZE)),
        page_size=PAGE_SIZE,
        kv_cache_dtype=fmt,
        scale_format=scale_format,
    )
    return plan_sparse_mla_scratch(caps), int(scale_format)


def _make_cache(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    fmt: str,
    *,
    page_size: int = PAGE_SIZE,
) -> torch.Tensor:
    tokens = int(kv_c.shape[0])
    record_bytes = 432 if fmt == "nvfp4_ds_mla" else 656
    blocks = math.ceil(tokens / page_size)
    cache = torch.empty(
        (blocks, page_size, record_bytes),
        dtype=torch.uint8,
        device=kv_c.device,
    )
    cache.zero_()
    slots = torch.arange(tokens, dtype=torch.int64, device=kv_c.device)
    scale = torch.ones((1,), dtype=torch.float32, device=kv_c.device)
    ops.concat_and_cache_mla(kv_c, k_pe, cache, slots, fmt, scale)
    torch.cuda.synchronize()
    return cache


def _decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    fmt: str,
    *,
    sm_scale: float,
) -> torch.Tensor:
    plan, scale_format = _make_plan(fmt, topk=int(indices.shape[1]), device=q.device)
    shape, dtype = plan.shapes_and_dtypes()[0]
    scratch = torch.empty(shape, dtype=dtype, device=q.device)
    lengths = torch.full((1,), int(indices.shape[1]), dtype=torch.int32, device=q.device)
    binding = plan.bind(
        scratch=scratch,
        q=q.contiguous(),
        selected_indices=indices.contiguous(),
        cache_seqlens_int32=lengths,
        nsa_cache_seqlens_int32=lengths,
    )
    out = sparse_mla_decode_forward(
        binding=binding,
        kv_cache=kv_cache,
        sm_scale=sm_scale,
        v_head_dim=D_NOPE,
        forced_num_splits=max(1, math.ceil(int(indices.shape[1]) / PAGE_SIZE)),
        scale_format=scale_format,
    )
    torch.cuda.synchronize()
    return out.float()


def _reference(
    q: torch.Tensor,
    latent: torch.Tensor,
    rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
) -> torch.Tensor:
    selected = indices[0].long()
    k_nope = latent[selected].float()
    k_rope = rope[selected].float()
    q_nope = q[0, :, :D_NOPE].float()
    q_rope = q[0, :, D_NOPE:].float()
    scores = q_nope @ k_nope.T
    scores = scores + q_rope @ k_rope.T
    probs = torch.softmax(scores * sm_scale, dim=-1)
    return (probs @ k_nope).unsqueeze(0)


def _cos_by_head(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(out[0].float(), ref[0].float(), dim=1)


def run_case(
    name: str,
    q: torch.Tensor,
    latent: torch.Tensor,
    rope: torch.Tensor,
    indices: torch.Tensor,
    fp8_cache: torch.Tensor,
    nvfp4_cache: torch.Tensor,
    *,
    sm_scale: float,
    previous_nvfp4: torch.Tensor | None,
    previous_fp8: torch.Tensor | None,
) -> tuple[DecodeResult, torch.Tensor, torch.Tensor]:
    ref = _reference(q, latent, rope, indices, sm_scale=sm_scale)
    fp8 = _decode(q, fp8_cache, indices, "fp8_ds_mla", sm_scale=sm_scale)
    nvfp4 = _decode(q, nvfp4_cache, indices, "nvfp4_ds_mla", sm_scale=sm_scale)

    fp8_cos = _cos_by_head(fp8, ref)
    nvfp4_cos = _cos_by_head(nvfp4, ref)
    cross_cos = F.cosine_similarity(nvfp4[0], fp8[0], dim=1)
    nvfp4_delta = (
        float((nvfp4 - previous_nvfp4).abs().max().item())
        if previous_nvfp4 is not None
        else float("nan")
    )
    fp8_delta = (
        float((fp8 - previous_fp8).abs().max().item())
        if previous_fp8 is not None
        else float("nan")
    )
    result = DecodeResult(
        name=name,
        fp8_min_cos=float(fp8_cos.min().item()),
        nvfp4_min_cos=float(nvfp4_cos.min().item()),
        fp8_mean_cos=float(fp8_cos.mean().item()),
        nvfp4_mean_cos=float(nvfp4_cos.mean().item()),
        nvfp4_vs_fp8_min_cos=float(cross_cos.min().item()),
        nvfp4_delta=nvfp4_delta,
        fp8_delta=fp8_delta,
    )
    return result, nvfp4, fp8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--sm-scale", type=float, default=1.0 / math.sqrt(D_Q))
    parser.add_argument("--min-top1-cos", type=float, default=0.985)
    parser.add_argument("--min-focus-cos", type=float, default=0.92)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    latent = torch.randn(args.tokens, D_NOPE, device=device, dtype=torch.bfloat16)
    rope = torch.randn(args.tokens, D_ROPE, device=device, dtype=torch.bfloat16)
    # Keep values in a range where FP4 quantization is meaningful without making
    # the synthetic attention too flat.
    latent.mul_(0.75)
    rope.mul_(0.25)

    fp8_cache = _make_cache(latent, rope, "fp8_ds_mla")
    nvfp4_cache = _make_cache(latent, rope, "nvfp4_ds_mla")

    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    top1_idx = torch.tensor([[17]], device=device, dtype=torch.int32)
    q_random = torch.randn(1, HEADS, D_Q, device=device, dtype=torch.bfloat16)
    cases.append(("top1_gather_pv", q_random, top1_idx))

    focus_indices = torch.arange(64, device=device, dtype=torch.int32).view(1, 64)
    for target in (5, 42):
        q_focus = torch.zeros(1, HEADS, D_Q, device=device, dtype=torch.bfloat16)
        q_focus[:, :, :D_NOPE] = latent[target].view(1, 1, D_NOPE) * 5.0
        q_focus[:, :, D_NOPE:] = rope[target].view(1, 1, D_ROPE) * 2.0
        q_focus += 0.01 * torch.randn_like(q_focus)
        cases.append((f"focus_token_{target}", q_focus, focus_indices))

    prev_nvfp4 = None
    prev_fp8 = None
    results: list[DecodeResult] = []
    for name, q, idx in cases:
        result, prev_nvfp4, prev_fp8 = run_case(
            name,
            q,
            latent,
            rope,
            idx,
            fp8_cache,
            nvfp4_cache,
            sm_scale=args.sm_scale,
            previous_nvfp4=prev_nvfp4,
            previous_fp8=prev_fp8,
        )
        results.append(result)
        print(
            f"{result.name}: "
            f"fp8_min={result.fp8_min_cos:.6f} fp8_mean={result.fp8_mean_cos:.6f} "
            f"nvfp4_min={result.nvfp4_min_cos:.6f} nvfp4_mean={result.nvfp4_mean_cos:.6f} "
            f"nvfp4_vs_fp8_min={result.nvfp4_vs_fp8_min_cos:.6f} "
            f"delta_nvfp4={result.nvfp4_delta:.6g} delta_fp8={result.fp8_delta:.6g}"
        )

    top1 = results[0]
    focus = results[1:]
    if top1.nvfp4_min_cos < args.min_top1_cos:
        raise SystemExit(
            "FAIL: nvfp4 top-1 gather/PV path diverged "
            f"(min cos {top1.nvfp4_min_cos:.6f} < {args.min_top1_cos})"
        )
    bad_focus = [r for r in focus if r.nvfp4_min_cos < args.min_focus_cos]
    if bad_focus:
        names = ", ".join(f"{r.name}:{r.nvfp4_min_cos:.6f}" for r in bad_focus)
        raise SystemExit(f"FAIL: nvfp4 QK-dependent focus decode diverged: {names}")
    if focus[-1].nvfp4_delta < args.min_delta:
        raise SystemExit(
            "FAIL: nvfp4 output is input-independent across focus cases "
            f"(max delta {focus[-1].nvfp4_delta:.6g} < {args.min_delta})"
        )
    print("PASS: nvfp4 one-step decode tracks fp8/reference and changes with Q")


if __name__ == "__main__":
    main()
