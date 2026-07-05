#!/usr/bin/env python3
"""GPU micro-test for fp8_ds_mla vs nvfp4_ds_mla prefill/extend attention.

The test uses the real vLLM cache writer and the real B12X SM120 single-pass
prefill path.  It compares both cache formats against a BF16 PyTorch reference
over identical synthetic Q and latent KV, with multiple Q rows to catch
input-independent corruption.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
from b12x.attention.mla.api import sparse_mla_extend_forward
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
class PrefillResult:
    name: str
    fp8_min_cos: float
    fp8_mean_cos: float
    nvfp4_min_cos: float
    nvfp4_mean_cos: float
    nvfp4_vs_fp8_min_cos: float
    nvfp4_row_delta: float
    fp8_row_delta: float


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    total = int(tensor.numel())
    raise SystemExit(
        f"FAIL: {name} contains non-finite values "
        f"(nan={nan_count} inf={inf_count} total={total})"
    )


def _make_plan(fmt: str, *, q_rows: int, topk: int, device: torch.device):
    scale_format = (
        ScaleFormat.NVFP4_E4M3
        if fmt == "nvfp4_ds_mla"
        else ScaleFormat.ARBITRARY_FP32
    )
    caps = B12XSparseMLAScratchCaps(
        device=device,
        num_q_heads=HEADS,
        max_q_rows=q_rows,
        max_width=topk,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        head_dim=D_Q,
        v_head_dim=D_NOPE,
        mode="extend",
        max_batch=q_rows,
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


def _prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    fmt: str,
    *,
    sm_scale: float,
) -> torch.Tensor:
    q_rows = int(q.shape[0])
    topk = int(indices.shape[1])
    plan, scale_format = _make_plan(fmt, q_rows=q_rows, topk=topk, device=q.device)
    shape, dtype = plan.shapes_and_dtypes()[0]
    scratch = torch.empty(shape, dtype=dtype, device=q.device)
    lengths = torch.full((q_rows,), topk, dtype=torch.int32, device=q.device)
    binding = plan.bind(
        scratch=scratch,
        q=q.contiguous(),
        selected_indices=indices.contiguous(),
        cache_seqlens_int32=lengths,
        nsa_cache_seqlens_int32=lengths,
    )
    out = sparse_mla_extend_forward(
        binding=binding,
        kv_cache=kv_cache,
        sm_scale=sm_scale,
        v_head_dim=D_NOPE,
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
    outs = []
    for row in range(int(q.shape[0])):
        selected = indices[row].long()
        k_nope = latent[selected].float()
        k_rope = rope[selected].float()
        q_nope = q[row, :, :D_NOPE].float()
        q_rope = q[row, :, D_NOPE:].float()
        scores = q_nope @ k_nope.T
        scores = scores + q_rope @ k_rope.T
        probs = torch.softmax(scores * sm_scale, dim=-1)
        outs.append(probs @ k_nope)
    return torch.stack(outs, dim=0)


def _cos_by_row_head(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(out.float(), ref.float(), dim=2)


def _run_case(
    name: str,
    q: torch.Tensor,
    latent: torch.Tensor,
    rope: torch.Tensor,
    indices: torch.Tensor,
    fp8_cache: torch.Tensor,
    nvfp4_cache: torch.Tensor,
    *,
    sm_scale: float,
) -> PrefillResult:
    ref = _reference(q, latent, rope, indices, sm_scale=sm_scale)
    fp8 = _prefill(q, fp8_cache, indices, "fp8_ds_mla", sm_scale=sm_scale)
    nvfp4 = _prefill(q, nvfp4_cache, indices, "nvfp4_ds_mla", sm_scale=sm_scale)

    _require_finite("reference output", ref)
    _require_finite("fp8 prefill output", fp8)
    _require_finite("nvfp4 prefill output", nvfp4)

    fp8_cos = _cos_by_row_head(fp8, ref)
    nvfp4_cos = _cos_by_row_head(nvfp4, ref)
    cross_cos = _cos_by_row_head(nvfp4, fp8)
    _require_finite("fp8 cosine", fp8_cos)
    _require_finite("nvfp4 cosine", nvfp4_cos)
    _require_finite("nvfp4-vs-fp8 cosine", cross_cos)
    row_delta_nvfp4 = float((nvfp4[1:] - nvfp4[:-1]).abs().max().item())
    row_delta_fp8 = float((fp8[1:] - fp8[:-1]).abs().max().item())
    return PrefillResult(
        name=name,
        fp8_min_cos=float(fp8_cos.min().item()),
        fp8_mean_cos=float(fp8_cos.mean().item()),
        nvfp4_min_cos=float(nvfp4_cos.min().item()),
        nvfp4_mean_cos=float(nvfp4_cos.mean().item()),
        nvfp4_vs_fp8_min_cos=float(cross_cos.min().item()),
        nvfp4_row_delta=row_delta_nvfp4,
        fp8_row_delta=row_delta_fp8,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--q-rows", type=int, default=4)
    parser.add_argument("--sm-scale", type=float, default=1.0 / math.sqrt(D_Q))
    parser.add_argument("--min-nvfp4-cos", type=float, default=0.90)
    parser.add_argument("--min-cross-cos", type=float, default=0.90)
    parser.add_argument("--min-row-delta", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    topk = int(args.tokens)
    q_rows = int(args.q_rows)
    if topk < PAGE_SIZE or topk % PAGE_SIZE != 0:
        raise SystemExit("--tokens must be a positive multiple of 64")
    if q_rows < 2:
        raise SystemExit("--q-rows must be >= 2")

    latent = torch.randn(topk, D_NOPE, device=device, dtype=torch.bfloat16)
    rope = torch.randn(topk, D_ROPE, device=device, dtype=torch.bfloat16)
    latent.mul_(0.70)
    rope.mul_(0.20)

    fp8_cache = _make_cache(latent, rope, "fp8_ds_mla")
    nvfp4_cache = _make_cache(latent, rope, "nvfp4_ds_mla")

    indices = torch.arange(topk, device=device, dtype=torch.int32).repeat(q_rows, 1)
    targets = torch.linspace(7, topk - 17, steps=q_rows, device=device).to(torch.long)
    q = torch.empty(q_rows, HEADS, D_Q, device=device, dtype=torch.bfloat16)
    for row, target in enumerate(targets.tolist()):
        q[row, :, :D_NOPE] = latent[target].view(1, D_NOPE) * 6.0
        q[row, :, D_NOPE:] = rope[target].view(1, D_ROPE) * 3.0
    q.add_(0.01 * torch.randn_like(q))

    result = _run_case(
        "prefill_topk512_multirow",
        q,
        latent,
        rope,
        indices,
        fp8_cache,
        nvfp4_cache,
        sm_scale=float(args.sm_scale),
    )
    print(
        f"{result.name}: "
        f"fp8_min={result.fp8_min_cos:.6f} fp8_mean={result.fp8_mean_cos:.6f} "
        f"nvfp4_min={result.nvfp4_min_cos:.6f} "
        f"nvfp4_mean={result.nvfp4_mean_cos:.6f} "
        f"nvfp4_vs_fp8_min={result.nvfp4_vs_fp8_min_cos:.6f} "
        f"delta_nvfp4={result.nvfp4_row_delta:.6g} "
        f"delta_fp8={result.fp8_row_delta:.6g}"
    )

    if result.nvfp4_min_cos < float(args.min_nvfp4_cos):
        raise SystemExit(
            "FAIL: nvfp4 prefill diverged from BF16 reference "
            f"(min cos {result.nvfp4_min_cos:.6f} < {float(args.min_nvfp4_cos):.6f})"
        )
    if result.nvfp4_vs_fp8_min_cos < float(args.min_cross_cos):
        raise SystemExit(
            "FAIL: nvfp4 prefill diverged from fp8 prefill "
            f"(min cos {result.nvfp4_vs_fp8_min_cos:.6f} < {float(args.min_cross_cos):.6f})"
        )
    if result.nvfp4_row_delta < float(args.min_row_delta):
        raise SystemExit(
            "FAIL: nvfp4 prefill output is input-independent across rows "
            f"(max delta {result.nvfp4_row_delta:.6g} < {float(args.min_row_delta):.6g})"
        )
    print("PASS: nvfp4 prefill tracks fp8/reference and changes with Q")


if __name__ == "__main__":
    main()
