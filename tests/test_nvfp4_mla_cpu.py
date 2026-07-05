#!/usr/bin/env python3
"""CPU-only numerics for the 432-byte nvfp4_ds_mla latent KV record."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "site-packages"))

from deep_gemm.utils.math import cast_back_from_fp4, per_token_cast_to_fp4  # noqa: E402


NOPE_DIM = 512
ROPE_DIM = 64
FP4_GROUP = 16
RECORD_BYTES = 432
NOPE_BYTES = 256
SCALE_BYTES = 32
ROPE_OFFSET = 304
ROPE_BYTES = 128
FP4_STEPS = NOPE_DIM // 64
FP4_STEP_DIM = 64


def decode_dequant_nvfp4_nope_steps(record: torch.Tensor) -> torch.Tensor:
    """CPU mirror of the decode-local NVFP4 NoPE dequant path.

    Returns [T, 8, 64] float32 values produced from the 256 packed E2M1 data
    bytes and 32 E4M3 group-16 scales. The dequantized values are rounded through
    BF16 to match the BF16 QK/PV MMA operands used by the decode kernel branch.
    """
    if record.ndim != 2 or record.shape[1] != RECORD_BYTES:
        raise ValueError(f"expected [T,432] record, got {tuple(record.shape)}")

    packed = record[:, :NOPE_BYTES].to(torch.uint8)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    codes = torch.empty((record.shape[0], NOPE_DIM), dtype=torch.uint8)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi

    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
    )
    value_idx = (codes & 0x07).to(torch.long)
    sign = (codes & 0x08) != 0
    values = fp4_values[value_idx]
    values = torch.where(sign & (value_idx != 0), -values, values).to(torch.bfloat16)

    scales = (
        record[:, NOPE_BYTES : NOPE_BYTES + SCALE_BYTES]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
        .to(torch.bfloat16)
    )
    nope = (
        values.reshape(record.shape[0], SCALE_BYTES, FP4_GROUP)
        * scales.unsqueeze(-1)
    ).to(torch.bfloat16)
    return nope.reshape(record.shape[0], FP4_STEPS, FP4_STEP_DIM).to(torch.float32)


def decode_dequant_nvfp4_record(record: torch.Tensor) -> torch.Tensor:
    nope = decode_dequant_nvfp4_nope_steps(record).reshape(record.shape[0], NOPE_DIM)
    rope = (
        record[:, ROPE_OFFSET : ROPE_OFFSET + ROPE_BYTES]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(record.shape[0], ROPE_DIM)
        .to(torch.float32)
    )
    return torch.cat([nope, rope], dim=1)


def pack_nvfp4_ds_mla(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if latent.ndim != 2 or latent.shape[1] != NOPE_DIM + ROPE_DIM:
        raise ValueError(f"expected [T,576] latent, got {tuple(latent.shape)}")
    latent_bf16 = latent.to(torch.bfloat16).contiguous()
    nope = latent_bf16[:, :NOPE_DIM].contiguous()
    rope = latent_bf16[:, NOPE_DIM:].contiguous()

    packed, sf_ref = per_token_cast_to_fp4(nope, use_ue8m0=False, gran_k=FP4_GROUP)
    sf_e4m3 = sf_ref.to(torch.float8_e4m3fn)

    record = torch.zeros((latent.shape[0], RECORD_BYTES), dtype=torch.uint8)
    record[:, :NOPE_BYTES] = packed.view(torch.uint8)
    record[:, NOPE_BYTES : NOPE_BYTES + SCALE_BYTES] = sf_e4m3.view(torch.uint8)
    record[:, ROPE_OFFSET : ROPE_OFFSET + ROPE_BYTES] = rope.view(torch.uint8).reshape(
        latent.shape[0], ROPE_BYTES
    )
    return record, packed.contiguous(), sf_ref.contiguous()


def unpack_nvfp4_ds_mla(record: torch.Tensor) -> torch.Tensor:
    if record.ndim != 2 or record.shape[1] != RECORD_BYTES:
        raise ValueError(f"expected [T,432] record, got {tuple(record.shape)}")
    packed = record[:, :NOPE_BYTES].contiguous().view(torch.int8)
    sf = (
        record[:, NOPE_BYTES : NOPE_BYTES + SCALE_BYTES]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    nope = cast_back_from_fp4(packed, sf, gran_k=FP4_GROUP)
    rope = (
        record[:, ROPE_OFFSET : ROPE_OFFSET + ROPE_BYTES]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(record.shape[0], ROPE_DIM)
        .to(torch.float32)
    )
    return torch.cat([nope, rope], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=257)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mse-threshold", type=float, default=1.0)
    parser.add_argument("--cos-threshold", type=float, default=0.99)
    parser.add_argument("--step-cos-threshold", type=float, default=0.985)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    base = torch.randn(args.tokens, NOPE_DIM + ROPE_DIM, dtype=torch.float32)
    # Mild per-group scale variation exercises small and large groups without
    # making the FP4 MSE gate vacuous.
    group_scale = torch.linspace(0.25, 3.0, NOPE_DIM // FP4_GROUP).repeat_interleave(
        FP4_GROUP
    )
    base[:, :NOPE_DIM] *= group_scale

    record, packed_ref, sf_ref = pack_nvfp4_ds_mla(base)
    recon = unpack_nvfp4_ds_mla(record)
    decode_steps = decode_dequant_nvfp4_nope_steps(record)
    decode_recon = decode_dequant_nvfp4_record(record)

    nope_target = base[:, :NOPE_DIM].to(torch.bfloat16).to(torch.float32)
    rope_target = base[:, NOPE_DIM:].to(torch.bfloat16).to(torch.float32)

    packed_again = record[:, :NOPE_BYTES].contiguous().view(torch.int8)
    expected_scale_bytes = sf_ref.to(torch.float8_e4m3fn).view(torch.uint8)
    actual_scale_bytes = record[:, NOPE_BYTES : NOPE_BYTES + SCALE_BYTES]
    assert torch.equal(packed_again, packed_ref), "packed FP4 bytes differ from DeepGEMM reference"
    assert torch.equal(actual_scale_bytes, expected_scale_bytes), (
        "E4M3 scale bytes differ from DeepGEMM-derived reference scales"
    )

    group_mse = (
        (recon[:, :NOPE_DIM] - nope_target)
        .reshape(args.tokens, NOPE_DIM // FP4_GROUP, FP4_GROUP)
        .pow(2)
        .mean(dim=2)
    )
    max_group_mse = float(group_mse.max().item())
    mean_group_mse = float(group_mse.mean().item())
    cos_nope = F.cosine_similarity(recon[:, :NOPE_DIM], nope_target, dim=1)
    cos_full = F.cosine_similarity(
        recon, torch.cat([nope_target, rope_target], dim=1), dim=1
    )
    step_target = nope_target.reshape(args.tokens, FP4_STEPS, FP4_STEP_DIM)
    step_cos = F.cosine_similarity(decode_steps, step_target, dim=2)
    decode_cos_full = F.cosine_similarity(
        decode_recon, torch.cat([nope_target, rope_target], dim=1), dim=1
    )
    min_nope_cos = float(cos_nope.min().item())
    min_full_cos = float(cos_full.min().item())
    min_step_cos = float(step_cos.min().item())
    min_decode_full_cos = float(decode_cos_full.min().item())
    rope_exact = torch.equal(recon[:, NOPE_DIM:].to(torch.bfloat16), rope_target.to(torch.bfloat16))
    decode_rope_exact = torch.equal(
        decode_recon[:, NOPE_DIM:].to(torch.bfloat16), rope_target.to(torch.bfloat16)
    )

    assert max_group_mse < args.mse_threshold, (
        f"max per-group MSE {max_group_mse:.6f} >= {args.mse_threshold}"
    )
    assert min_nope_cos > args.cos_threshold, (
        f"min NoPE cosine {min_nope_cos:.6f} <= {args.cos_threshold}"
    )
    assert min_full_cos > args.cos_threshold, (
        f"min full latent cosine {min_full_cos:.6f} <= {args.cos_threshold}"
    )
    assert min_step_cos > args.step_cos_threshold, (
        f"min per-step decode cosine {min_step_cos:.6f} <= {args.step_cos_threshold}"
    )
    assert min_decode_full_cos > args.cos_threshold, (
        f"min decode full latent cosine {min_decode_full_cos:.6f} <= {args.cos_threshold}"
    )
    assert rope_exact, "RoPE BF16 bytes were not preserved exactly"
    assert decode_rope_exact, "Decode-dequant RoPE BF16 bytes were not preserved exactly"

    print(
        "PASS nvfp4_ds_mla CPU numerics: "
        f"tokens={args.tokens} record_bytes={RECORD_BYTES} "
        f"max_group_mse={max_group_mse:.6f} mean_group_mse={mean_group_mse:.6f} "
        f"min_nope_cos={min_nope_cos:.6f} min_full_cos={min_full_cos:.6f} "
        f"min_step_cos={min_step_cos:.6f} "
        f"min_decode_full_cos={min_decode_full_cos:.6f} "
        f"rope_bf16_exact={rope_exact} decode_rope_bf16_exact={decode_rope_exact}"
    )


if __name__ == "__main__":
    main()
