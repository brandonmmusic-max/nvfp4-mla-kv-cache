// SPDX-License-Identifier: Apache-2.0
// Standalone registration for the vLLM NVFP4 MLA KV-cache writer.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

#define NVFP4_ENABLE_ELTS16 1
#include "libtorch_stable/quantization/fp4/nvfp4_utils.cuh"

namespace vllm {

template <typename scalar_t>
__global__ void concat_and_cache_nvfp4_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    uint8_t* __restrict__ kv_cache,     // [num_blocks, block_size, 432]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size                       //
) {
  using CudaType = typename CUDATypeConverter<scalar_t>::Type;
  using PVec = PackedVec<CudaType, CVT_FP4_PACK16>;

  static constexpr int kNopeBytes = 256;
  static constexpr int kScaleBytes = 32;
  static constexpr int kPadBytes = 16;
  static constexpr int kRopeOffset = kNopeBytes + kScaleBytes + kPadBytes;
  static constexpr int kFp4GroupSize = CVT_FP4_SF_VEC_SIZE;
  static constexpr int kEltsPerThread = CVT_FP4_ELTS_PER_THREAD;
  static constexpr int kThreadsPerScale = kFp4GroupSize / kEltsPerThread;

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  uint8_t* __restrict__ token_dst =
      kv_cache + block_idx * block_stride + block_offset * entry_stride;

  const CudaType* __restrict__ token_src =
      reinterpret_cast<const CudaType*>(kv_c) + token_idx * kv_c_stride;

  const int group_count = kv_lora_rank / kFp4GroupSize;
  const int thread_group_count = blockDim.x / kThreadsPerScale;
  const int thread_group = threadIdx.x / kThreadsPerScale;
  const int thread_group_lane = threadIdx.x % kThreadsPerScale;

  for (int group = thread_group; group < group_count;
       group += thread_group_count) {
    PVec in_vec;
    const CudaType* __restrict__ src =
        token_src + group * kFp4GroupSize + thread_group_lane * kEltsPerThread;

#pragma unroll
    for (int i = 0; i < kEltsPerThread / 2; ++i) {
      in_vec.elts[i] =
          reinterpret_cast<const typename PackedTypeConverter<CudaType>::Type*>(
              src)[i];
    }

    uint8_t scale_byte;
    uint8_t* scale_out = (thread_group_lane == 0) ? &scale_byte : nullptr;
    fp4_packed_t packed =
        cvt_warp_fp16_to_fp4<CudaType, kThreadsPerScale>(in_vec, 1.0f,
                                                         scale_out);

#if CVT_FP4_PACK16
    uint8_t* data_dst = token_dst + group * 8;
    reinterpret_cast<uint64_t*>(data_dst)[0] =
        (uint64_t(packed.hi) << 32) | uint64_t(packed.lo);
#else
    uint8_t* data_dst = token_dst + group * 8 + thread_group_lane * 4;
    reinterpret_cast<uint32_t*>(data_dst)[0] = packed;
#endif

    if (scale_out != nullptr) {
      token_dst[kNopeBytes + group] = scale_byte;
    }
  }

  for (int i = threadIdx.x; i < kPadBytes; i += blockDim.x) {
    token_dst[kNopeBytes + kScaleBytes + i] = 0;
  }

  scalar_t* __restrict__ rope_dst =
      reinterpret_cast<scalar_t*>(token_dst + kRopeOffset);
  const scalar_t* __restrict__ rope_src = k_pe + token_idx * k_pe_stride;
  for (int i = threadIdx.x; i < pe_dim; i += blockDim.x) {
    rope_dst[i] = rope_src[i];
  }
}

}  // namespace vllm

namespace {

void check_cuda_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void concat_and_cache_nvfp4_mla(torch::Tensor kv_c, torch::Tensor k_pe,
                                torch::Tensor kv_cache,
                                torch::Tensor slot_mapping,
                                torch::Tensor scale) {
  (void)scale;
  check_cuda_tensor(kv_c, "kv_c");
  check_cuda_tensor(k_pe, "k_pe");
  check_cuda_tensor(kv_cache, "kv_cache");
  check_cuda_tensor(slot_mapping, "slot_mapping");

  const int64_t num_tokens = slot_mapping.size(0);
  const int64_t kv_lora_rank = kv_c.size(1);
  const int64_t pe_dim = k_pe.size(1);

  TORCH_CHECK(kv_lora_rank == 512,
              "kv_lora_rank must be 512 for nvfp4_ds_mla");
  TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for nvfp4_ds_mla");
  TORCH_CHECK(kv_cache.element_size() == 1,
              "kv_cache must be uint8 for nvfp4_ds_mla");
  TORCH_CHECK(kv_cache.size(2) == 432,
              "kv_cache.size(2) must be 432 bytes for nvfp4_ds_mla");
  TORCH_CHECK(kv_c.element_size() == 2,
              "kv_c.element_size() must be 2 for nvfp4_ds_mla");
  TORCH_CHECK(k_pe.element_size() == 2,
              "k_pe.element_size() must be 2 for nvfp4_ds_mla");
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long,
              "slot_mapping must be int64");

  const int64_t block_size = kv_cache.size(1);
  const int64_t kv_c_stride = kv_c.stride(0);
  const int64_t k_pe_stride = k_pe.stride(0);
  const int64_t block_stride = kv_cache.stride(0);
  const int64_t entry_stride = kv_cache.stride(1);

  c10::cuda::CUDAGuard device_guard(kv_c.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(num_tokens);
  dim3 block(128);

  switch (kv_c.scalar_type()) {
    case at::ScalarType::Half: {
      using scalar_t = c10::Half;
      vllm::concat_and_cache_nvfp4_mla_kernel<scalar_t>
          <<<grid, block, 0, stream>>>(
              kv_c.const_data_ptr<scalar_t>(), k_pe.const_data_ptr<scalar_t>(),
              kv_cache.mutable_data_ptr<uint8_t>(),
              slot_mapping.const_data_ptr<int64_t>(),
              static_cast<int>(block_stride), static_cast<int>(entry_stride),
              static_cast<int>(kv_c_stride), static_cast<int>(k_pe_stride),
              static_cast<int>(kv_lora_rank), static_cast<int>(pe_dim),
              static_cast<int>(block_size));
      break;
    }
    case at::ScalarType::BFloat16: {
      using scalar_t = c10::BFloat16;
      vllm::concat_and_cache_nvfp4_mla_kernel<scalar_t>
          <<<grid, block, 0, stream>>>(
              kv_c.const_data_ptr<scalar_t>(), k_pe.const_data_ptr<scalar_t>(),
              kv_cache.mutable_data_ptr<uint8_t>(),
              slot_mapping.const_data_ptr<int64_t>(),
              static_cast<int>(block_stride), static_cast<int>(entry_stride),
              static_cast<int>(kv_c_stride), static_cast<int>(k_pe_stride),
              static_cast<int>(kv_lora_rank), static_cast<int>(pe_dim),
              static_cast<int>(block_size));
      break;
    }
    default:
      TORCH_CHECK(false,
                  "concat_and_cache_nvfp4_mla supports only half and bfloat16");
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_cache_ops, m) {
  m.def("concat_and_cache_nvfp4_mla(Tensor kv_c, Tensor k_pe,"
        "                           Tensor! kv_cache,"
        "                           Tensor slot_mapping,"
        "                           Tensor scale) -> ()");
}

TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA, m) {
  m.impl("concat_and_cache_nvfp4_mla", &concat_and_cache_nvfp4_mla);
}
