/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <lib.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

using executorch::aten::RuntimeContext;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::ArrayRef;
using executorch::runtime::KernelRuntimeContext;
using torch::executor::Error;
using torch::executor::optional;

namespace impl {
namespace vision {
namespace native {

// Forward declaration of hardware-optimized mean function
extern "C" void simd_mean_pool_2x2_to_1x1_float32(
    float32_t* restrict output,
    const float32_t* restrict input,
    int N);

Tensor& mean_out(
    RuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      torch::executor::check_mean_dim_args(in, dim_list, keepdim, dtype, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      torch::executor::resize_reduction_out(in, dim_list, keepdim, out) ==
          Error::Ok,
      InvalidArgument,
      out);

  constexpr auto name = "mean.out";

  // Check if we can use hardware-optimized path
  // Requires: float32, specific reduction pattern (2x2 spatial to 1x1)
  bool optimized = false;
  
  if (in.scalar_type() == ScalarType::Float && 
      out.scalar_type() == ScalarType::Float &&
      dim_list.has_value()) {
    
    auto dims = dim_list.value();
    int num_inp_dims = in.dim();
    
    // Check for 4D tensor with reduction on last 2 dimensions (H, W)
    // Input: [N, C, H, W], reduce [H, W] -> [N, C, 1, 1]
    if (num_inp_dims == 4 && dims.size() == 2) {
      // Normalize negative dimensions
      int64_t dim0 = dims[0] < 0 ? dims[0] + num_inp_dims : dims[0];
      int64_t dim1 = dims[1] < 0 ? dims[1] + num_inp_dims : dims[1];
      
      // Check if reducing dimensions 2 and 3 (H and W in NCHW format)
      if ((dim0 == 2 && dim1 == 3) || (dim0 == 3 && dim1 == 2)) {
        // Check if spatial dimensions are 2x2
        if (in.size(2) == 2 && in.size(3) == 2) {
          optimized = true;
        }
      }
    }
  }

  if (optimized) {
    TIME_DECL(mean_simd_optimized);
    TIME_START(mean_simd_optimized);
    
    const float* input_data = in.const_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();
    
    int batch_size = in.size(0);
    int channels = in.size(1);
    int total_channels = batch_size * channels;

    // Invalidate input cache so DMA reads fresh data from system memory
    xthal_dcache_region_invalidate((void*)input_data, sizeof(float) * in.numel());
    
    // Check if DRAM buffers are available for DMA
    bool dram0_available = (ptr_dram0 != nullptr) && (IDMA_BUFFER_SIZE_DRAM0 > 0);
    bool dram1_available = (ptr_dram1 != nullptr) && (IDMA_BUFFER_SIZE_DRAM1 > 0);
    
    size_t inp_bytes = total_channels * 4 * FLT32_SIZE;  // 4 floats per channel
    size_t out_bytes = total_channels * FLT32_SIZE;       // 1 float per channel
    
    // Use DMA with separate DRAM banks: DRAM0 for input, DRAM1 for output
    // This avoids same-bank buffer sharing issues
    bool use_dma = dram0_available && dram1_available &&
                   (inp_bytes <= IDMA_BUFFER_SIZE_DRAM0) &&
                   (out_bytes <= IDMA_BUFFER_SIZE_DRAM1);
    
    if (use_dma) {
      float32_t* inp_local = (float32_t*)ptr_dram0;
      float32_t* out_local = (float32_t*)ptr_dram1;
      
      dma_2dm_init(0);
      dma_2dm_init(1);
      
      // DMA load input via channel 0: system memory -> DRAM0
      dma_1dm(0, (void*)input_data, inp_local, inp_bytes);
      idma_hw_wait_all(0);
      
      // SIMD process: DRAM0 input -> DRAM1 output
      simd_mean_pool_2x2_to_1x1_float32(out_local, inp_local, total_channels * 4);
      
      // DMA store output via channel 1: DRAM1 -> system memory
      dma_1dm(0, out_local, (void*)out_data, out_bytes);
      idma_hw_wait_all(0);
      
      // // Invalidate output cache so next operator sees DMA-written data
      // xthal_dcache_region_invalidate(out_data, sizeof(float) * out.numel());
      
      TIME_END(mean_simd_optimized);
      TIME_DISPLAY(mean_simd_optimized, total_channels, "channels (DMA)");
      return out;
    }
    
    // Fallback: Direct SIMD without DMA (data fits or no DRAM)
    simd_mean_pool_2x2_to_1x1_float32(out_data, input_data, total_channels * 4);
    xthal_dcache_region_writeback(out_data, sizeof(float) * out.numel());

    TIME_END(mean_simd_optimized);
    TIME_DISPLAY(mean_simd_optimized, total_channels, "channels (SIMD no-DMA)");
    return out;
  }

  // Fallback to portable implementation
  TIME_DECL(mean_portable_fallback);
  TIME_START(mean_portable_fallback);
  ET_SWITCH_REALHB_TYPES(in.scalar_type(), ctx, name, CTYPE_IN, [&] {
    ET_SWITCH_FLOATH_TYPES(out.scalar_type(), ctx, name, CTYPE_OUT, [&] {
      CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
      const size_t num = torch::executor::get_reduced_dim_product(in, dim_list);

      for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
        CTYPE_OUT sum = 0;
        if (in.numel() > 0) {
          sum = torch::executor::map_reduce_over_dim_list<CTYPE_IN, CTYPE_OUT>(
              [](CTYPE_IN v) { return static_cast<CTYPE_OUT>(v); },
              [](CTYPE_OUT outv, CTYPE_OUT acc) { return acc + outv; },
              in,
              dim_list,
              out_ix);
        }
        out_data[out_ix] = sum / static_cast<float>(num);
      }
    });
  });

  TIME_END(mean_portable_fallback);
  TIME_DISPLAY(mean_portable_fallback, out.numel(), "elements (portable)");

  return out;
}

Tensor& mean_dim_out(
    RuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  return mean_out(ctx, in, dim_list, keepdim, dtype, out);
}

} // namespace native
} // namespace vision
} // namespace impl
