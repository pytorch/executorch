/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <api.h>
#include <lib.h>
#include <algorithm>
#include <cmath>
#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::getLeadingDims;
using executorch::runtime::KernelRuntimeContext;

namespace impl {
namespace vision {
namespace native {

// Generic fallback implementation
template <typename T>
void quantized_linear_per_tensor_generic_(
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t src_zero_point,
    const int64_t weight_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    Tensor& out) {
  
  const int64_t leading_dims = getLeadingDims(src, src.dim() - 1);
  const int64_t out_dim = weight.size(0);
  const int64_t in_dim = weight.size(1);

  const T* __restrict__ in_data = src.const_data_ptr<T>();
  const T* __restrict__ weight_data = weight.const_data_ptr<T>();
  const int32_t* __restrict__ bias_data = bias.const_data_ptr<int32_t>();
  T* __restrict__ out_data = out.mutable_data_ptr<T>();

  // Compute the requant_scale from out_multiplier and out_shift
  const float requant_scale =
      -out_multiplier * 1.0 / (1 << 31) * pow(2, out_shift);

  for (size_t i = 0; i < leading_dims; ++i) {
    for (size_t j = 0; j < out_dim; ++j) {
      int32_t sum = bias_data[j];
      for (size_t k = 0; k < in_dim; ++k) {
        int32_t x = (int32_t)in_data[i * in_dim + k] - src_zero_point;
        int32_t w = (int32_t)weight_data[j * in_dim + k] - (int32_t)weight_zero_point;
        sum += x * w;
      }
      
      out_data[i * out_dim + j] = 
          ::impl::generic::kernels::quantize<T>(sum, requant_scale, out_zero_point);
    }
  }
}

void quantized_linear_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t src_zero_point,
    const int64_t weight_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    __ET_UNUSED const std::optional<Tensor>& offset,
    Tensor& out) {
  
  TIME_DECL(quantized_linear);
  TIME_START(quantized_linear);

  const int64_t leading_dims = getLeadingDims(src, src.dim() - 1);
  const int64_t out_dim = weight.size(0);
  const int64_t in_dim = weight.size(1);
  const size_t numel = leading_dims * out_dim;  // Total output elements
  
  bool use_optimized = false;
  // Check if we can use Vision SIMD path for quantized data
  if (src.scalar_type() == ScalarType::Char &&
      weight.scalar_type() == ScalarType::Char &&
      out.scalar_type() == ScalarType::Char &&
      in_dim >= 16) {  // Minimum size for SIMD benefit
    use_optimized = true;
  }

  if (use_optimized) {
    const int8_t* in_data = src.const_data_ptr<int8_t>();
    const int8_t* weight_data = weight.const_data_ptr<int8_t>();
    const int32_t* bias_data = bias.const_data_ptr<int32_t>();
    int8_t* out_data = out.mutable_data_ptr<int8_t>();

    // Cache coherency: invalidate inputs that may have been DMA-written by prior ops
    xthal_dcache_region_invalidate((void*)in_data, sizeof(int8_t) * src.numel());
    xthal_dcache_region_invalidate((void*)weight_data, sizeof(int8_t) * weight.numel());
    xthal_dcache_region_invalidate((void*)bias_data, sizeof(int32_t) * bias.numel());
    
    const int32_t in_zp = static_cast<int32_t>(src_zero_point);
    const int32_t weight_zp = static_cast<int32_t>(weight_zero_point);
    const int32_t out_zp = static_cast<int32_t>(out_zero_point);
    
    // Compute requant scale
    const float requant_scale = 
        -out_multiplier * 1.0f / (1 << 31) * std::pow(2.0f, (float)out_shift);
    
    // Check if DRAM buffers are available for DMA
    bool dram0_available = (ptr_dram0 != nullptr) && (IDMA_BUFFER_SIZE_DRAM0 > 0);
    bool dram1_available = (ptr_dram1 != nullptr) && (IDMA_BUFFER_SIZE_DRAM1 > 0);
    
    // DMA threshold: only beneficial for larger problems
    const size_t DMA_THRESHOLD = 512;  // Minimum input dimension
    bool use_dma = (in_dim >= DMA_THRESHOLD) && dram0_available && dram1_available;
    
    if (use_dma && leading_dims == 1) {
      // Single sample: DMA-optimized tiling (block prefetch) processing
      size_t input_buffer_size = in_dim;  // One input row
      // Determine max number of weight rows (tile) that fit in DRAM1
      size_t max_tile_rows = IDMA_BUFFER_SIZE_DRAM1 / in_dim;
      if (max_tile_rows == 0) max_tile_rows = 1;
      // Limit tile size to out_dim if needed
      size_t tile_rows = (max_tile_rows < out_dim) ? max_tile_rows : out_dim;
      // Allocate buffers
      int8_t* input_cache = (int8_t*)ptr_dram0;
      int8_t* weight_tile = (int8_t*)ptr_dram1;
      // Initialize DMA Channel 0
      dma_2dm_init(0);
      // Load input row ONCE and cache it
      int32_t idx_in = idma_copy_2d_desc(0, input_cache, (void*)in_data, 
                                        input_buffer_size, DESC_IDMA_PRIOR_H, 1, 0, 0);
      idma_desc_done(0, idx_in);
      // Process output in tiles
      for (size_t j_tile = 0; j_tile < out_dim; j_tile += tile_rows) {
        size_t curr_tile = ((j_tile + tile_rows) <= out_dim) ? tile_rows : (out_dim - j_tile);
        // Prefetch a tile of weight rows
        int32_t idx_weight = idma_copy_2d_desc(0, weight_tile, (void*)(weight_data + j_tile * in_dim),
                                              curr_tile * in_dim, DESC_IDMA_PRIOR_H, 1, 0, 0);
        idma_desc_done(0, idx_weight);
        // Compute for all rows in tile
        for (size_t j = 0; j < curr_tile; ++j) {
          int32_t acc = bias_data[j_tile + j];
          acc = rvdot_zeropt(
              acc,
              input_cache,
              weight_tile + j * in_dim,
              in_zp,
              weight_zp,
              (int)in_dim);
          out_data[j_tile + j] = ::impl::generic::kernels::quantize<int8_t>(acc, requant_scale, out_zp);
        }
      }
      // Writeback output from cache to system memory for DMA coherency
      xthal_dcache_region_writeback(out_data, sizeof(int8_t) * numel);

      TIME_END(quantized_linear);
      TIME_DISPLAY(quantized_linear, numel, "SIMD elements (DMA tiling)");
      return;
    }
    
    // Fallback: No DMA or multi-sample - use direct SIMD
    for (size_t i = 0; i < leading_dims; ++i) {
      const int8_t* in_row = &in_data[i * in_dim];
      
      for (size_t j = 0; j < out_dim; ++j) {
        const int8_t* weight_row = &weight_data[j * in_dim];
        
        int32_t acc = bias_data[j];
        acc = rvdot_zeropt(
            acc,
            in_row,
            weight_row,
            in_zp,
            weight_zp,
            (int)in_dim);
        
        out_data[i * out_dim + j] = 
            ::impl::generic::kernels::quantize<int8_t>(acc, requant_scale, out_zp);
      }
    }
    
    // Writeback output from cache to system memory for DMA coherency
    xthal_dcache_region_writeback(out_data, sizeof(int8_t) * numel);

    TIME_END(quantized_linear);
    TIME_DISPLAY(quantized_linear, numel, "SIMD elements (no DMA)");
    
  } else {
    // Fallback: use generic implementation
    if (out.scalar_type() == ScalarType::Char) {
      quantized_linear_per_tensor_generic_<int8_t>(
          src, weight, bias,
          src_zero_point, weight_zero_point,
          out_multiplier, out_shift, out_zero_point,
          out);
    } else if (out.scalar_type() == ScalarType::Byte) {
      quantized_linear_per_tensor_generic_<uint8_t>(
          src, weight, bias,
          src_zero_point, weight_zero_point,
          out_multiplier, out_shift, out_zero_point,
          out);
    } else {
      ET_CHECK_MSG(
          false,
          "Unhandled output dtype %hhd",
          static_cast<int8_t>(out.scalar_type()));
    }
    
    TIME_END(quantized_linear);
    TIME_DISPLAY(quantized_linear, numel, "Generic elements");
  }
}

// Wrapper functions for different quantization schemes
void quantized_linear_asym8sxasym8s_asym8s_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t src_zero_point,
    const int64_t weight_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const std::optional<Tensor>& offset,
    Tensor& out) {
  quantized_linear_per_tensor_out(
      ctx, src, weight, bias,
      src_zero_point, weight_zero_point,
      out_multiplier, out_shift, out_zero_point,
      offset, out);
}

void quantized_linear_asym8uxasym8u_asym8u_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& src,
    const Tensor& weight,
    const Tensor& bias,
    const int64_t src_zero_point,
    const int64_t weight_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    const int64_t out_zero_point,
    const std::optional<Tensor>& offset,
    Tensor& out) {
  quantized_linear_per_tensor_out(
      ctx, src, weight, bias,
      src_zero_point, weight_zero_point,
      out_multiplier, out_shift, out_zero_point,
      offset, out);
}

} // namespace native
} // namespace vision
} // namespace impl
