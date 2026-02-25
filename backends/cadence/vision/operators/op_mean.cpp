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
    
    // Calculate number of elements to process
    // For NCHW with 2x2 spatial: N * C * 4 input -> N * C output
    int batch_size = in.size(0);
    int channels = in.size(1);
    int total_channels = batch_size * channels;
    
    // DMA optimization for large channel counts
    bool ping_pong_process = false;
    bool ping_process_pong = false;
    size_t chunk_channels = 0;
    
    float32_t* inp_buff[2];
    float32_t* out_buff[2];
    
    // Check if DRAM buffers are available
    bool dram0_available = (ptr_dram0 != nullptr) && (DRAM0_BUFF_SIZE > 0);
    bool dram1_available = (ptr_dram1 != nullptr) && (DRAM1_BUFF_SIZE > 0);
    
    // DMA threshold: beneficial for larger channel counts
    // Each channel: 4 float32 input (16 bytes) + 1 float32 output (4 bytes)
    const size_t DMA_THRESHOLD = 128;  // 128 channels = 2KB input
    bool use_dma = (total_channels >= DMA_THRESHOLD);
    
    // Strategy 1: Ping-pong (2 input + 2 output buffers)
    // Split: 80% input (4 floats/channel), 20% output (1 float/channel)
    if (use_dma && dram0_available && dram1_available && (total_channels >= 2)) {
      // Calculate 80% of DRAM for input, aligned to 64 bytes for DMA efficiency
      size_t inp_buffer_bytes = ((DRAM0_BUFF_SIZE * 4) / 5) & ~0x3F;  // 64-byte aligned
      size_t out_buffer_bytes = ((DRAM0_BUFF_SIZE * 1) / 5) & ~0x3F;  // 64-byte aligned
      
      // How many channels fit in input buffer (4 floats per channel = 16 bytes)
      // CRITICAL: SIMD function processes 16 channels at a time (64 input floats)
      // So chunk_channels MUST be multiple of 16 for correct SIMD operation
      size_t inp_per_buffer = inp_buffer_bytes / (4 * FLT32_SIZE);
      inp_per_buffer = (inp_per_buffer / 16) * 16;  // Round down to multiple of 16
      size_t out_per_buffer = out_buffer_bytes / FLT32_SIZE;
      
      // Check if buffers fit (minimum 16 channels = 64 input floats = 256 bytes)
      if ((inp_per_buffer >= 16) && (out_per_buffer >= inp_per_buffer)) {
        
        // Allocate with 80/20 split, aligned offsets
        inp_buff[0] = (float32_t*)ptr_dram0;
        out_buff[0] = (float32_t*)((uint8_t*)ptr_dram0 + inp_buffer_bytes);
        
        inp_buff[1] = (float32_t*)ptr_dram1;
        out_buff[1] = (float32_t*)((uint8_t*)ptr_dram1 + inp_buffer_bytes);
        
        chunk_channels = inp_per_buffer;
        ping_pong_process = true;
      }
    }
    
    // Strategy 2: Ping-process-pong (1 input + 1 output buffer)
    if (use_dma && !ping_pong_process && dram0_available && dram1_available) {
      size_t inp_capacity = DRAM0_BUFF_SIZE / (4 * FLT32_SIZE);  // channels (4 floats each)
      size_t out_capacity = DRAM1_BUFF_SIZE / FLT32_SIZE;  // channels (1 float each)
      
      if ((inp_capacity > 0) && (out_capacity >= inp_capacity)) {
        inp_buff[0] = (float32_t*)ptr_dram0;
        out_buff[0] = (float32_t*)ptr_dram1;
        
        chunk_channels = (inp_capacity < out_capacity) ? inp_capacity : out_capacity;
        ping_process_pong = true;
      }
    }
    
    if (ping_pong_process || ping_process_pong) {
      const float32_t* ptr_inp = input_data;
      float32_t* ptr_out = out_data;
      
      // Initialize DMA Channel 0
      idma_init(0, 0, MAX_BLOCK_16, 8, TICK_CYCLES_1, 0, NULL);
      idma_init_loop(0, buffer_idma_ch_2d, IDMA_2D_DESC, 1, NULL, NULL);
      
      if (ping_pong_process) {
        // Ping-pong: overlap load/compute/store
        size_t num_chunks = (total_channels + chunk_channels - 1) / chunk_channels;
        size_t channels_remaining = total_channels;
        
        int32_t idx_in = 0, idx_out = 0;
        size_t current_chunk_channels = 0;
        
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
          current_chunk_channels = (channels_remaining > chunk_channels) ? chunk_channels : channels_remaining;
          
          // Ensure transfer sizes are properly aligned (64-byte minimum for DMA)
          size_t current_chunk_inp_bytes = current_chunk_channels * 4 * FLT32_SIZE;  // 4 floats/channel
          size_t current_chunk_out_bytes = current_chunk_channels * FLT32_SIZE;  // 1 float/channel
          
          int swap = chunk_idx & 1;
          
          // Load input chunk (async)
          idx_in = idma_copy_2d_desc(0, inp_buff[swap], (void*)ptr_inp,
                                     current_chunk_inp_bytes, DESC_IDMA_PRIOR_H, 1, 0, 0);
          
          // Wait for load to complete
          idma_desc_done(0, idx_in);
          
          // Process chunk
          simd_mean_pool_2x2_to_1x1_float32(
              out_buff[swap], 
              inp_buff[swap],
              current_chunk_channels * 4);
          
          // Store output chunk (async)
          idx_out = idma_copy_2d_desc(0, (void*)ptr_out, out_buff[swap],
                                      current_chunk_out_bytes, DESC_IDMA_PRIOR_H, 1, 0, 0);
          
          // Wait for store to complete before next iteration
          idma_desc_done(0, idx_out);
          
          ptr_inp += current_chunk_channels * 4;
          ptr_out += current_chunk_channels;
          channels_remaining -= current_chunk_channels;
        }
        
        TIME_END(mean_simd_optimized);
        TIME_DISPLAY(mean_simd_optimized, total_channels, "channels (DMA ping-pong)");
        return out;
        
      } else if (ping_process_pong) {
        // Ping-process-pong: load→process→store sequentially
        size_t num_chunks = (total_channels + chunk_channels - 1) / chunk_channels;
        size_t channels_remaining = total_channels;
        
        int32_t idx_in = 0, idx_out = 0;
        
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
          size_t current_chunk_channels = (channels_remaining > chunk_channels) ? chunk_channels : channels_remaining;
          size_t current_chunk_inp_bytes = current_chunk_channels * 4 * FLT32_SIZE;
          size_t current_chunk_out_bytes = current_chunk_channels * FLT32_SIZE;
          
          // Load
          idx_in = idma_copy_2d_desc(0, inp_buff[0], (void*)ptr_inp,
                                     current_chunk_inp_bytes, DESC_IDMA_PRIOR_H, 1, 0, 0);
          idma_desc_done(0, idx_in);
          
          // Process
          simd_mean_pool_2x2_to_1x1_float32(
              out_buff[0],
              inp_buff[0],
              current_chunk_channels * 4);
          
          // Store
          idx_out = idma_copy_2d_desc(0, (void*)ptr_out, out_buff[0],
                                      current_chunk_out_bytes, DESC_IDMA_PRIOR_H, 1, 0, 0);
          idma_desc_done(0, idx_out);
          
          ptr_inp += current_chunk_channels * 4;
          ptr_out += current_chunk_channels;
          channels_remaining -= current_chunk_channels;
        }
        
        TIME_END(mean_simd_optimized);
        TIME_DISPLAY(mean_simd_optimized, total_channels, "channels (DMA sequential)");
        return out;
      }
    }
    
    // Fallback: Direct SIMD without DMA
    simd_mean_pool_2x2_to_1x1_float32(
        out_data, 
        input_data, 
        total_channels * 4);
    
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
