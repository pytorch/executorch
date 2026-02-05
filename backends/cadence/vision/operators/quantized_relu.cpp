/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <lib.h>
#include <algorithm>
#include <cmath>
#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

// Forward declaration of Vision SIMD quantized ReLU
extern "C" void vrelU(
    uint8_t* y,
    const int8_t* x,
    const uint8_t minVal,
    uint8_t maxVal,
    int N);

#define ET_FORALL_CADENCE_QUANTIZED_TYPES(_) \
  _(uint8_t, Byte)                           \
  _(int8_t, Char)

namespace impl {
namespace vision {
namespace native {

// Generic fallback implementation (from generic/operators/quantized_relu_out.cpp)
template <typename T>
void quantized_relu_per_tensor_out_(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    int64_t in_zero_point,
    int64_t out_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    Tensor& output) {
  const T* __restrict__ in = input.const_data_ptr<T>();
  T* __restrict__ out = output.mutable_data_ptr<T>();

  // Compute the out_scale from out_multiplier and out_shift
  const float out_scale = -out_multiplier * 1.0 / (1 << 31) * pow(2, out_shift);

  for (size_t i = 0, e = input.numel(); i < e; ++i) {
    const float temp = in[i] > in_zero_point ? (in[i] - in_zero_point) : 0;
    out[i] = generic::kernels::quantize<T>(temp, out_scale, out_zero_point);
  }
}


void quantized_relu_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output) {
  
  TIME_DECL(quantized_relu);
  TIME_START(quantized_relu);

  size_t numel = input.numel();
  
  // Check if we can use Vision SIMD path for quantized data
  // vrelU supports int8/uint8 input and output (with appropriate casting)
  bool use_optimized = (input.scalar_type() == ScalarType::Char || input.scalar_type() == ScalarType::Byte) &&
                       (output.scalar_type() == ScalarType::Char || output.scalar_type() == ScalarType::Byte) &&
                       (numel >= 16);

  if (use_optimized) {
    // Vision-optimized SIMD path using vrelU with iDMA support
    // vrelU requires int8_t* input and uint8_t* output, cast appropriately
    const int8_t* in_data;
    if (input.scalar_type() == ScalarType::Char) {
      in_data = input.const_data_ptr<int8_t>();
    } else {
      in_data = reinterpret_cast<const int8_t*>(input.const_data_ptr<uint8_t>());
    }
    
    uint8_t* out_data;
    if (output.scalar_type() == ScalarType::Byte) {
      out_data = output.mutable_data_ptr<uint8_t>();
    } else {
      out_data = reinterpret_cast<uint8_t*>(output.mutable_data_ptr<int8_t>());
    }
    
    // For quantized operations and dumps, we need int8_t view of output
    int8_t* out_data_int8 = reinterpret_cast<int8_t*>(out_data);
    
    // vrelU clamps: max(max(x, 0), minVal) and min(result, maxVal)
    uint8_t minVal = 0;      // ReLU minimum is 0
    uint8_t maxVal = 255;    // uint8 max
    
    // Common quantization parameters (used by both DMA and non-DMA paths)
    const float out_scale = -out_multiplier * 1.0f / (1 << 31) * std::pow(2.0f, (float)out_shift);
    const int32_t in_zp = static_cast<int32_t>(in_zero_point);
    const int32_t out_zp = static_cast<int32_t>(out_zero_point);
    
    // DMA setup
    bool ping_pong_process = false;
    bool ping_process_pong = false;
    size_t chunk_size = 0;

    int8_t* inp_buff[2];
    uint8_t* out_buff[2];

    // Check if DRAM buffers are available
    bool dram0_available = (ptr_dram0 != nullptr) && (DRAM0_BUFF_SIZE > 0);
    bool dram1_available = (ptr_dram1 != nullptr) && (DRAM1_BUFF_SIZE > 0);
    
    // DMA has overhead - only beneficial for larger tensors
    // Threshold: 1024 elements (~1KB for int8 input/output)
    const size_t DMA_THRESHOLD = 1024;
    bool use_dma = (numel >= DMA_THRESHOLD);

    // Strategy 1: Try ping-pong processing (2 input + 2 output buffers)
    // Using 50/50 split: both int8/uint8 are 1 byte each
    if (use_dma && dram0_available && dram1_available && (numel >= 2)) {
      size_t per_buffer = (DRAM0_BUFF_SIZE / 2);  // 50% for int8 input (in bytes)
      
      // Check if 50/50 split fits in both DRAMs
      if ((per_buffer > 0) && 
          ((DRAM0_BUFF_SIZE / 2 + DRAM0_BUFF_SIZE / 2) <= DRAM0_BUFF_SIZE) &&
          ((DRAM1_BUFF_SIZE / 2 + DRAM1_BUFF_SIZE / 2) <= DRAM1_BUFF_SIZE)) {
        
        // Allocate buffers with 50/50 split
        inp_buff[0] = (int8_t*)ptr_dram0;
        out_buff[0] = (uint8_t*)((uint8_t*)ptr_dram0 + (DRAM0_BUFF_SIZE / 2));
        
        inp_buff[1] = (int8_t*)ptr_dram1;
        out_buff[1] = (uint8_t*)((uint8_t*)ptr_dram1 + (DRAM1_BUFF_SIZE / 2));
        
        chunk_size = per_buffer;
        ping_pong_process = true;
      }
    }
    
    // Strategy 2: Fallback to ping-process-pong (1 input + 1 output buffer)
    // Use full DRAM0 for input, full DRAM1 for output (no split needed)
    if (use_dma && !ping_pong_process && dram0_available && dram1_available) {
      size_t inp_capacity = DRAM0_BUFF_SIZE;  // Full DRAM0 for int8 input (in bytes)
      size_t out_capacity = DRAM1_BUFF_SIZE;  // Full DRAM1 for uint8 output (in bytes)
      
      if ((inp_capacity > 0) && (out_capacity >= inp_capacity)) {
        inp_buff[0] = (int8_t*)ptr_dram0;
        out_buff[0] = (uint8_t*)ptr_dram1;
        
        chunk_size = (inp_capacity < out_capacity) ? inp_capacity : out_capacity;
        ping_process_pong = true;
      }
    }

    if (ping_pong_process || ping_process_pong) {
      const int8_t* ptr_inp = in_data;

      /* Initialize DMA Channel 0 */
      idma_init(0, 0, MAX_BLOCK_16, 8, TICK_CYCLES_1, 0, NULL);
      idma_init_loop(0, buffer_idma_ch_2d, IDMA_2D_DESC, 1, NULL, NULL);

      if (ping_pong_process) {
        // Ping-pong processing for better throughput
        size_t num_chunks = (numel + chunk_size - 1) / chunk_size;
        
        if (num_chunks == 0) num_chunks = 1;

        int32_t pp_swap = 0;
        int32_t idx_in, idx_in_prev, idx_out = 0;

        int8_t* ptr_in = (int8_t*)ptr_inp;
        uint8_t* ptr_out = out_data;

        // Load first chunk
        size_t current_chunk = (numel < chunk_size) ? numel : chunk_size;
        idx_in_prev = idma_copy_2d_desc(0, inp_buff[pp_swap], ptr_in, sizeof(int8_t) * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
        pp_swap = pp_swap ^ 1;

        size_t remaining = numel - current_chunk;
        ptr_in += current_chunk;

        // Pipeline: load next, process current, store previous
        for (size_t i = 0; i < (num_chunks - 1); i++) {
          // Start loading NEXT chunk into alternate buffer
          size_t next_chunk = (remaining < chunk_size) ? remaining : chunk_size;
          idx_in = idma_copy_2d_desc(0, inp_buff[pp_swap], ptr_in, sizeof(int8_t) * next_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
          pp_swap = pp_swap ^ 1;
          
          // Wait for PREVIOUS load to complete (not the one just started)
          idma_desc_done(0, idx_in_prev);

          // Wait for previous store to complete
          if (i > 0) {
            idma_desc_done(0, idx_out);
          }

          // Process current buffer (the one loaded in previous iteration)
          int8_t* out_chunk_int8 = reinterpret_cast<int8_t*>(out_buff[pp_swap]);
          vrelU_quantized(out_chunk_int8, inp_buff[pp_swap], in_zp, out_zp, out_scale, (int)current_chunk);

          // Store result
          idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[pp_swap], sizeof(uint8_t) * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);

          ptr_in += next_chunk;
          ptr_out += current_chunk;
          remaining -= next_chunk;
          current_chunk = next_chunk;
          idx_in_prev = idx_in;  // Save for next iteration
        }

        pp_swap = pp_swap ^ 1;

        // Process last chunk
        idma_desc_done(0, idx_in);  // Wait for last load
        idma_desc_done(0, idx_out);  // Wait for previous store
        int8_t* out_last_int8 = reinterpret_cast<int8_t*>(out_buff[pp_swap]);
        vrelU_quantized(out_last_int8, inp_buff[pp_swap], in_zp, out_zp, out_scale, (int)current_chunk);

        idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[pp_swap], sizeof(uint8_t) * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
        idma_desc_done(0, idx_out);
        
        TIME_END(quantized_relu);
        TIME_DISPLAY(quantized_relu, numel, "elements (DMA ping-pong)");
      } 
      else if (ping_process_pong) {
        // Simple sequential processing
        size_t remaining = numel;
        int8_t* ptr_in = (int8_t*)ptr_inp;
        uint8_t* ptr_out = out_data;

        while (remaining > 0) {
          size_t current_chunk = (remaining < chunk_size) ? remaining : chunk_size;

          // Load chunk
          int32_t idx_in = idma_copy_2d_desc(0, inp_buff[0], ptr_in, sizeof(int8_t) * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
          idma_desc_done(0, idx_in);

          // Process
          int8_t* out_chunk_int8 = reinterpret_cast<int8_t*>(out_buff[0]);
          vrelU_quantized(out_chunk_int8, inp_buff[0], in_zp, out_zp, out_scale, (int)current_chunk);

          // Store
          int32_t idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[0], sizeof(uint8_t) * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
          idma_desc_done(0, idx_out);

          ptr_in += current_chunk;
          ptr_out += current_chunk;
          remaining -= current_chunk;
        }
        
        TIME_END(quantized_relu);
        TIME_DISPLAY(quantized_relu, numel, "elements (DMA ping-process-pong)");
      }
    } else {
      // Fallback: use SIMD function directly without DMA
      // Use common parameters already computed above
      
      vrelU_quantized(
          out_data_int8,
          in_data,
          in_zp,
          out_zp,
          out_scale,
          (int)numel);
      
      TIME_END(quantized_relu);
      TIME_DISPLAY(quantized_relu, numel, "elements (HW-optimized, no DMA)");
    }
    
  } else {
    // Fallback: use generic implementation with template dispatching
    
#define typed_quantized_relu(ctype, dtype)    \
  case executorch::aten::ScalarType::dtype: { \
    quantized_relu_per_tensor_out_<ctype>(    \
        ctx,                                  \
        input,                                \
        in_zero_point,                        \
        out_zero_point,                       \
        out_multiplier,                       \
        out_shift,                            \
        output);                              \
    break;                                    \
  }

    executorch::aten::ScalarType dtype = input.scalar_type();
    switch (dtype) {
      ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_relu)
      default:
        ET_DCHECK_MSG(
            false, "Unhandled dtype %s", torch::executor::toString(dtype));
    }

#undef typed_quantized_relu
    
    TIME_END(quantized_relu);
    TIME_DISPLAY(quantized_relu, numel, "elements (generic template)");
  }
}

} // namespace native
} // namespace vision
} // namespace impl
