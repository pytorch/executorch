/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <lib.h>
#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::dequantize;

namespace impl {
namespace vision {
namespace native {

// Forward declaration of hardware-optimized dequantize function
extern "C" void dequantize_asym8s_f32(
    float32_t* restrict ptr_out,
    const int8_t* restrict ptr_inp,
    float32_t scale,
    int zero_bias,
    int N);

Tensor& dequantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  float* out_data = out.mutable_data_ptr<float>();
  size_t numel = out.numel();

  if (input.scalar_type() == ScalarType::Byte) {
    const uint8_t* input_data = input.const_data_ptr<uint8_t>();
    dequantize<uint8_t>(out_data, input_data, scale, zero_point, numel);
  } else if (input.scalar_type() == ScalarType::Char) {
    TIME_DECL(dequantize_asym8s);
    TIME_START(dequantize_asym8s);
    
    const int8_t* input_data = input.const_data_ptr<int8_t>();
    
    // Hardware-optimized int8 dequantization with DMA support
    bool ping_pong_process = false;
    bool ping_process_pong = false;
    size_t chunk_size = 0;

    int8_t* inp_buff[2];
    float32_t* out_buff[2];

    // Check if DRAM buffers are available
    bool dram0_available = (ptr_dram0 != nullptr) && (DRAM0_BUFF_SIZE > 0);
    bool dram1_available = (ptr_dram1 != nullptr) && (DRAM1_BUFF_SIZE > 0);
    
    // DMA has overhead - only beneficial for larger tensors
    // Threshold: 1024 elements (~1KB for int8, ~4KB for float32)
    const size_t DMA_THRESHOLD = 1024;
    bool use_dma = (numel >= DMA_THRESHOLD);

    // Strategy 1: Try ping-pong processing (2 input + 2 output buffers)
    // Using 20/80 split: 20% for int8 input, 80% for float32 output in each DRAM
    if (use_dma && dram0_available && dram1_available && (numel >= 2)) {
      size_t inp_per_buffer = (DRAM0_BUFF_SIZE * 1) / 5;  // 20% for int8 input (in bytes)
      size_t out_per_buffer = (DRAM0_BUFF_SIZE * 4) / (5 * FLT32_SIZE);  // 80% for float32 output

      // Check if 20/80 split fits in both DRAMs
      if ((inp_per_buffer > 0) && 
          (out_per_buffer >= inp_per_buffer) &&
          ((DRAM0_BUFF_SIZE * 1) / 5 + (DRAM0_BUFF_SIZE * 4) / 5 <= DRAM0_BUFF_SIZE) &&
          ((DRAM1_BUFF_SIZE * 1) / 5 + (DRAM1_BUFF_SIZE * 4) / 5 <= DRAM1_BUFF_SIZE)) {
        
        // Allocate buffers with 20/80 split
        inp_buff[0] = (int8_t*)ptr_dram0;
        out_buff[0] = (float32_t*)((uint8_t*)ptr_dram0 + (DRAM0_BUFF_SIZE * 1) / 5);
        
        inp_buff[1] = (int8_t*)ptr_dram1;
        out_buff[1] = (float32_t*)((uint8_t*)ptr_dram1 + (DRAM1_BUFF_SIZE * 1) / 5);
        
        chunk_size = inp_per_buffer;
        ping_pong_process = true;
      }
    }
    
    // Strategy 2: Fallback to ping-process-pong (1 input + 1 output buffer)
    // Use full DRAM0 for input, full DRAM1 for output (no split needed)
    if (use_dma && !ping_pong_process && dram0_available && dram1_available) {
      size_t inp_capacity = DRAM0_BUFF_SIZE;  // Full DRAM0 for int8 input (in bytes)
      size_t out_capacity = DRAM1_BUFF_SIZE / FLT32_SIZE;  // Full DRAM1 for float32 output
      
      if ((inp_capacity > 0) && (out_capacity >= inp_capacity)) {
        inp_buff[0] = (int8_t*)ptr_dram0;
        out_buff[0] = (float32_t*)ptr_dram1;
        
        chunk_size = (inp_capacity < out_capacity) ? inp_capacity : out_capacity;
        ping_process_pong = true;
      }
    }

    if (ping_pong_process || ping_process_pong) {
      const int8_t* ptr_inp = input_data;

      /* Initialize DMA Channel 0 */
      idma_init(0, 0, MAX_BLOCK_16, 8, TICK_CYCLES_1, 0, NULL);
      idma_init_loop(0, buffer_idma_ch_2d, IDMA_2D_DESC, 1, NULL, NULL);

      if (ping_pong_process) {
        // Ping-pong processing for better throughput
        size_t num_chunks = (numel + chunk_size - 1) / chunk_size;
        
        if (num_chunks == 0) num_chunks = 1;

        int32_t pp_swap = 0;
        int32_t idx_in, idx_out = 0;

        int8_t* ptr_in = (int8_t*)ptr_inp;
        float32_t* ptr_out = out_data;

        // Load first chunk
        size_t current_chunk = (numel < chunk_size) ? numel : chunk_size;
        idx_in = idma_copy_2d_desc(0, inp_buff[pp_swap], ptr_in, sizeof(int8_t) * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
        pp_swap = pp_swap ^ 1;

        size_t remaining = numel - current_chunk;
        ptr_in += current_chunk;

        // Pipeline: load next, process current, store previous
        for (size_t i = 0; i < (num_chunks - 1); i++) {
          // Wait for previous store to complete
          if (i > 0) {
            idma_desc_done(0, idx_out);
          }

          // Load next chunk
          size_t next_chunk = (remaining < chunk_size) ? remaining : chunk_size;
          idx_in = idma_copy_2d_desc(0, inp_buff[pp_swap], ptr_in, sizeof(int8_t) * next_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
          pp_swap = pp_swap ^ 1;

          // Wait for load to complete
          idma_desc_done(0, idx_in);

          // Process (pp_swap now points to the buffer that was just loaded)
          dequantize_asym8s_f32(out_buff[pp_swap], inp_buff[pp_swap], (float)scale, (int)zero_point, (int)current_chunk);

          // Store result
          idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[pp_swap], FLT32_SIZE * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);

          ptr_in += next_chunk;
          ptr_out += current_chunk;
          remaining -= next_chunk;
          current_chunk = next_chunk;
        }

        pp_swap = pp_swap ^ 1;

        // Process last chunk
        idma_desc_done(0, idx_in);
        dequantize_asym8s_f32(out_buff[pp_swap], inp_buff[pp_swap], (float)scale, (int)zero_point, (int)current_chunk);

        idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[pp_swap], FLT32_SIZE * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
        idma_desc_done(0, idx_out);
        
        TIME_END(dequantize_asym8s);
        TIME_DISPLAY(dequantize_asym8s, numel, "elements (DMA ping-pong)");
      } 
      else if (ping_process_pong) {
        // Simple sequential processing
        size_t remaining = numel;
        int8_t* ptr_in = (int8_t*)ptr_inp;
        float32_t* ptr_out = out_data;

        while (remaining > 0) {
          size_t current_chunk = (remaining < chunk_size) ? remaining : chunk_size;

          // Load chunk
          int32_t idx_in = idma_copy_2d_desc(0, inp_buff[0], ptr_in, sizeof(int8_t) * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
          idma_desc_done(0, idx_in);

          // Process
          dequantize_asym8s_f32(out_buff[0], inp_buff[0], (float)scale, (int)zero_point, (int)current_chunk);

          // Store
          int32_t idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[0], FLT32_SIZE * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
          idma_desc_done(0, idx_out);

          ptr_in += current_chunk;
          ptr_out += current_chunk;
          remaining -= current_chunk;
        }
        
        TIME_END(dequantize_asym8s);
        TIME_DISPLAY(dequantize_asym8s, numel, "elements (DMA ping-process-pong)");
      }
      
      // TIME_END and TIME_DISPLAY now called inside each branch
    } else {
      // No DMA: use hardware function on full tensor at once
      dequantize_asym8s_f32(out_data, input_data, (float)scale, (int)zero_point, (int)numel);
      TIME_END(dequantize_asym8s);
      TIME_DISPLAY(dequantize_asym8s, numel, "elements (HW-optimized, no DMA)");
    }
  } else if (
      input.scalar_type() == ScalarType::Bits16 ||
      input.scalar_type() == ScalarType::UInt16) {
    const uint16_t* input_data = input.const_data_ptr<uint16_t>();
    dequantize<uint16_t>(out_data, input_data, scale, zero_point, numel);
  } else if (input.scalar_type() == ScalarType::Short) {
    const int16_t* input_data = input.const_data_ptr<int16_t>();
    dequantize<int16_t>(out_data, input_data, scale, zero_point, numel);
  } else if (input.scalar_type() == ScalarType::Int) {
    const int32_t* input_data = input.const_data_ptr<int32_t>();
    dequantize<int32_t>(out_data, input_data, scale, zero_point, numel);
  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(input.scalar_type()));
  }
  return out;
}

// int8 dequantization - uses generic template
Tensor& dequantize_per_tensor_asym8s_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  float* out_data = out.mutable_data_ptr<float>();
  size_t numel = out.numel();
  const int8_t* input_data = input.const_data_ptr<int8_t>();
  dequantize<int8_t>(out_data, input_data, scale, zero_point, numel);
  return out;
}

// uint8 dequantization - uses generic template
Tensor& dequantize_per_tensor_asym8u_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  float* out_data = out.mutable_data_ptr<float>();
  size_t numel = out.numel();
  const uint8_t* input_data = input.const_data_ptr<uint8_t>();
  dequantize<uint8_t>(out_data, input_data, scale, zero_point, numel);
  return out;
}

// int16 dequantization - uses generic template
Tensor& dequantize_per_tensor_asym16s_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  float* out_data = out.mutable_data_ptr<float>();
  size_t numel = out.numel();
  const int16_t* input_data = input.const_data_ptr<int16_t>();
  dequantize<int16_t>(out_data, input_data, scale, zero_point, numel);
  return out;
}

// uint16 dequantization - uses generic template
Tensor& dequantize_per_tensor_asym16u_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  float* out_data = out.mutable_data_ptr<float>();
  size_t numel = out.numel();
  const uint16_t* input_data = input.const_data_ptr<uint16_t>();
  dequantize<uint16_t>(out_data, input_data, scale, zero_point, numel);
  return out;
}

// int32 dequantization - uses generic template
Tensor& dequantize_per_tensor_asym32s_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  float* out_data = out.mutable_data_ptr<float>();
  size_t numel = out.numel();
  const int32_t* input_data = input.const_data_ptr<int32_t>();
  dequantize<int32_t>(out_data, input_data, scale, zero_point, numel);
  return out;
}

} // namespace native
} // namespace vision
} // namespace impl
