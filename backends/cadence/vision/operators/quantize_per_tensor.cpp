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

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::quantize;

namespace impl {
namespace vision {
namespace native {

// Forward declaration of hardware-optimized quantize function
extern "C" void quantize_f32_asym8s(
    int8_t* restrict ptr_out,
    const float32_t* restrict ptr_inp,
    float32_t scale,
    int zero_bias,
    int N);

// Quantize the input tensor (PT2 version). Note that quant_<min,max> are not
// used in any computation.
Tensor& quantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
 
  if (out.scalar_type() == ScalarType::Byte) {
    uint8_t* out_data = out.mutable_data_ptr<uint8_t>();
    quantize<uint8_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Char) {
    TIME_DECL(quantize_asym8s);
    TIME_START(quantize_asym8s);
    
    int8_t* out_data = out.mutable_data_ptr<int8_t>();
    
    // Hardware-optimized int8 quantization with DMA support
    bool ping_pong_process = false;
    bool ping_process_pong = false;
    size_t chunk_size = 0;

    float32_t* inp_buff[2];
    int8_t* out_buff[2];

    // Check if DRAM buffers are available
    bool dram0_available = (ptr_dram0 != nullptr) && (DRAM0_BUFF_SIZE > 0);
    bool dram1_available = (ptr_dram1 != nullptr) && (DRAM1_BUFF_SIZE > 0);
    
    // DMA has overhead - only beneficial for larger tensors
    // Threshold: 1024 elements (~4KB for float32, ~1KB for int8)
    const size_t DMA_THRESHOLD = 1024;
    bool use_dma = (numel >= DMA_THRESHOLD);

    // Strategy 1: Try ping-pong processing (2 input + 2 output buffers)
    // Using 80/20 split: 80% for input, 20% for output in each DRAM
    if (use_dma && dram0_available && dram1_available && (numel >= 2)) {
      size_t inp_per_buffer = (DRAM0_BUFF_SIZE * 4) / (5 * FLT32_SIZE);  // 80% for float32 input
      size_t out_per_buffer_dram0 = (DRAM0_BUFF_SIZE * 1) / 5;  // 20% for int8 output
      size_t out_per_buffer_dram1 = (DRAM1_BUFF_SIZE * 1) / 5;  // 20% for int8 output

      // Check if 80/20 split fits in both DRAMs
      if ((inp_per_buffer > 0) && 
          (out_per_buffer_dram0 >= inp_per_buffer) &&
          (out_per_buffer_dram1 >= inp_per_buffer) &&
          ((DRAM0_BUFF_SIZE * 4) / 5 + DRAM0_BUFF_SIZE / 5 <= DRAM0_BUFF_SIZE) &&
          ((DRAM1_BUFF_SIZE * 4) / 5 + DRAM1_BUFF_SIZE / 5 <= DRAM1_BUFF_SIZE)) {
        
        // Allocate buffers with 80/20 split
        inp_buff[0] = (float32_t*)ptr_dram0;
        out_buff[0] = (int8_t*)((uint8_t*)ptr_dram0 + (DRAM0_BUFF_SIZE * 4) / 5);
        
        inp_buff[1] = (float32_t*)ptr_dram1;
        out_buff[1] = (int8_t*)((uint8_t*)ptr_dram1 + (DRAM1_BUFF_SIZE * 4) / 5);
        
        chunk_size = inp_per_buffer;
        ping_pong_process = true;
      }
    }
    
    // Strategy 2: Fallback to ping-process-pong (1 input + 1 output buffer)
    // Use full DRAM0 for input, full DRAM1 for output (no split needed)
    if (use_dma && !ping_pong_process && dram0_available && dram1_available) {
      size_t inp_capacity = DRAM0_BUFF_SIZE / FLT32_SIZE;  // Full DRAM0 for input
      size_t out_capacity = DRAM1_BUFF_SIZE;  // Full DRAM1 for output
      
      if ((inp_capacity > 0) && (out_capacity >= inp_capacity)) {
        inp_buff[0] = (float32_t*)ptr_dram0;
        out_buff[0] = (int8_t*)ptr_dram1;
        
        chunk_size = (inp_capacity < out_capacity) ? inp_capacity : out_capacity;
        ping_process_pong = true;
      }
    }

    if (ping_pong_process || ping_process_pong) {
      const float32_t* ptr_inp = (float32_t*)input_data;

      /* Initialize DMA Channel 0 */
      idma_init(0, 0, MAX_BLOCK_16, 8, TICK_CYCLES_1, 0, NULL);
      idma_init_loop(0, buffer_idma_ch_2d, IDMA_2D_DESC, 1, NULL, NULL);

      if (ping_pong_process) {
        // Ping-pong processing for better throughput
        size_t num_chunks = (numel + chunk_size - 1) / chunk_size;
        
        if (num_chunks == 0) num_chunks = 1;

        int32_t pp_swap = 0;
        int32_t idx_in, idx_out = 0;

        float32_t* ptr_in = (float32_t*)ptr_inp;
        int8_t* ptr_out = out_data;

        // Load first chunk
        size_t current_chunk = (numel < chunk_size) ? numel : chunk_size;
        idx_in = idma_copy_2d_desc(0, inp_buff[pp_swap], ptr_in, FLT32_SIZE * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
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
          idx_in = idma_copy_2d_desc(0, inp_buff[pp_swap], ptr_in, FLT32_SIZE * next_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
          pp_swap = pp_swap ^ 1;

          // Wait for load to complete
          idma_desc_done(0, idx_in);

          // Process (pp_swap now points to the buffer that was just loaded)
          quantize_f32_asym8s(out_buff[pp_swap], inp_buff[pp_swap], (float)scale, (int)zero_point, (int)current_chunk);

          // Store result
          idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[pp_swap], sizeof(int8_t) * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);

          ptr_in += next_chunk;
          ptr_out += current_chunk;
          remaining -= next_chunk;
          current_chunk = next_chunk;
        }

        pp_swap = pp_swap ^ 1;

        // Process last chunk
        idma_desc_done(0, idx_in);
        quantize_f32_asym8s(out_buff[pp_swap], inp_buff[pp_swap], (float)scale, (int)zero_point, (int)current_chunk);

        idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[pp_swap], sizeof(int8_t) * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
        idma_desc_done(0, idx_out);
        
        TIME_END(quantize_asym8s);
        TIME_DISPLAY(quantize_asym8s, numel, "elements (DMA ping-pong)");
      } 
      else if (ping_process_pong) {
        // Simple sequential processing
        size_t remaining = numel;
        float32_t* ptr_in = (float32_t*)ptr_inp;
        int8_t* ptr_out = out_data;

        while (remaining > 0) {
          size_t current_chunk = (remaining < chunk_size) ? remaining : chunk_size;

          // Load chunk
          int32_t idx_in = idma_copy_2d_desc(0, inp_buff[0], ptr_in, FLT32_SIZE * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
          idma_desc_done(0, idx_in);

          // Process
          quantize_f32_asym8s(out_buff[0], inp_buff[0], (float)scale, (int)zero_point, (int)current_chunk);

          // Store
          int32_t idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[0], sizeof(int8_t) * current_chunk, DESC_IDMA_PRIOR_H, 1, 0, 0);
          idma_desc_done(0, idx_out);

          ptr_in += current_chunk;
          ptr_out += current_chunk;
          remaining -= current_chunk;
        }
        
        TIME_END(quantize_asym8s);
        TIME_DISPLAY(quantize_asym8s, numel, "elements (DMA ping-process-pong)");
      }
      
      // TIME_END and TIME_DISPLAY now called inside each branch
    } else {
      // No DMA: use hardware function on full tensor at once
      quantize_f32_asym8s(out_data, input_data, (float)scale, (int)zero_point, (int)numel);
      TIME_END(quantize_asym8s);
      TIME_DISPLAY(quantize_asym8s, numel, "elements (HW-optimized, no DMA)");
    }

  } else if (
      out.scalar_type() == ScalarType::Bits16 ||
      out.scalar_type() == ScalarType::UInt16) {
    uint16_t* out_data = out.mutable_data_ptr<uint16_t>();
    quantize<uint16_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Short) {
    int16_t* out_data = out.mutable_data_ptr<int16_t>();
    quantize<int16_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Int) {
    int32_t* out_data = out.mutable_data_ptr<int32_t>();
    quantize<int32_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(out.scalar_type()));
  }
  return out;
}

// int8 quantization - uses generic template
Tensor& quantize_per_tensor_asym8s_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  int8_t* out_data = out.mutable_data_ptr<int8_t>();
  quantize<int8_t>(out_data, input_data, 1. / scale, zero_point, numel);
  return out;
}

// uint8 quantization - uses generic template
Tensor& quantize_per_tensor_asym8u_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  uint8_t* out_data = out.mutable_data_ptr<uint8_t>();
  quantize<uint8_t>(out_data, input_data, 1. / scale, zero_point, numel);
  return out;
}

// int16 quantization - uses generic template
Tensor& quantize_per_tensor_asym16s_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  int16_t* out_data = out.mutable_data_ptr<int16_t>();
  quantize<int16_t>(out_data, input_data, 1. / scale, zero_point, numel);
  return out;
}

// uint16 quantization - uses generic template
Tensor& quantize_per_tensor_asym16u_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  uint16_t* out_data = out.mutable_data_ptr<uint16_t>();
  quantize<uint16_t>(out_data, input_data, 1. / scale, zero_point, numel);
  return out;
}

// int32 quantization - uses generic template
Tensor& quantize_per_tensor_asym32s_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  int32_t* out_data = out.mutable_data_ptr<int32_t>();
  quantize<int32_t>(out_data, input_data, 1. / scale, zero_point, numel);
  return out;
}

} // namespace native
} // namespace vision
} // namespace impl
