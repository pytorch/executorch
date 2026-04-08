/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <lib.h>
#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::can_cast;
using executorch::runtime::KernelRuntimeContext;
using torch::executor::Error;

namespace impl {
namespace vision {
namespace native {

// Forward declaration of hardware-optimized vector addition function
extern "C" void rvaddf(
    float32_t* restrict z,
    const float32_t* restrict x,
    const float32_t* restrict y,
    int N);

Tensor& add_out(
    KernelRuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  
  // Check if we can use optimized path: same shape, float32, alpha=1.0
  bool same_shape = executorch::runtime::tensors_have_same_shape(a, b) &&
                    executorch::runtime::tensors_have_same_shape(a, out);
  bool is_float = (a.scalar_type() == ScalarType::Float) &&
                  (b.scalar_type() == ScalarType::Float) &&
                  (out.scalar_type() == ScalarType::Float);
  
  // Extract alpha value to check if it's 1.0
  float alpha_val = 1.0f;
  bool alpha_is_one = false;
  if (is_float && torch::executor::native::utils::extract_scalar(alpha, &alpha_val)) {
    alpha_is_one = (alpha_val == 1.0f);
  }

  size_t numel = out.numel();
  
  // Use optimized path if: float32, same shape, alpha=1.0, sufficient size, aligned
  // Require numel to be even (2 floats = 8 bytes) for 8-byte aligned DMA
  bool use_optimized = same_shape && is_float && alpha_is_one &&
                       (numel >= 8) && ((numel % 2) == 0);

  if (use_optimized) {
    TIME_DECL(add_float);
    TIME_START(add_float);

    const float* a_data = a.const_data_ptr<float>();
    const float* b_data = b.const_data_ptr<float>();
    float* out_data = out.mutable_data_ptr<float>();

    // Check if source data is 8-byte aligned (required for DMA)
    bool src_aligned = (((uintptr_t)a_data & 0x7) == 0) &&
                       (((uintptr_t)b_data & 0x7) == 0) &&
                       (((uintptr_t)out_data & 0x7) == 0);

    // DMA setup for two inputs + one output
    bool ping_pong_process = false;
    bool ping_process_pong = false;
    size_t chunk_size = 0;

    float32_t* inp_a_buff[2];
    float32_t* inp_b_buff[2];
    float32_t* out_buff[2];

    // Check if DRAM buffers are available
    bool dram0_available = (ptr_dram0 != nullptr) && (IDMA_BUFFER_SIZE_DRAM0 > 0);
    bool dram1_available = (ptr_dram1 != nullptr) && (IDMA_BUFFER_SIZE_DRAM1 > 0);
    
    // DMA threshold - beneficial for larger tensors
    const size_t DMA_THRESHOLD = 1024;
    bool use_dma = (numel >= DMA_THRESHOLD) && src_aligned;
    
    // Strategy 1: Ping-pong processing (2 sets of buffers)
    // Need to fit: 2 inputs + 1 output per buffer (3 float32 arrays total)
    // Split: 33% input_a, 33% input_b, 33% output per DRAM
    if (use_dma && dram0_available && dram1_available && (numel >= 2)) {
      // Try 128-byte alignment first (optimal for rvaddf SIMD)
      size_t per_array_128 = (IDMA_BUFFER_SIZE_DRAM0 / 3) & ~0x7F;  // 128-byte alignment
      size_t chunk_elements_128 = per_array_128 / FLT32_SIZE;
      
      // If 128-byte alignment gives us 0 chunks, try 8-byte alignment (minimum for float32)
      size_t per_array = per_array_128;
      size_t chunk_elements = chunk_elements_128;
      
      if (chunk_elements == 0) {
        per_array = (IDMA_BUFFER_SIZE_DRAM0 / 3) & ~0x7;  // Fallback to 8-byte alignment
        chunk_elements = per_array / FLT32_SIZE;
      }
      
      if (chunk_elements == 0) {
        // Verify all buffers are 8-byte aligned
        if (((uintptr_t)ptr_dram0 & 0x7) != 0 || ((uintptr_t)ptr_dram1 & 0x7) != 0) {
          // Buffer base addresses not aligned, fall back to non-DMA
          use_dma = false;
        }
      } else {
        // DRAM0: input_a[0] | input_b[0] | output[0] (all 128-byte aligned)
        inp_a_buff[0] = (float32_t*)ptr_dram0;
        inp_b_buff[0] = (float32_t*)((uint8_t*)ptr_dram0 + per_array);
        out_buff[0] = (float32_t*)((uint8_t*)ptr_dram0 + 2 * per_array);
        
        // DRAM1: input_a[1] | input_b[1] | output[1] (all 8-byte aligned)
        inp_a_buff[1] = (float32_t*)ptr_dram1;
        inp_b_buff[1] = (float32_t*)((uint8_t*)ptr_dram1 + per_array);
        out_buff[1] = (float32_t*)((uint8_t*)ptr_dram1 + 2 * per_array);
        
        chunk_size = chunk_elements;
        ping_pong_process = true;
      }
    }
    
    // Strategy 2: Ping-process-pong (1 set of buffers)
    // Use DRAM0 entirely for inputs (50% a, 50% b), DRAM1 for output
    if (use_dma && !ping_pong_process && dram0_available && dram1_available) {
      size_t inp_per_array = (IDMA_BUFFER_SIZE_DRAM0 / 2) & ~0x7;  // Round down to 8-byte boundary
      size_t inp_capacity = inp_per_array / FLT32_SIZE;
      size_t out_capacity = IDMA_BUFFER_SIZE_DRAM1 / FLT32_SIZE;
      
      if ((inp_capacity > 0) && (out_capacity >= inp_capacity)) {
        inp_a_buff[0] = (float32_t*)ptr_dram0;
        inp_b_buff[0] = (float32_t*)((uint8_t*)ptr_dram0 + inp_per_array);
        out_buff[0] = (float32_t*)ptr_dram1;
        
        chunk_size = (inp_capacity < out_capacity) ? inp_capacity : out_capacity;
        ping_process_pong = true;
      }
    }

    if (ping_pong_process || ping_process_pong) {
      /* Initialize DMA Channel 0 (loads) and Channel 1 (stores) */
      dma_2dm_init(0);
      dma_2dm_init(1);

      if (ping_pong_process) {
        // Ping-pong processing for better throughput
        size_t num_chunks = (numel + chunk_size - 1) / chunk_size;
        if (num_chunks == 0) num_chunks = 1;

        int32_t pp_swap = 0;

        const float* ptr_a = a_data;
        const float* ptr_b = b_data;
        float* ptr_out = out_data;

        // Load first chunk (both inputs) into buffer 0 via ch0
        size_t current_chunk = (numel < chunk_size) ? numel : chunk_size;

        // Writeback inputs from cache to system memory for DMA coherency
        xthal_dcache_region_writeback((void*)a_data, FLT32_SIZE * numel);
        xthal_dcache_region_writeback((void*)b_data, FLT32_SIZE * numel);

        dma_1dm(0, (void*)ptr_a, inp_a_buff[pp_swap], FLT32_SIZE * current_chunk);
        dma_1dm(0, (void*)ptr_b, inp_b_buff[pp_swap], FLT32_SIZE * current_chunk);

        size_t remaining = numel - current_chunk;
        ptr_a += current_chunk;
        ptr_b += current_chunk;

        // Pipeline: load (ch0) and store (ch1) overlap with processing
        for (size_t i = 0; i < num_chunks - 1; i++) {
          size_t next_chunk = (remaining < chunk_size) ? remaining : chunk_size;

          // Wait for current loads to complete
          idma_hw_wait_all(0);

          // Start loading next chunk into alternate buffer via ch0
          dma_1dm(0, (void*)ptr_a, inp_a_buff[pp_swap ^ 1], FLT32_SIZE * next_chunk);
          dma_1dm(0, (void*)ptr_b, inp_b_buff[pp_swap ^ 1], FLT32_SIZE * next_chunk);

          // Process current buffer (ch0 loads next in parallel)
          rvaddf(out_buff[pp_swap], inp_a_buff[pp_swap], inp_b_buff[pp_swap], (int)current_chunk);

          // Wait for previous store to complete before reusing out_buff
          idma_hw_wait_all(1);

          // Store result via ch1
          dma_1dm(1, out_buff[pp_swap], (void*)ptr_out, FLT32_SIZE * current_chunk);

          ptr_a += next_chunk;
          ptr_b += next_chunk;
          ptr_out += current_chunk;
          remaining -= next_chunk;
          current_chunk = next_chunk;
          pp_swap ^= 1;
        }
        
        // Process last chunk
        idma_hw_wait_all(0);
        rvaddf(out_buff[pp_swap], inp_a_buff[pp_swap], inp_b_buff[pp_swap], (int)current_chunk);

        idma_hw_wait_all(1);
        dma_1dm(1, out_buff[pp_swap], (void*)ptr_out, FLT32_SIZE * current_chunk);
        idma_hw_wait_all(1);
        
        // Invalidate cache for DMA-written output so next operator sees fresh data
        xthal_dcache_region_invalidate(out_data, FLT32_SIZE * numel);
        
        TIME_END(add_float);
        TIME_DISPLAY(add_float, numel, "elements (DMA ping-pong)");
      } 
      else if (ping_process_pong) {
        // Sequential processing
        size_t remaining = numel;
        const float* ptr_a = a_data;
        const float* ptr_b = b_data;
        float* ptr_out = out_data;

        while (remaining > 0) {
          size_t current_chunk = (remaining < chunk_size) ? remaining : chunk_size;

          // Load both input chunks via ch0 (overlaps with any pending ch1 store)
          dma_1dm(0, (void*)ptr_a, inp_a_buff[0], FLT32_SIZE * current_chunk);
          dma_1dm(0, (void*)ptr_b, inp_b_buff[0], FLT32_SIZE * current_chunk);
          // Wait for previous store to complete
          idma_hw_wait_all(1);
          // Wait for loads to complete
          idma_hw_wait_all(0);

          // Process: out = a + b
          rvaddf(out_buff[0], inp_a_buff[0], inp_b_buff[0], (int)current_chunk);

          // Store result via ch1
          dma_1dm(1, out_buff[0], (void*)ptr_out, FLT32_SIZE * current_chunk);

          ptr_a += current_chunk;
          ptr_b += current_chunk;
          ptr_out += current_chunk;
          remaining -= current_chunk;
        }
        idma_hw_wait_all(1);
        
        // Invalidate cache for DMA-written output so next operator sees fresh data
        xthal_dcache_region_invalidate(out_data, FLT32_SIZE * numel);
        
        TIME_END(add_float);
        TIME_DISPLAY(add_float, numel, "elements (DMA ping-process-pong)");
      }
    } else {
      // Fallback: use hardware-optimized vector addition directly without DMA
      rvaddf(out_data, a_data, b_data, (int)numel);

      // Writeback output from cache to system memory for DMA coherency
      xthal_dcache_region_writeback(out_data, FLT32_SIZE * numel);

      TIME_END(add_float);
      TIME_DISPLAY(add_float, numel, "elements (HW-optimized, no DMA)");
    }
  } else {
    // Fallback: Use full generic portable implementation
    // This handles: broadcasting, non-float dtypes, alpha!=1.0, small tensors, all corner cases
    
    TIME_DECL(add_generic);
    TIME_START(add_generic);
    
    namespace utils = torch::executor::native::utils;
    using torch::executor::check_alpha_type;
    using torch::executor::promoteTypes;
    using torch::executor::canCast;
    using torch::executor::resize_to_broadcast_target_size;
    using torch::executor::tensors_have_same_dim_order;
    using torch::executor::Error;
    
    // Common Dtype
    ScalarType common_type = promoteTypes(a.scalar_type(), b.scalar_type());

    // Check Common Dtype
    ET_KERNEL_CHECK(
        ctx,
        (canCast(common_type, out.scalar_type()) &&
         check_alpha_type(utils::get_scalar_dtype(alpha), common_type)),
        InvalidArgument,
        out);

    // Check Dim Order
    ET_KERNEL_CHECK(
        ctx, 
        tensors_have_same_dim_order(a, b, out), 
        InvalidArgument, 
        out);

    // Resize
    ET_KERNEL_CHECK(
        ctx,
        resize_to_broadcast_target_size(a, b, out) == Error::Ok,
        InvalidArgument,
        out);

    // Compute Dtype
    ScalarType compute_type = utils::get_compute_type(common_type);

    // @lint-ignore CLANGTIDY facebook-hte-CArray
    static constexpr const char op_name[] = "add.out";

    ET_SWITCH_REALB_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
      CTYPE_COMPUTE val_alpha;
      ET_KERNEL_CHECK(
          ctx, utils::extract_scalar(alpha, &val_alpha), InvalidArgument, );
      utils::apply_bitensor_elementwise_fn<
          CTYPE_COMPUTE,
          op_name,
          utils::SupportedTensorDtypes::REALHBBF16>(
          [val_alpha](const auto& val_a, const auto& val_b) {
            return val_a + val_alpha * val_b;
          },
          ctx,
          a,
          utils::SupportedTensorDtypes::REALHBBF16,
          b,
          utils::SupportedTensorDtypes::REALHBBF16,
          out);
    });
    
    TIME_END(add_generic);
    TIME_DISPLAY(add_generic, numel, "elements (generic template)");
  }
  return out;
}

} // namespace native
} // namespace vision
} // namespace impl
