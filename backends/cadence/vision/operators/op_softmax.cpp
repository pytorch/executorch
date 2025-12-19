/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <lib.h>
#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using torch::executor::Error;

namespace impl {
namespace vision {
namespace native {

Tensor& _softmax_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      torch::executor::check_softmax_args(in, dim, half_to_float, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, 
      executorch::runtime::tensors_have_same_dim_order(in, out), 
      InvalidArgument, 
      out);

  // Adjust for negative dim
  dim = dim < 0 ? dim + executorch::runtime::nonzero_dim(in) : dim;

  const executorch::aten::optional<int64_t>& dim_t = dim;
  const size_t d = ET_NORMALIZE_IX(dim_t.value(), in.dim());
  const size_t size = in.size(d);

  size_t stride = 1, outer_size = 1;

  size_t outer_stride = 1;

  constexpr auto name = "_softmax.out";
  constexpr int MaxDim = 5;

  bool optimized = true;
  bool ping_pong_process = false;
  bool ping_process_pong = false;

  float32_t *inp_buff[2];
  float32_t *out_buff[2];

  if ((d == in.dim() - 1)){
    if ((4 * FLT32_SIZE * size <= (DRAM0_BUFF_SIZE + DRAM1_BUFF_SIZE)) && (in.dim() != 1)){
      // For ping-pong processing we need to have enough buffer to hold 2 input and 2 output blocks
      if (2 * FLT32_SIZE * size <= DRAM0_BUFF_SIZE &&  2 * FLT32_SIZE * size <= DRAM1_BUFF_SIZE){
        // Both DRAM0 and DRAM1 can hold 2 input and 2 output blocks
        inp_buff[0] = (float32_t *)ptr_dram0;
        inp_buff[1] = (float32_t *)ptr_dram1;
        out_buff[0] = (float32_t *)(ptr_dram0) + size;
        out_buff[1] = (float32_t *)(ptr_dram1) + size;
        ping_pong_process = true;
      }
      else if (4 * FLT32_SIZE * size <= DRAM0_BUFF_SIZE){
        // DRAM0 can hold 2 input and 2 output blocks
        inp_buff[0] = (float32_t *)ptr_dram0;
        inp_buff[1] = (float32_t *)(ptr_dram0) + size; 
        out_buff[0] = (float32_t *)(ptr_dram0) + 2 * size;
        out_buff[1] = (float32_t *)(ptr_dram0) + 3 * size;
        ping_pong_process = true;
      }
      else if (4 * FLT32_SIZE * size <= DRAM1_BUFF_SIZE){
        // DRAM1 can hold 2 input and 2 output blocks
        inp_buff[0] = (float32_t *)ptr_dram1;
        inp_buff[1] = (float32_t *)(ptr_dram1) + size; 
        out_buff[0] = (float32_t *)(ptr_dram1) + 2 * size;
        out_buff[1] = (float32_t *)(ptr_dram1) + 3 * size;
        ping_pong_process = true;
      }
      else if (3 * FLT32_SIZE * size <= DRAM0_BUFF_SIZE && FLT32_SIZE * size <= DRAM1_BUFF_SIZE){
        // DRAM0 can hold 2 output and 1 input blocks, DRAM1 can hold 1 input block
        inp_buff[0] = (float32_t *)ptr_dram0;
        inp_buff[1] = (float32_t *)ptr_dram1;
        out_buff[0] = (float32_t *)(ptr_dram0) + size;
        out_buff[1] = (float32_t *)(ptr_dram0) + 2 * size;
        ping_pong_process = true;
      }
      else if (FLT32_SIZE * size <= DRAM0_BUFF_SIZE && 3 * FLT32_SIZE * size <= DRAM1_BUFF_SIZE){
        // DRAM1 can hold 2 output and 1 input blocks, DRAM0 can hold 1 input block
        inp_buff[0] = (float32_t *)ptr_dram0;
        inp_buff[1] = (float32_t *)ptr_dram1;
        out_buff[0] = (float32_t *)(ptr_dram1) + size;
        out_buff[1] = (float32_t *)(ptr_dram1) + 2 * size;
        ping_pong_process = true;
      }
    }
    else if (2 * FLT32_SIZE * size <= (DRAM0_BUFF_SIZE + DRAM1_BUFF_SIZE)){
      // For ping-process-pong we need to have enough buffer to hold 1 input and 1 output block
      if (FLT32_SIZE * size <= DRAM0_BUFF_SIZE && FLT32_SIZE * size <= DRAM1_BUFF_SIZE){
        // Both DRAM0 and DRAM1 can hold 1 input and 1 output block
        inp_buff[0] = (float32_t *)ptr_dram0;
        out_buff[0] = (float32_t *)ptr_dram1;
        ping_process_pong = true;
      }
      else if (2 * FLT32_SIZE * size <= DRAM0_BUFF_SIZE){
        // DRAM0 can hold 1 input and 1 output block
        inp_buff[0] = (float32_t *)ptr_dram0;
        out_buff[0] = (float32_t *)(ptr_dram0) + size;
        ping_process_pong = true;
      }
      else if (2 * FLT32_SIZE * size <= DRAM1_BUFF_SIZE){
        // DRAM1 can hold 1 input and 1 output block
        inp_buff[0] = (float32_t *)ptr_dram1;
        out_buff[0] = (float32_t *)(ptr_dram1) + size;
        ping_process_pong = true;
      }
    }
  }

  if (out.scalar_type() != ScalarType::Float)
    optimized = false;

  if (in.dim() > MaxDim)
    optimized = false;

  if (optimized){
    const float32_t *ptr_inp = (float32_t *)in.const_data_ptr<float>();
    float32_t *out_data = (float32_t *)out.mutable_data_ptr<float>();

	  /* Channel 0*/
	  idma_init(0, 0, MAX_BLOCK_16, 8, TICK_CYCLES_1, 0, NULL);
	  idma_init_loop(0, buffer_idma_ch_2d, IDMA_2D_DESC, 1, NULL, NULL);

    if (ping_pong_process) {
      for (int i = 0; i < in.dim(); i++){
        if (i != d)
          outer_size *= in.size(i);
      }

      outer_stride = size;
      stride = size;

      int32_t pp_swap = 0;
      int32_t idx_in, idx_out = 0;

	    float32_t *ptr_out = out_data;
	    float32_t *ptr_in = (float32_t *) ptr_inp;

      idx_in = idma_copy_2d_desc(0, inp_buff[pp_swap], ptr_in, 4 * stride, DESC_IDMA_PRIOR_H, 1, 0, 0);
      pp_swap = pp_swap ^ 1;

      for (int i = 0; i < (outer_size - 1); i++){ 
          idma_desc_done(0, idx_out);
          ptr_in += outer_stride;
          idx_in = idma_copy_2d_desc(0, inp_buff[pp_swap], ptr_in, 4 * stride, DESC_IDMA_PRIOR_H, 1, 0, 0);
          pp_swap = pp_swap ^ 1;

          idma_desc_done(0, idx_in);
          /* PROCESS CALL */
          vsoftmaxf(out_buff[pp_swap], inp_buff[pp_swap], stride);

          idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[pp_swap], 4 * stride, DESC_IDMA_PRIOR_H, 1, 0, 0);
          ptr_out += outer_stride;
        }

      pp_swap = pp_swap ^ 1;

      idma_desc_done(0, idx_in);
      /* PROCESS CALL */
      vsoftmaxf(out_buff[pp_swap], inp_buff[pp_swap], stride);

      idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[pp_swap], 4 * stride, DESC_IDMA_PRIOR_H, 1, 0, 0);
      idma_desc_done(0, idx_out);

      return out;
    } else if (ping_process_pong) {
      for (int i = 0; i < in.dim(); i++){
        if (i != d)
          outer_size *= in.size(i);
      }

      outer_stride = size;
      stride = size;

      int32_t idx_in, idx_out = 0;

	    float32_t *ptr_out = out_data;
	    float32_t *ptr_in = (float32_t *) ptr_inp;

	    for (int i = 0; i < outer_size; i++){
        idx_in = idma_copy_2d_desc(0, inp_buff[0], ptr_in, 4 * stride, DESC_IDMA_PRIOR_H, 1, 0, 0);
        idma_desc_done(0, idx_in);

		    vsoftmaxf(out_buff[0], inp_buff[0], stride);

        idx_out = idma_copy_2d_desc(0, ptr_out, out_buff[0], 4 * stride, DESC_IDMA_PRIOR_H, 1, 0, 0);        
        idma_desc_done(0, idx_out);

        ptr_in += outer_stride;
		    ptr_out += outer_stride;
	    }

      return out;
    } else {
      int num_inp_dims = in.dim();
      int num_out_dims = num_inp_dims;

      int ptr_inp_shape[MaxDim];
      int ptr_out_shape[MaxDim];
      int ptr_permute_vec[MaxDim];

      for (int i = 0; i < num_inp_dims; i++)
        ptr_inp_shape[i] = in.size(i);

      for (int i = 0; i < num_inp_dims; i++) {
        if (i == d)
          ptr_permute_vec[i] = num_inp_dims - 1;
        else if (i == (num_inp_dims - 1))
          ptr_permute_vec[num_inp_dims - 1] = d;
        else
          ptr_permute_vec[i] = i;

        ptr_out_shape[i] = ptr_inp_shape[ptr_permute_vec[i]];

        if (i != d)
          outer_size = outer_size * ptr_inp_shape[i];
      }

      outer_stride = size;

      executorch::runtime::Result<void*> temp_mem_res = ctx.allocate_temp(out.numel() * sizeof(float));
      float* ptr_out =
          (float*)(temp_mem_res.ok() ? temp_mem_res.get() : nullptr);

      ET_KERNEL_CHECK(ctx, ptr_out != nullptr, MemoryAllocationFailed, out);

      executorch::runtime::Result<void*> temp_mem_res1 = ctx.allocate_temp(out.numel() * sizeof(float));
      float* ptr_out1 =
          (float*)(temp_mem_res1.ok() ? temp_mem_res1.get() : nullptr);

      ET_KERNEL_CHECK(ctx, ptr_out1 != nullptr, MemoryAllocationFailed, out);

      tensor_transposef(
        ptr_out,
        ptr_out_shape,
        ptr_inp,
        ptr_inp_shape,
        ptr_permute_vec,
        num_out_dims,
        num_inp_dims);

      for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
        size_t outer = outer_idx * outer_stride;
        for (size_t inner_idx = 0; inner_idx < stride; ++inner_idx) {
          size_t base = outer + inner_idx;
        
          float *ptr_in_data = &ptr_out[base];
          float *ptr_out_data = &ptr_out1[base];

          vsoftmaxf(ptr_out_data, ptr_in_data, size);
        }
      }

      tensor_transposef(
        out_data,
        ptr_inp_shape,
        ptr_out1,
        ptr_out_shape,
        ptr_permute_vec,
        num_out_dims,
        num_inp_dims);

      return out;
    }
  }

  ET_SWITCH_FLOATHBF16_TYPES(
      in.scalar_type(), ctx, "_softmax.out", CTYPE, [&]() {
        const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
        CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

        torch::executor::apply_over_dim(
            [in_data, out_data](
                const size_t size, const size_t stride, const size_t base) {
              // calculate max in softmax dim. During softmax computation each
              // value is subtracted by the maximum in value before calling exp
              // to preserve numerical stability.
              const CTYPE max_in = torch::executor::apply_unary_reduce_fn(
                  [](const CTYPE val_in, CTYPE val_accum) {
                    return std::max(val_in, val_accum);
                  },
                  in_data + base,
                  size,
                  stride);

              const CTYPE temp_sum = 
                  torch::executor::apply_unary_map_reduce_fn<CTYPE, CTYPE>(
                      [max_in](const CTYPE val_in) {
                      return std::exp(val_in - max_in);
                      },
                      [](const CTYPE mapped_in, CTYPE val_accum) {
                      return val_accum + mapped_in;
                      },
                      in_data + base,
                      size,
                      stride);

              torch::executor::apply_unary_map_fn(
                  [max_in, temp_sum](const CTYPE val_in) {
                    return std::exp(val_in - max_in) / temp_sum;
                  },
                  in_data + base,
                  out_data + base,
                  size,
                  stride);
            },
            in,
            dim);
      });

  return out;
}

} // namespace native
} // namespace vision
} // namespace impl
