/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <api.h>
#include <executorch/backends/cadence/vision/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <idma_init.h>
#include <stdio.h>

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

  if ((d == in.dim() - 1)) {
    if (size <= IDMA_BUFF_SIZE / 4 && in.dim() != 1) {
      ping_pong_process = true;
    } else if (size <= IDMA_BUFF_SIZE / 2) {
      ping_process_pong = true;
    }
  }

  if (out.scalar_type() != ScalarType::Float)
    optimized = false;

  if (in.dim() > MaxDim)
    optimized = false;

  if (optimized) {
    const float* ptr_inp = (float*)in.const_data_ptr<float>();
    float* out_data = (float*)out.mutable_data_ptr<float>();

    /* Channel 0*/
    idma_init(0, 0, MAX_BLOCK_16, 8, TICK_CYCLES_1, 0, NULL);
    idma_init_loop(0, descbuf[0], IDMA_2D_DESC, 1, NULL, NULL);

    /* Channel 1*/
    idma_init(1, 0, MAX_BLOCK_16, 8, TICK_CYCLES_1, 0, NULL);
    idma_init_loop(1, descbuf[1], IDMA_2D_DESC, 1, NULL, NULL);

    if (ping_pong_process) {
      for (int i = 0; i < in.dim(); i++) {
        if (i != d)
          outer_size *= in.size(i);
      }

      outer_stride = size;
      stride = size;

      int pp_swap = 0;

      float32_t* ptr_out = out_data;
      float32_t* ptr_in = (float32_t*)ptr_inp;

      idma_copy_2d_desc(
          0, inpData[pp_swap], ptr_in, 4 * stride, DESC_IDMA_PRIOR_H, 1, 0, 0);
      pp_swap = 1;

      for (int i = 0; i < (outer_size - 1); i++) {
        IDMA_HW_WAIT_ALL(0);
        ptr_in += outer_stride;
        idma_copy_2d_desc(
            0,
            inpData[pp_swap],
            ptr_in,
            4 * stride,
            DESC_IDMA_PRIOR_H,
            1,
            0,
            0);
        pp_swap = pp_swap ^ 1;

        /* PROCESS CALL */
        vsoftmaxf(outData[pp_swap], inpData[pp_swap], stride);

        IDMA_HW_WAIT_ALL(1);
        idma_copy_2d_desc(
            1,
            ptr_out,
            outData[pp_swap],
            4 * stride,
            DESC_IDMA_PRIOR_H,
            1,
            0,
            0);
        ptr_out += outer_stride;
      }

      IDMA_HW_WAIT_ALL(0);
      pp_swap = pp_swap ^ 1;

      /* PROCESS CALL */
      vsoftmaxf(outData[pp_swap], inpData[pp_swap], stride);

      IDMA_HW_WAIT_ALL(1);
      idma_copy_2d_desc(
          1, ptr_out, outData[pp_swap], 4 * stride, DESC_IDMA_PRIOR_H, 1, 0, 0);

      IDMA_HW_WAIT_ALL(1);

      return out;
    } else if (ping_process_pong) {
      for (int i = 0; i < in.dim(); i++) {
        if (i != d)
          outer_size *= in.size(i);
      }

      outer_stride = size;
      stride = size;

      float32_t* ptr_out = out_data;
      float32_t* ptr_in = (float32_t*)ptr_inp;

      for (int i = 0; i < outer_size; i++) {
        idma_copy_2d_desc(
            0, data_dram0, ptr_in, 4 * stride, DESC_IDMA_PRIOR_H, 1, 0, 0);
        IDMA_HW_WAIT_ALL(0);

        vsoftmaxf(data_dram1, data_dram0, stride);

        idma_copy_2d_desc(
            1, ptr_out, data_dram1, 4 * stride, DESC_IDMA_PRIOR_H, 1, 0, 0);
        IDMA_HW_WAIT_ALL(1);

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

      float* ptr_out = (float*)kernels::allocate_temp_memory(
          ctx, out.numel() * sizeof(float));

      ET_KERNEL_CHECK(ctx, ptr_out != nullptr, MemoryAllocationFailed, out);

      float* ptr_out1 = (float*)kernels::allocate_temp_memory(
          ctx, out.numel() * sizeof(float));

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

          float* ptr_in_data = &ptr_out[base];
          float* ptr_out_data = &ptr_out1[base];

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
