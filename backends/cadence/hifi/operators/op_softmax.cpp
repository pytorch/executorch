/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using torch::executor::Error;

namespace cadence {
namespace impl {
namespace HiFi {
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

  const std::optional<int64_t>& dim_t = dim;
  const size_t d = ET_NORMALIZE_IX(dim_t.value(), in.dim());
  const size_t size = in.size(d);

  size_t stride = 1, outer_size = 1;

  size_t outer_stride = 1;

  constexpr auto name = "_softmax.out";
  constexpr int kNnlibMaxDim = 16;

  bool optimized = true;

  if (out.scalar_type() != ScalarType::Float)
    optimized = false;

  if (in.dim() > kNnlibMaxDim)
    optimized = false;

  if (optimized) {
    int* p_inp = (int*)in.const_data_ptr<float>();
    int* out_data = (int*)out.mutable_data_ptr<float>();
    int num_inp_dims = in.dim();
    int num_out_dims = num_inp_dims;

    int p_inp_shape[kNnlibMaxDim];
    int p_out_shape[kNnlibMaxDim];
    int p_permute_vec[kNnlibMaxDim];

    for (int i = 0; i < num_inp_dims; i++)
      p_inp_shape[i] = in.size(i);

    for (int i = 0; i < num_inp_dims; i++) {
      if (i == d)
        p_permute_vec[i] = num_inp_dims - 1;
      else if (i == (num_inp_dims - 1))
        p_permute_vec[num_inp_dims - 1] = d;
      else
        p_permute_vec[i] = i;

      p_out_shape[i] = p_inp_shape[p_permute_vec[i]];

      if (i != d)
        outer_size = outer_size * p_inp_shape[i];
    }

    outer_stride = size;

    WORD32 ret_val = 0;

    // Check if the input is permuted. If not, then we don't need to transpose
    bool is_permuted = false;
    for (int i = 0; i < num_inp_dims; i++) {
      if (p_permute_vec[i] != i) {
        is_permuted = true;
        break;
      }
    }

    if (!is_permuted) {
      const float* p_inpf = in.const_data_ptr<float>();
      float* out_dataf = out.mutable_data_ptr<float>();

      for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
        size_t outer = outer_idx * outer_stride;
        for (size_t inner_idx = 0; inner_idx < stride; ++inner_idx) {
          size_t base = outer + inner_idx;

          float* p_in_data = (float*)&p_inpf[base];
          float* p_out_data = (float*)&out_dataf[base];

          ret_val = xa_nn_vec_softmax_f32_f32(p_out_data, p_in_data, size);

          ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
        }
      }
      return out;
    }

    int* p_out =
        (int*)kernels::allocate_temp_memory(ctx, out.numel() * sizeof(int));

    ET_KERNEL_CHECK(ctx, p_out != nullptr, MemoryAllocationFailed, out);

    int* p_out1 =
        (int*)kernels::allocate_temp_memory(ctx, out.numel() * sizeof(int));

    ET_KERNEL_CHECK(ctx, p_out1 != nullptr, MemoryAllocationFailed, out);

    ret_val = xa_nn_transpose_32_32(
        p_out,
        p_out_shape,
        p_inp,
        p_inp_shape,
        p_permute_vec,
        num_out_dims,
        num_inp_dims);

    ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);

    for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
      size_t outer = outer_idx * outer_stride;
      for (size_t inner_idx = 0; inner_idx < stride; ++inner_idx) {
        size_t base = outer + inner_idx;

        float* p_in_data = (float*)&p_out[base];
        float* p_out_data = (float*)&p_out1[base];

        ret_val = xa_nn_vec_softmax_f32_f32(p_out_data, p_in_data, size);

        ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
      }
    }

    ret_val = xa_nn_transpose_32_32(
        out_data,
        p_inp_shape,
        p_out1,
        p_out_shape,
        p_permute_vec,
        num_out_dims,
        num_inp_dims);
    ET_KERNEL_CHECK(ctx, ret_val == 0, Internal, out);
    return out;
  }

  ET_SWITCH_FLOATH_TYPES(in.scalar_type(), ctx, name, CTYPE, [&]() {
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

Tensor& softmax_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  return _softmax_out(ctx, in, dim, half_to_float, out);
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
