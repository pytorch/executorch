/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

#include <executorch/backends/cadence/hifi/kernels/kernels.h>

using executorch::aten::RuntimeContext;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::ArrayRef;
using torch::executor::Error;
using torch::executor::optional;

namespace impl {
namespace HiFi {
namespace native {

int prepare_data(
    const Tensor& in,
    Tensor& out,
    optional<ArrayRef<int64_t>> dim_list,
    int* inp_shape,
    int* out_shape,
    int* p_axis,
    int num_inp_dims,
    int num_out_dims) {
  for (int i = 0; i < num_inp_dims; i++) {
    inp_shape[i] = in.size(i);
  }

  for (int i = 0; i < num_out_dims; i++) {
    out_shape[i] = out.size(i);
  }

  int num_axis_dims = 0;
  for (const auto& d : dim_list.value()) {
    if (d < 0) {
      p_axis[num_axis_dims] = num_inp_dims + d;
      num_axis_dims++;
    } else {
      p_axis[num_axis_dims] = d;
      num_axis_dims++;
    }
  }

  return num_axis_dims;
}

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
  constexpr int kNnlibMaxDim = 4;

  bool optimized = 1;

  if (out.scalar_type() != ScalarType::Float)
    optimized = 0;

  if (in.dim() > kNnlibMaxDim)
    optimized = 0;

  if (optimized) {
    float* __restrict__ p_out = out.mutable_data_ptr<float>();
    const float* __restrict__ p_inp =
        (const float* __restrict__)in.const_data_ptr<float>();

    int num_elm = in.numel();

    int num_inp_dims = in.dim();
    int num_out_dims = out.dim();

    int inp_shape[kNnlibMaxDim];
    int out_shape[kNnlibMaxDim];
    int p_axis[kNnlibMaxDim];

    for (int i = 0; i < kNnlibMaxDim; i++) {
      out_shape[i] = 1;
      inp_shape[i] = 1;
      p_axis[i] = 1;
    }

    int num_axis_dims = prepare_data(
        in,
        out,
        dim_list,
        inp_shape,
        out_shape,
        p_axis,
        num_inp_dims,
        num_out_dims);

    if (num_axis_dims == num_inp_dims) {
      num_out_dims = 1;
      out_shape[0] = 1;
    }

    int scratch_size = xa_nn_reduce_getsize_nhwc(
        -3, inp_shape, num_inp_dims, p_axis, num_axis_dims, 1);

    void* __restrict__ p_scratch_in =
        (void* __restrict__)kernels::allocate_temp_memory(
            ctx, scratch_size * sizeof(int));

    ET_KERNEL_CHECK(ctx, p_scratch_in != nullptr, MemoryAllocationFailed, out);

    xa_nn_reduce_mean_4D_f32_f32(
        p_out,
        out_shape,
        p_inp,
        inp_shape,
        p_axis,
        num_out_dims,
        num_inp_dims,
        num_axis_dims,
        p_scratch_in);

    return out;
  }

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
} // namespace HiFi
} // namespace impl
