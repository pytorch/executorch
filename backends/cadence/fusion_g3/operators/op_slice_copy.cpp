/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/slice_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <xa_nnlib_kernels_api.h>
#include <cstring>

using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::Error;
using torch::executor::KernelRuntimeContext;

/* ScalarType in Executorch do not have support for below data types.
 * So, creating a placeholder for these data types. Once, ScalarTypes is
 * updated to have support for below data types, these can be removed and
 * operator need to be updated accordingly
 */
enum datatype {
  Ushort = 20,
  Uint = 23,
};

namespace cadence {
namespace impl {
namespace G3 {
namespace native {

#define XT_KERNEL_CHECK(ctx, out, kernel, ...) \
  const auto ret = kernel(__VA_ARGS__);        \
  ET_KERNEL_CHECK_MSG(                         \
      ctx,                                     \
      ret == 0,                                \
      InvalidArgument,                         \
      out,                                     \
      "Failed to run kernel: " #kernel "(" #__VA_ARGS__ ")");

Tensor& slice_copy_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    exec_aten::optional<int64_t> start_val,
    exec_aten::optional<int64_t> end_val,
    int64_t step,
    Tensor& out) {
  (void)ctx;

  if (dim < 0) {
    dim += in.dim();
  }
  // If user do not set value to end_val, set end to in.size(dim) (largest
  // value available)
  int64_t end = end_val.has_value() ? end_val.value() : in.size(dim);
  // If user do not set value to start_val, set start to 0 (smallest value
  // available)
  int64_t start = start_val.has_value() ? start_val.value() : 0;
  int64_t length =
      torch::executor::adjust_slice_indices(in.size(dim), &start, &end, step);

  int kTensorDimensionLimit = executorch::runtime::kTensorDimensionLimit;

#ifdef OP_ARG_CHECK
  ET_KERNEL_CHECK(
      ctx,
      torch::executor::check_slice_copy_args(in, dim, step, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(in, out),
      InvalidArgument,
      out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  Tensor::SizesType target_sizes[kTensorDimensionLimit];
  size_t target_ndim = 0;
  torch::executor::get_slice_copy_out_target_size(
      in, dim, length, target_sizes, &target_ndim);
  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::resize_tensor(out, {target_sizes, target_ndim}) ==
          Error::Ok,
      InvalidArgument,
      out);
#endif

  const exec_aten::ArrayRef<Tensor::SizesType> in_size = in.sizes();
  const exec_aten::ArrayRef<Tensor::SizesType> out_size = out.sizes();

  int inp_shape[kTensorDimensionLimit];
  int out_shape[kTensorDimensionLimit];

  /* input shapes and output shapes */
  for (auto i = 0; i < in_size.size(); i++) {
    inp_shape[i] = in_size[i];
  }

  for (auto i = 0; i < out_size.size(); i++) {
    out_shape[i] = out_size[i];
  }

  signed char* out_data = out.mutable_data_ptr<signed char>();
  const signed char* const inp_data = in.const_data_ptr<signed char>();

  if (out.scalar_type() == ScalarType::Int) {
    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_slice,
        out_data,
        out_shape,
        inp_data,
        inp_shape,
        in.dim(),
        (int)start,
        (int)(end - 1),
        (int)step,
        (int)dim,
        sizeof(int));
  } else if (out.scalar_type() == ScalarType::Short) {
    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_slice,
        out_data,
        out_shape,
        inp_data,
        inp_shape,
        in.dim(),
        (int)start,
        (int)(end - 1),
        (int)step,
        (int)dim,
        sizeof(short));
  } else if (out.scalar_type() == ScalarType::Char) {
    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_slice,
        out_data,
        out_shape,
        inp_data,
        inp_shape,
        in.dim(),
        (int)start,
        (int)(end - 1),
        (int)step,
        (int)dim,
        sizeof(char));

  } else if (out.scalar_type() == (ScalarType)Uint) {
    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_slice,
        out_data,
        out_shape,
        inp_data,
        inp_shape,
        in.dim(),
        (int)start,
        (int)(end - 1),
        (int)step,
        (int)dim,
        sizeof(int));
  } else if (out.scalar_type() == (ScalarType)Ushort) {
    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_slice,
        out_data,
        out_shape,
        inp_data,
        inp_shape,
        in.dim(),
        (int)start,
        (int)(end - 1),
        (int)step,
        (int)dim,
        sizeof(short));
  } else if (out.scalar_type() == ScalarType::Byte) {
    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_slice,
        out_data,
        out_shape,
        inp_data,
        inp_shape,
        in.dim(),
        (int)start,
        (int)(end - 1),
        (int)step,
        (int)dim,
        sizeof(char));
  } else {
    torch::executor::compute_slice(in, dim, start, length, step, out);
  }

  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
} // namespace cadence