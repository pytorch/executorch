/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& squeeze_copy_dim_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_squeeze_copy_dim_args(in, dim, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  if (dim < 0) {
    dim += nonzero_dim(in);
  }

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_squeeze_copy_dim_out_target_size(
      in, dim, expected_out_size, &expected_out_dim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  if (in.nbytes() > 0) {
    // Note that this check is important. It's valid for a tensor with numel 0
    // to have a null data pointer, but in some environments it's invalid to
    // pass a null pointer to memcpy() even when the size is zero.
    memcpy(out.mutable_data_ptr(), in.const_data_ptr(), in.nbytes());
  }
  return out;
}

Tensor& squeeze_copy_dims_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    exec_aten::ArrayRef<int64_t> dims,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_squeeze_copy_dims_args(in, dims, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_squeeze_copy_dims_out_target_size(
      in, dims, expected_out_size, &expected_out_dim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  if (in.nbytes() > 0) {
    // Note that this check is important. It's valid for a tensor with numel 0
    // to have a null data pointer, but in some environments it's invalid to
    // pass a null pointer to memcpy() even when the size is zero.
    memcpy(out.mutable_data_ptr(), in.const_data_ptr(), in.nbytes());
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
