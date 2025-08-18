/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace impl {

using Tensor = executorch::aten::Tensor;

Tensor& stack_out(
    KernelRuntimeContext& ctx,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  (void)ctx;

  if (dim < 0) {
    dim += out.dim();
  }

  ET_KERNEL_CHECK(
      ctx, check_stack_args(tensors, dim, out), InvalidArgument, out);

  for (size_t i = 0; i < tensors.size(); ++i) {
    ET_KERNEL_CHECK(
        ctx,
        tensors_have_same_dim_order(tensors[i], out),
        InvalidArgument,
        out);
  }

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(out), InvalidArgument, out);

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_stack_out_target_size(tensors, dim, expected_out_size, &expected_out_dim);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  const size_t outer = getLeadingDims(out, dim);
  const size_t inner = getTrailingDims(out, dim);
  const size_t ninputs = tensors.size();

  const auto out_type = out.scalar_type();
  ET_SWITCH_REALHBBF16_TYPES(out_type, ctx, "stack.out", CTYPE_OUT, [&] {
    CTYPE_OUT* out_ptr = out.mutable_data_ptr<CTYPE_OUT>();
    for (size_t i = 0; i < outer; ++i) {
      for (size_t j = 0; j < ninputs; ++j) {
        const auto in_type = tensors[j].scalar_type();
        ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, "stack.out", CTYPE_IN, [&] {
          const CTYPE_IN* const in_ptr =
              tensors[j].const_data_ptr<CTYPE_IN>() + i * inner;

          for (size_t k = 0; k < inner; ++k) {
            out_ptr[k] = static_cast<CTYPE_OUT>(in_ptr[k]);
          }
          out_ptr += inner;
        });
      }
    }
  });

  return out;
}

} // namespace impl

Tensor& stack_out(
    KernelRuntimeContext& ctx,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  return impl::stack_out(ctx, tensors, dim, out);
}

namespace utils {

Tensor& stack_out(
    KernelRuntimeContext& ctx,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  return impl::stack_out(ctx, tensors, dim, out);
}

std::tuple<
    Error,
    std::array<executorch::aten::SizesType, kTensorDimensionLimit>,
    size_t>
stack_out_shape(executorch::aten::ArrayRef<Tensor> tensors, int64_t dim) {
  std::array<executorch::aten::SizesType, kTensorDimensionLimit> out_sizes{};
  size_t out_dim = 0;

  // Check if tensors array is empty
  if (tensors.size() == 0) {
    return std::make_tuple(Error::InvalidArgument, out_sizes, out_dim);
  }

  // Normalize negative dimension
  int64_t normalized_dim = dim;
  if (normalized_dim < 0) {
    normalized_dim += tensors[0].dim() + 1;
  }

  // Check if dimension is valid
  if (normalized_dim < 0 || normalized_dim > tensors[0].dim()) {
    return std::make_tuple(Error::InvalidArgument, out_sizes, out_dim);
  }

  // Check that all tensors have the same shape
  for (size_t i = 1; i < tensors.size(); ++i) {
    if (tensors[i].dim() != tensors[0].dim()) {
      return std::make_tuple(Error::InvalidArgument, out_sizes, out_dim);
    }
    for (const auto d : c10::irange(tensors[0].dim())) {
      if (tensors[i].size(d) != tensors[0].size(d)) {
        return std::make_tuple(Error::InvalidArgument, out_sizes, out_dim);
      }
    }
  }

  // Compute output shape using the existing utility
  ::torch::executor::get_stack_out_target_size(
      tensors, normalized_dim, out_sizes.data(), &out_dim);

  return std::make_tuple(Error::Ok, out_sizes, out_dim);
}

} // namespace utils
} // namespace native
} // namespace executor
} // namespace torch
