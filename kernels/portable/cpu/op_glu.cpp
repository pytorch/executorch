/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cinttypes>
#include <cmath>
#include <cstdint>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

namespace {

struct SplitGLUInputTensor {
  explicit SplitGLUInputTensor(const Tensor& self, int64_t dim);
  using SizesArray =
      std::array<executorch::aten::SizesType, kTensorDimensionLimit>;
  SizesArray half_sizes;
  TensorImpl first_half_impl;
  TensorImpl second_half_impl;
  Tensor first_half;
  Tensor second_half;

 private:
  static SizesArray get_half_sizes(const Tensor& self, int64_t dim) {
    SizesArray half_sizes;
    std::copy(self.sizes().begin(), self.sizes().end(), half_sizes.begin());
    half_sizes[dim] /= 2;
    return half_sizes;
  }
};

SplitGLUInputTensor::SplitGLUInputTensor(const Tensor& self, int64_t dim)
    : half_sizes(get_half_sizes(self, dim)),
      first_half_impl(
          self.scalar_type(),
          self.dim(),
          half_sizes.data(),
          self.mutable_data_ptr(),
          const_cast<executorch::aten::DimOrderType*>(self.dim_order().data()),
          const_cast<executorch::aten::StridesType*>(self.strides().data()),
          self.shape_dynamism()),
      second_half_impl(
          self.scalar_type(),
          self.dim(),
          half_sizes.data(),
          reinterpret_cast<char*>(self.mutable_data_ptr()) +
              self.strides()[dim] * self.size(dim) / 2 * self.element_size(),
          const_cast<executorch::aten::DimOrderType*>(self.dim_order().data()),
          const_cast<executorch::aten::StridesType*>(self.strides().data()),
          self.shape_dynamism()),
      first_half(&first_half_impl),
      second_half(&second_half_impl) {}

/**
 * Applies the gated linear unit function
 *
 * Based on the characteristic of glu function, the output should be in
 * floating point type (Float and Double). The input and output tensors don't
 * necessarily need to have the same type. Here are the assertions:
 *  1. The input shall be in any float types (Float, Double)
 *  2. The output shall be in float types (Float, Double)
 */
template <typename CTYPE_IN, typename CTYPE_OUT>
Tensor& glu_out_tensor(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      self.dim() <= static_cast<ssize_t>(kTensorDimensionLimit),
      InvalidArgument,
      out);
  SplitGLUInputTensor split_input(self, dim);
  ScalarType compute_type =
      executorch::runtime::isFloatingType(self.scalar_type())
      ? self.scalar_type()
      : ScalarType::Float;
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "glu.out";
  ET_SWITCH_FLOATHBF16_TYPES(compute_type, ctx, op_name, CTYPE_COMPUTE, [&]() {
    utils::apply_bitensor_elementwise_fn<
        CTYPE_COMPUTE,
        op_name,
        utils::SupportedTensorDtypes::FLOATHBF16>(
        [](const auto val_a, const auto val_b) -> CTYPE_COMPUTE {
          // TODO: rewrite this to be vectorization-capable? the
          // tensors might not be contiguous; need to have
          // apply_bitensor_elementwise_fn check that.
          const auto one = static_cast<decltype(val_a)>(1.0);
          return val_a * (one / (one + std::exp(-val_b)));
        },
        ctx,
        split_input.first_half,
        utils::SupportedTensorDtypes::FLOATHBF16,
        split_input.second_half,
        utils::SupportedTensorDtypes::FLOATHBF16,
        out,
        utils::internal::SupportNoncontiguousInputTensors());
  });
  return out;
}
} // namespace

/**
 * Applies the gated linear unit function
 *
 * Based on the characteristic of glu function, the output should be in
 * floating point type (Float and Double). The input and output tensors don't
 * necessarily need to have the same type. Here are the assertions:
 *  1. The input shall be in any float types (Float, Double)
 *  2. The output shall be in float types (Float, Double)
 */
Tensor& glu_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, resize_glu_out(self, dim, out) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(self, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, check_glu_args(self, dim, out), InvalidArgument, out);

  const size_t non_negative_dim = dim < 0 ? dim + self.dim() : dim;
  const auto in_dtype = self.scalar_type();

  ET_SWITCH_FLOATHBF16_TYPES(in_dtype, ctx, "glu", CTYPE_IN, [&]() {
    ET_SWITCH_FLOATHBF16_TYPES(out.scalar_type(), ctx, "glu", CTYPE_OUT, [&]() {
      glu_out_tensor<CTYPE_IN, CTYPE_OUT>(ctx, self, non_negative_dim, out);
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
