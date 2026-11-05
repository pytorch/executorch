/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/math_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <tuple>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

/**
 * Computes the running minimum of self along dim, and the index of that
 * minimum. Like PyTorch, a tie moves the index to the later element and a NaN
 * becomes the minimum from where it appears on.
 */
std::tuple<Tensor&, Tensor&> cummin_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    Tensor& values,
    Tensor& indices) {
  std::tuple<Tensor&, Tensor&> ret_val(values, indices);

  ET_KERNEL_CHECK(
      ctx, dim_is_valid(dim, nonzero_dim(self)), InvalidArgument, ret_val);
  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dtype(self, values), InvalidArgument, ret_val);
  ET_KERNEL_CHECK(
      ctx, indices.scalar_type() == ScalarType::Long, InvalidArgument, ret_val);
  ET_KERNEL_CHECK(
      ctx,
      tensor_is_default_dim_order(self) &&
          tensor_is_default_dim_order(values) &&
          tensor_is_default_dim_order(indices),
      InvalidArgument,
      ret_val);
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(values, self.sizes()) == Error::Ok &&
          resize_tensor(indices, self.sizes()) == Error::Ok,
      InvalidArgument,
      ret_val);

  if (self.numel() == 0) {
    return ret_val;
  }

  dim = dim < 0 ? dim + nonzero_dim(self) : dim;
  const size_t dim_size = nonempty_size(self, dim);
  const size_t leading_dims = getLeadingDims(self, dim);
  const size_t trailing_dims = self.dim() == 0 ? 1 : getTrailingDims(self, dim);

  ET_SWITCH_REALHBBF16_TYPES(
      self.scalar_type(), ctx, "cummin.out", CTYPE, [&]() {
        const CTYPE* in_data = self.const_data_ptr<CTYPE>();
        CTYPE* values_data = values.mutable_data_ptr<CTYPE>();
        int64_t* indices_data = indices.mutable_data_ptr<int64_t>();

        for (size_t i = 0; i < leading_dims; i++) {
          const size_t start = i * dim_size * trailing_dims;
          for (size_t j = 0; j < dim_size; j++) {
            const size_t cur = start + j * trailing_dims;
            const size_t prev = cur - trailing_dims;
            for (size_t k = 0; k < trailing_dims; k++) {
              const CTYPE v = in_data[cur + k];
              if (j == 0 || utils::isnan_override(v) ||
                  (!utils::isnan_override(values_data[prev + k]) &&
                   v <= values_data[prev + k])) {
                values_data[cur + k] = v;
                indices_data[cur + k] = static_cast<int64_t>(j);
              } else {
                values_data[cur + k] = values_data[prev + k];
                indices_data[cur + k] = indices_data[prev + k];
              }
            }
          }
        }
      });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
