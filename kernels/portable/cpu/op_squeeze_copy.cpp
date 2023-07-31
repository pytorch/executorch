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

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

void check_squeeze_copy_dim_out(
    const Tensor input,
    int64_t dim,
    const Tensor out) {
  if (input.dim() != 0 && input.size(dim) == 1) {
    ET_CHECK(input.dim() == out.dim() + 1);

    for (size_t d = 0; d < out.dim(); ++d) {
      if (d < dim) {
        // d < dim
        ET_CHECK_MSG(
            input.size(d) == out.size(d),
            "input.size(%zu) %zd != out.size(%zu) %zd | dim = %" PRId64,
            d,
            input.size(d),
            d,
            out.size(d),
            dim);
      } else {
        // d >= dim
        ET_CHECK_MSG(
            input.size(d + 1) == out.size(d),
            "input.size(%zu) %zd != out.size(%zu) %zd | dim = %" PRId64,
            d + 1,
            input.size(d),
            d,
            out.size(d),
            dim);
      }
    }
  } else {
    ET_CHECK(input.dim() == out.dim());

    for (size_t d = 0; d < out.dim(); ++d) {
      ET_CHECK_MSG(
          input.size(d) == out.size(d),
          "input.size(%zu) %zd != out.size(%zu) %zd | dim = %" PRId64,
          d,
          input.size(d),
          d,
          out.size(d),
          dim);
    }
  }
}
} // namespace

//
// squeeze_copy.dim_out(Tensor self, int dim, Tensor(a!) out) -> Tensor(a!)
//
Tensor& squeeze_copy_dim_out(
    RuntimeContext& context,
    const Tensor& self,
    int64_t dim,
    Tensor& out) {
  (void)context;
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];

  // The input and out shall share same dtype
  ET_CHECK_SAME_DTYPE2(self, out);

  // A valid dim must be in [-self.dim(), self.dim())
  ET_CHECK_MSG(
      (self.dim() == 0 && dim == 0) || (dim >= -self.dim() && dim < self.dim()),
      "dim %" PRId64 " out of range [-%zd,%zd)",
      dim,
      self.dim(),
      self.dim());

  if (dim < 0) {
    dim += self.dim();
  }

  size_t expected_out_dim = (self.dim() == 0 || self.size(dim) != 1)
      ? self.dim()
      : std::max<ssize_t>(self.dim() - 1, 0);

  if (dim == self.dim() || self.size(dim) != 1) {
    for (size_t i = 0; i < expected_out_dim; ++i) {
      expected_output_size[i] = self.size(i);
    }
  } else {
    // 0 <= dim < self.dim() AND self.size(dim) == 1
    for (size_t i = 0; i < expected_out_dim; ++i) {
      if (i < dim) {
        expected_output_size[i] = self.size(i);
      } else {
        // Squeeze the given dimension 'dim'
        expected_output_size[i] = self.size(i + 1);
      }
    }
  }
  ET_CHECK_MSG(
      Error::Ok == resize_tensor(out, {expected_output_size, expected_out_dim}),
      "Failed to resize output tensor.");
  check_squeeze_copy_dim_out(self, dim, out);

  if (self.nbytes() > 0) {
    // Note that this check is important. It's valid for a tensor with numel 0
    // to have a null data pointer, but in some environments it's invalid to
    // pass a null pointer to memcpy() even when the size is zero.
    memcpy(out.data_ptr(), self.data_ptr(), self.nbytes());
  }
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
