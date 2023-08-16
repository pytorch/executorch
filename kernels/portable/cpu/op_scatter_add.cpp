/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {

void check_arguments(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  ET_CHECK_SAME_SHAPE_AND_DTYPE2(self, out);
  ET_CHECK_SAME_DTYPE2(self, src);
  ET_CHECK_MSG(
      index.scalar_type() == ScalarType::Long,
      "Expected dypte int64 for index");
  ET_CHECK_MSG(
      dim >= -self.dim() && dim < self.dim(),
      "dim %" PRId64 " >= 0 && dim %" PRId64 " < self.dim() %zd",
      dim,
      dim,
      self.dim());
  ET_CHECK_MSG(
      self.dim() == src.dim() && self.dim() == index.dim(),
      "self, index and src should have same number of dimensions.");
  dim = dim < 0 ? dim + self.dim() : dim;
  for (size_t d = 0; d < self.dim(); ++d) {
    ET_CHECK_MSG(
        index.size(d) <= src.size(d),
        "size of dimension %zd of index should be smaller than the size of that dimension of src",
        d);
    if (d != dim) {
      ET_CHECK_MSG(
          index.size(d) <= self.size(d),
          "size of dimension %zd of index should be smaller than the size of that dimension of self if dimension %zd != dim %zd",
          d,
          d,
          (size_t)dim);
    }
  }
  const long* index_data = index.const_data_ptr<long>();
  for (size_t i = 0; i < index.numel(); ++i) {
    ET_CHECK_MSG(
        index_data[i] < self.size(dim),
        "Index is out of bounds for dimension %zd with size %zd",
        (size_t)dim,
        self.size(dim));
  }
}

/**
 * Add input_data to output_data in the fashion of scatter recursively
 */
template <typename CTYPE>
void scatter_add_helper(
    const CTYPE* src_data,
    const long* index_data,
    CTYPE* out_data,
    const Tensor& src,
    const Tensor& index,
    Tensor& out,
    int64_t dim,
    int64_t current_dim,
    int64_t dim_offset) {
  // the last dimension, copy data
  if (current_dim == index.dim() - 1) {
    size_t trailing_dims = getTrailingDims(out, dim);
    CTYPE* out_data_base = out_data - (size_t)dim_offset * trailing_dims;
    for (size_t i = 0; i < index.size(current_dim); ++i) {
      out_data = out_data_base + (size_t)index_data[i] * trailing_dims;
      // if dim is the last dimension, do not need to traverse again
      if (dim == current_dim) {
        *out_data += src_data[i];
      } else {
        out_data[i] += src_data[i];
      }
    }
    return;
  }
  size_t trailing_dims_out = getTrailingDims(out, current_dim);
  size_t trailing_dims_src = getTrailingDims(src, current_dim);
  size_t trailing_dims_index = getTrailingDims(index, current_dim);
  size_t current_dim_offset = 0;
  // recursively set data for the next dimension
  for (size_t i = 0; i < index.size(current_dim); ++i) {
    scatter_add_helper<CTYPE>(
        src_data,
        index_data,
        out_data,
        src,
        index,
        out,
        dim,
        current_dim + 1,
        current_dim_offset);
    src_data += trailing_dims_src;
    out_data += trailing_dims_out;
    index_data += trailing_dims_index;
    if (current_dim == dim) {
      current_dim_offset += 1;
    }
  }
}

} // namespace

Tensor& scatter_add_out(
    RuntimeContext& ctx,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  (void)ctx;

  check_arguments(self, dim, index, src, out);

  ScalarType self_type = self.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, self_type, ctx, __func__, CTYPE, [&]() {
    const CTYPE* self_data = self.const_data_ptr<CTYPE>();
    const long* index_data = index.const_data_ptr<long>();
    const CTYPE* src_data = src.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
    memcpy(out_data, self_data, self.nbytes());
    scatter_add_helper<CTYPE>(
        src_data, index_data, out_data, src, index, out, dim, 0, 0);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
