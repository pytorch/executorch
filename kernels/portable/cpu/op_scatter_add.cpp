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

/**
 * Add input_data to output_data in the fashion of scatter recursively
 */
template <typename CTYPE_DATA>
void scatter_add_helper(
    CTYPE_DATA* src_data,
    long* index_data,
    CTYPE_DATA* output_data,
    const Tensor& src,
    const Tensor& index,
    Tensor& out,
    int64_t dim,
    int64_t current_dim,
    int64_t dim_offset) {
  // the last dimension, copy data
  if (current_dim == index.dim() - 1) {
    size_t trailing_dims = getTrailingDims(out, dim);
    CTYPE_DATA* output_data_base =
        output_data - (size_t)dim_offset * trailing_dims;
    for (size_t i = 0; i < index.size(current_dim); ++i) {
      output_data = output_data_base + (size_t)index_data[i] * trailing_dims;
      // if dim is the last dimension, do not need to traverse again
      if (dim == current_dim) {
        *output_data += src_data[i];
      } else {
        output_data[i] += src_data[i];
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
    scatter_add_helper<CTYPE_DATA>(
        src_data,
        index_data,
        output_data,
        src,
        index,
        out,
        dim,
        current_dim + 1,
        current_dim_offset);
    src_data += trailing_dims_src;
    output_data += trailing_dims_out;
    index_data += trailing_dims_index;
    if (current_dim == dim) {
      current_dim_offset += 1;
    }
  }
}

template <typename CTYPE_DATA>
void scatter_add(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  long* index_data = index.mutable_data_ptr<long>();
  for (size_t i = 0; i < index.numel(); ++i) {
    ET_CHECK_MSG(
        index_data[i] < self.size(dim),
        "Index is out of bounds for dimension %zd with size %zd",
        (size_t)dim,
        self.size(dim));
  }
  CTYPE_DATA* self_data = self.mutable_data_ptr<CTYPE_DATA>();
  CTYPE_DATA* src_data = src.mutable_data_ptr<CTYPE_DATA>();
  CTYPE_DATA* out_data = out.mutable_data_ptr<CTYPE_DATA>();
  memcpy(out_data, self_data, self.nbytes());
  scatter_add_helper<CTYPE_DATA>(
      src_data, index_data, out_data, src, index, out, dim, 0, 0);
}

} // namespace

/**
 * Adds all values from the tensor src into self at the indices specified in the
 * index tensor in a similar fashion as scatter(). For each value in src, it is
 * added to an index in self which is specified by its index in src for
 * dimension != dim and by the corresponding value in index for dimension = dim.
 *
 * Assume tensor self, src, out have the same dtype, and shall be in any real
 * types (Byte, Char, Short, Int, Long, Float, Double), and index tensor shall
 * be in Long (int64) type.
 */
Tensor& scatter_add_out(
    RuntimeContext& context,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  (void)context;
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

#define SCATTER_ADD(ctype, dtype)                   \
  case ScalarType::dtype:                           \
    scatter_add<ctype>(self, dim, index, src, out); \
    break;

  switch (self.scalar_type()) {
    ET_FORALL_REAL_TYPES(SCATTER_ADD)
    default:
      ET_CHECK_MSG(false, "Unhandled input dtype %hhd", self.scalar_type());
  }

#undef SCATTER_ADD
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
