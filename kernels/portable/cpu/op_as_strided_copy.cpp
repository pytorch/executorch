/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {

/**
 * Copy input_data to output_data according to the stride and shape recursively
 */
template <typename CTYPE_IN>
void as_strided_copy(
    CTYPE_IN* input_data,
    CTYPE_IN* output_data,
    Tensor& out,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    int64_t dim) {
  // the last dimension, copy data
  if (dim == size.size() - 1) {
    for (size_t i = 0; i < size.at(dim); ++i) {
      output_data[i] = *input_data;
      input_data += stride.at(dim);
    }
    return;
  }
  size_t trailing_dims = getTrailingDims(out, dim);
  // recursively set data for the next dimension
  for (size_t i = 0; i < size.at(dim); ++i) {
    as_strided_copy<CTYPE_IN>(
        input_data, output_data, out, size, stride, dim + 1);
    input_data += stride.at(dim);
    output_data += trailing_dims;
  }
}

} // namespace

Tensor& as_strided_copy_out(
    RuntimeContext& ctx,
    const Tensor& self,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    optional<int64_t> storage_offset,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_as_strided_copy_args(self, size, stride, storage_offset, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, size) == torch::executor::Error::Ok,
      InvalidArgument,
      out);

  if (self.numel() == 0) {
    return out;
  }

  size_t offset = storage_offset.has_value() ? storage_offset.value() : 0;

  ET_SWITCH_ALL_TYPES(self.scalar_type(), ctx, __func__, CTYPE, [&] {
    CTYPE* self_data = self.mutable_data_ptr<CTYPE>() + offset;
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

    if (size.empty()) {
      out_data[0] = self_data[0];
    } else {
      as_strided_copy<CTYPE>(self_data, out_data, out, size, stride, 0);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
