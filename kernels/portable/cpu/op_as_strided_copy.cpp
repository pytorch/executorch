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
size_t compute_storage_nbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize_bytes) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  size_t size = 1;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == 0) {
      return 0;
    }
    size += strides[i] * (sizes[i] - 1);
  }
  return size * itemsize_bytes;
}

void check_inbounds_for_storage(
    const Tensor& self,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    int64_t storage_offset) {
  size_t storage_size_bytes =
      compute_storage_nbytes(size, stride, self.element_size());
  size_t storage_offset_bytes = storage_offset * self.element_size();
  if (storage_size_bytes == 0) {
    return;
  }
  size_t new_storage_size_bytes = self.nbytes();
  ET_CHECK_MSG(
      storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
      "Requiring a storage size of %zd are out of bounds for storage of size %zd",
      storage_size_bytes + storage_offset_bytes,
      new_storage_size_bytes);
}

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

void check_preconditions(
    const Tensor& self,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    optional<int64_t> storage_offset,
    Tensor& out) {
  ET_CHECK_SAME_DTYPE2(self, out);
  ET_CHECK_MSG(
      size.size() == stride.size(), "mismatch in length of strides and shape");
  for (const auto& val : stride) {
    ET_CHECK_MSG(
        val >= 0,
        "as_strided: Negative strides are not supported at the moment");
  }
  ET_CHECK_MSG(
      out.sizes().size() == size.size(),
      "output tensor should have same shape as size");
  for (size_t i = 0; i < out.sizes().size(); ++i) {
    ET_CHECK_MSG(
        out.sizes().at(i) == size.at(i),
        "output tensor should have same shape as size");
  }
  int64_t offset = storage_offset.has_value() ? storage_offset.value() : 0;
  ET_CHECK_MSG(offset >= 0, "Negative storage offset");
  check_inbounds_for_storage(self, size, stride, offset);
}

} // namespace

/**
 * Copy the tener `self` to `out`, assume `self` and `out` have same type and
 * shape
 */
Tensor& as_strided_copy_out(
    RuntimeContext& ctx,
    const Tensor& self,
    ArrayRef<int64_t> size,
    ArrayRef<int64_t> stride,
    optional<int64_t> storage_offset,
    Tensor& out) {
  (void)ctx;

  torch::executor::Error err = resize_tensor(out, size);
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in as_strided_copy_out");

  check_preconditions(self, size, stride, storage_offset, out);
  size_t offset = storage_offset.has_value() ? storage_offset.value() : 0;

#define AS_STRIDED_COPY_TENSOR(ctype, dtype)                    \
  case ScalarType::dtype:                                       \
    as_strided_copy<ctype>(                                     \
        /*input_data=*/self.mutable_data_ptr<ctype>() + offset, \
        /*output_data=*/out.mutable_data_ptr<ctype>(),          \
        out,                                                    \
        size,                                                   \
        stride,                                                 \
        /*dim=*/0);                                             \
    break;

  switch (self.scalar_type()) {
    ET_FORALL_SCALAR_TYPES(AS_STRIDED_COPY_TENSOR)
    default:
      ET_CHECK_MSG(false, "Unhandled dtype %hhd", self.scalar_type());
  }
#undef AS_STRIDED_COPY_TENSOR
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
