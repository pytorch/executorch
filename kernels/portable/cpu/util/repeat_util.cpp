/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

namespace {

bool check_repeat_args(
    Tensor self,
    exec_aten::ArrayRef<int64_t> repeats,
    Tensor& out) {
  // Ensure the self tensors list is non-empty.
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      repeats.size() >= self.dim(),
      "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

  // Repeat arrayref shall not contain negative element.
  bool all_non_negative = true;
  for (auto repeat : repeats) {
    all_non_negative = all_non_negative && (repeat >= 0);
  }
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      all_non_negative, "Trying to create tensor with negative dimension");

  /// Check if out.size() is legal.
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      out.dim() == repeats.size(),
      "The dimension of out shall equal size of repeats, but now is %zd and %zd",
      out.dim(),
      repeats.size());

  // Right now we only support the tensors whose dimension is no greater than
  // kTensorDimensionLimit. Only check out tensor because the number of
  // dimension of out tensor shall have more than or equal to self tensor
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      out.dim() <= kTensorDimensionLimit,
      "The dimension of input and output should not be larger than %zd",
      kTensorDimensionLimit);

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(out, self));

  // We pad one to the beginning of self.size() to make its length equal
  // repeats, and called it reformat_self_size. We then make point-to-point mul
  // of reformat_self_size and repeats. The result should equal out.size().
  size_t reformat_self_size[kTensorDimensionLimit];
  for (size_t i = 0; i < out.dim() - self.dim(); i++) {
    reformat_self_size[i] = 1;
  }

  for (int64_t i = 0; i < self.dim(); i++) {
    reformat_self_size[out.dim() - 1 - i] = self.size(self.dim() - 1 - i);
  }
  for (size_t i = 0; i < repeats.size(); i++) {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        reformat_self_size[i] * repeats[i] == out.size(i),
        "Expect out size at dimension %zu is %" PRId64 ", but now is %zd",
        i,
        reformat_self_size[i] * repeats[i],
        out.size(i));
  }

  return true;
}

// Given the indices to a point in an n-D tensor, and the stride (in bytes)
// along each dimension, return the offset from origin to that point.
size_t compute_access_offset(
    const size_t* indices,
    const size_t* strides,
    size_t num_entries) {
  size_t offset = 0;
  for (int i = num_entries - 1; i >= 0; --i) {
    // @lint-ignore CLANGTIDY indices and strides share same length.
    offset += indices[i] * strides[i];
  }
  return offset;
}

// Copy an self array to multiple coordinates of the out tensor.
//'in_offset' identifies the offset to the source data in self tensor.
//'out_offset' identifies the offset in out tensor where the source data
// should be copied.
//'strides' indicates the stride along each dimension of the out tensor.
void repeat_internal(
    const Tensor& self,
    Tensor& out,
    size_t in_offset,
    size_t out_offset,
    const size_t* strides) {
  const char* src = self.const_data_ptr<char>() + in_offset;
  char* dest = out.mutable_data_ptr<char>() + out_offset;

  // Treats zero-dim self as one-dim tensor with size {1}.
  ssize_t self_dim = self.dim() ? self.dim() : 1;
  int32_t one = 1;
  exec_aten::ArrayRef<int32_t> self_size =
      self.dim() ? self.sizes() : exec_aten::ArrayRef<int32_t>(&one, 1);

  // Get the size of the array in bytes.
  size_t num_bytes = self_size[self_dim - 1] * out.element_size();
  if (num_bytes == 0) {
    return;
  }

  // Visualize the out tensor as a set of 1D arrays. Given an n-dimensional
  // out X[d0, d1, ..., d{N-2}, d{N-1}], we can view the out as a tensor
  // X'[d0, d1, ..., d{N-2}, 1], where each point is a array of length
  // size(d{N-1}). Below is the strategy to iterate over the relevant points in
  // X'. We create an n-D slot array, where each index corresponds to a
  // dimension of X'. A valid value of slot array is one which corresponds to
  // a data point in X'.
  size_t slots[kTensorDimensionLimit];
  memset(slots, 0, self_dim * sizeof(slots[0]));

  // The increment along index of slot array to reach the next possible valid
  // value.
  int64_t incr[kTensorDimensionLimit];
  for (size_t i = 0; i < self_dim; i++) {
    incr[i] = self_size[i];
  }

  // And now copy the self data to possibly multiple points in the out
  // tensor. Note that if the self is n-dimensional tensor, we limit copying
  // to only n dimensions in out tensor (out can be higher-dimensional
  // than self).
  size_t index = self_dim - 1;
  size_t start = out.dim() - self_dim;
  while (slots[0] != out.size(start)) {
    // Compute the offset (from origin) in the out tensor where this self
    // data will be copied to.
    size_t offset = compute_access_offset(slots, strides, self_dim);
    memcpy(dest + offset, src, num_bytes);

    // Find the next valid value of slot array.
    slots[index] += incr[index];
    // If we have reached the limit in the innermost dimension, successively
    // increment the slot index of outer dimensions.
    while (slots[index] == out.size(start + index)) {
      if (index == 0) {
        break;
      }
      slots[index--] = 0;
      slots[index] += incr[index];
    }
    index = self_dim - 1;
  }
}

} // namespace

// TODO(gasoonjia): dynamic allocate array to support tensor dimension larger
// than kTensorDimensionLimit.
Error repeat_tensor(
    const Tensor& self,
    exec_aten::ArrayRef<int64_t> repeats,
    Tensor& out) {
  // Verify that the args are valid.
  ET_CHECK_OR_RETURN_ERROR(
      check_repeat_args(self, repeats, out),
      InvalidArgument,
      "Repeat arguments are invalid.");

  // Returns out if out.numel == 0, nothing needs to be repeated.
  if (out.numel() == 0) {
    return Error::Ok;
  }

  ssize_t element_size = out.element_size();

  // The underlying data of tensor out shall equal tensor self.
  // Treats it specially to circumvent zero-dim tensor issue.
  if (out.numel() == 1) {
    const char* src = self.const_data_ptr<char>();
    char* dest = out.mutable_data_ptr<char>();
    memcpy(dest, src, element_size);
    return Error::Ok;
  }

  // Treats zero-dim self as one-dim tensor with size {1}.
  ssize_t self_dim = self.dim() ? self.dim() : 1;
  int32_t one = 1;
  exec_aten::ArrayRef<int32_t> self_size = self.sizes().empty()
      ? exec_aten::ArrayRef<int32_t>(&one, 1)
      : self.sizes();

  // Compute the stride (in bytes) along each out tensor dimension.
  size_t strides[kTensorDimensionLimit];
  memset(strides, 0, sizeof(strides[0]) * self_dim);
  size_t start = out.dim() - self_dim;
  size_t accum_offset = element_size;

  for (ssize_t i = self_dim - 1; i >= 0; i--) {
    strides[i] = accum_offset;
    accum_offset *= out.size(start + i);
  }

  // Given an n-dimensional self X[d0, d1, ..., d{N-2}, d{N-1}], we can view
  // the self as a tensor X'[d0, d1, ..., d{N-2}, 1], where each point is a
  // 1D array of length size(d{N-1}). Now we need a strategy to iterate over
  // all the points in X'. We do not want to use getLeadingDims(), as we want
  // to know the indices explicitly so that we can compute the appropriate
  // offset for both self and out tensor at that index. To achieve this,
  // we create an n-D slot array, where each index corresponds to a dimension
  // of X'. A valid value of slot array is the one that corresponds to a
  // valid index in X'.
  size_t slots[kTensorDimensionLimit];
  memset(slots, 0, self_dim * sizeof(slots[0]));

  // 'limits' indicates the upper bound on each index in slot. Note that we
  // copy the entire array along the innermost dimension as a direct memcpy,
  // so we reset the upper bound of innermost dim to 1. 'in_incr' indicates
  // the size (in bytes) of the self data.
  int64_t limits[kTensorDimensionLimit];
  for (size_t i = 0; i < self_dim; i++) {
    limits[i] = self_size[i];
  }

  // @lint-ignore CLANGTIDY Here limits is guaranteend a non-empty array.
  size_t in_incr = limits[self_dim - 1] * element_size;
  limits[self_dim - 1] = 1;
  // 'in_offset' indicates the offset (in bytes) from the origin to an self
  // data.
  size_t in_offset = 0;
  size_t index = self_dim - 1;

  // Below, we copy the entire self tensor into the out tensor (at origin),
  // one array a time. To do so, we iterate over all the valid values of slots
  // array. The repeat_internal() takes care of replicating the array along the
  // coordinates specified by repeats array.
  while (slots[0] != limits[0]) {
    // Compute the offset (from origin) in the out tensor where the self
    // array (with indices in self tensor indicated by slots) will be copied.
    size_t out_offset = compute_access_offset(slots, strides, self_dim);
    // Now repeatedly copy the array to multiple coordinates in the out
    // tensor. Curtail the copy along as many dimensions as the self tensor.
    // The copy along remaining higher dimensions can be done via trivial
    // memcpy.
    repeat_internal(self, out, in_offset, out_offset, strides);

    // Find the next valid value of slot array
    slots[index]++;
    // If we have reached the limit in the innermost dimension, successively
    // increment the slot index of outer dimensions.
    while (slots[index] == limits[index]) {
      if (index == 0) {
        break;
      }
      slots[index--] = 0;
      slots[index]++;
    }
    index = self_dim - 1;
    in_offset += in_incr;
  }

  // And now if an n-D self was meant to be replicated to m dimensions where
  // m>n, we can just do simple memcpy for (m-n) dimensions.
  const char* src = out.const_data_ptr<char>();
  char* dest = out.mutable_data_ptr<char>() + accum_offset;
  for (int i = start - 1; i >= 0; --i) {
    for (int j = 0; j < repeats[i] - 1; ++j) {
      memcpy(dest, src, accum_offset);
      dest += accum_offset;
    }
    accum_offset *= out.size(i);
  }

  return Error::Ok;
}

} // namespace executor
} // namespace torch
