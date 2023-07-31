/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>

#include <executorch/kernels/portable/cpu/scalar_utils.h>

namespace torch {
namespace executor {
namespace native {

namespace {

void check_input_output(const Tensor& self, const Tensor& out) {
  ET_CHECK_SAME_DTYPE2(self, out);
  ET_CHECK_MSG(
      self.dim() == out.dim(),
      "self and out must have the same number of dims");
}

void check_padding_arg(const Tensor& self, IntArrayRef pad) {
  ET_CHECK_MSG(pad.size() % 2 == 0, "Padding array must be a multiple of 2");

  ET_CHECK_MSG(
      pad.size() / 2 <= self.dim(), "Padding array contains too many elements");
}

Error resize_to_expected_out_size(
    const Tensor& self,
    IntArrayRef pad,
    Tensor& out) {
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];

  int pad_i = self.dim() - 1;
  for (size_t i = 0; i < self.dim(); ++i, --pad_i) {
    expected_output_size[i] = self.size(i);
    if (pad_i >= 0 && pad_i < pad.size() / 2) {
      expected_output_size[i] += pad[2 * pad_i] + pad[2 * pad_i + 1];
    }
  }

  ArrayRef<Tensor::SizesType> output_size{
      expected_output_size, static_cast<size_t>(self.dim())};
  auto error = resize_tensor(out, output_size);

  return error;
}

template <typename CTYPE>
void set_all_to_value(CTYPE* out_data, size_t step_len, CTYPE value) {
  for (size_t i = 0; i < step_len; ++i) {
    out_data[i] = value;
  }
}

template <typename CTYPE>
void apply_padding_to_dim(
    size_t ndim,
    const CTYPE* self_data,
    IntArrayRef self_sizes,
    IntArrayRef self_strides,
    CTYPE* out_data,
    IntArrayRef out_sizes,
    IntArrayRef out_strides,
    IntArrayRef pad,
    const CTYPE value,
    size_t last_padded_dim,
    size_t dim) {
  if (dim >= ndim) {
    return;
  }

  size_t pad_i = ndim - 1 - dim;

  size_t pad_before = 0;
  size_t pad_after = 0;
  if (pad_i >= 0 && pad_i < pad.size() / 2) {
    pad_before = pad[2 * pad_i];
    pad_after = pad[2 * pad_i + 1];
  }

  size_t out_step_len = out_strides[dim];
  size_t in_step_len = self_strides[dim];

  for (size_t i = 0; i < pad_before; ++i) {
    set_all_to_value(out_data, out_step_len, value);
    out_data += out_step_len;
  }

  // If subsequent dims are not padded, then the whole block of memory can be
  // copied.
  if (dim >= last_padded_dim) {
    size_t copy_len = in_step_len * self_sizes[dim];
    size_t copy_nbytes = copy_len * sizeof(CTYPE);

    memcpy(out_data, self_data, copy_nbytes);

    out_data += copy_len;
    self_data += copy_len;
  }
  // Otherwise, call this function recursively
  else {
    for (size_t i = 0; i < self_sizes[dim]; ++i) {
      apply_padding_to_dim(
          ndim,
          self_data,
          self_sizes,
          self_strides,
          out_data,
          out_sizes,
          out_strides,
          pad,
          value,
          last_padded_dim,
          dim + 1);

      out_data += out_step_len;
      self_data += in_step_len;
    }
  }

  for (int i = 0; i < pad_after; ++i) {
    set_all_to_value(out_data, out_step_len, value);
    out_data += out_step_len;
  }
}

template <typename CTYPE>
void constant_pad_nd_out_impl(
    const Tensor& self,
    IntArrayRef pad,
    CTYPE value_v,
    Tensor& out) {
  const CTYPE* self_data = self.data_ptr<CTYPE>();
  CTYPE* out_data = out.data_ptr<CTYPE>();

  size_t ndim = self.dim();

  int64_t self_sizes[kTensorDimensionLimit];
  int64_t self_strides[kTensorDimensionLimit];
  int64_t out_sizes[kTensorDimensionLimit];
  int64_t out_strides[kTensorDimensionLimit];

  // Collect sizes and strides of input and output tensors and determine the
  // last padded dimension
  size_t last_padded_dim = 0;
  for (size_t i = 0; i < ndim; ++i) {
    self_sizes[i] = self.size(i);
    self_strides[i] = getTrailingDims(self, static_cast<int64_t>(i));
    out_sizes[i] = out.size(i);
    out_strides[i] = getTrailingDims(out, static_cast<int64_t>(i));

    size_t pad_i = ndim - 1 - i;
    if (pad_i >= 0 && pad_i < pad.size() / 2) {
      if (pad[2 * pad_i] + pad[2 * pad_i + 1] > 0) {
        last_padded_dim = i;
      }
    }
  }

  IntArrayRef self_sizes_ref(self_sizes, ndim);
  IntArrayRef self_strides_ref(self_strides, ndim);
  IntArrayRef out_sizes_ref(out_sizes, ndim);
  IntArrayRef out_strides_ref(out_strides, ndim);

  apply_padding_to_dim(
      ndim,
      self_data,
      self_sizes_ref,
      self_strides_ref,
      out_data,
      out_sizes_ref,
      out_strides_ref,
      pad,
      value_v,
      last_padded_dim,
      0);
}

} // namespace

Tensor& constant_pad_nd_out(
    RuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef pad,
    const Scalar& value,
    Tensor& out) {
  (void)ctx;

  check_input_output(in, out);
  check_padding_arg(in, pad);

  // resize out tensor for dynamic shapes
  auto error = resize_to_expected_out_size(in, pad, out);
  // TODO: Construct error message with requested output sizes.
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  ScalarType in_type = in.scalar_type();
  ScalarType value_type = utils::get_scalar_dtype(value);
  ScalarType out_type = out.scalar_type();

  ET_CHECK(in_type == out_type);

  ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, __func__, CTYPE, [&]() {
    CTYPE value_v;
    ET_SWITCH_SCALAR_OBJ_TYPES(value_type, ctx, __func__, CTYPE_VALUE, [&]() {
      CTYPE_VALUE val;
      ET_EXTRACT_SCALAR(value, val);
      value_v = static_cast<CTYPE>(val);
    });
    constant_pad_nd_out_impl<CTYPE>(in, pad, value_v, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
