/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cmath>
#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>

namespace torch {
namespace executor {
namespace native {

namespace {

template <typename CTYPE>
void set_all_to_value(CTYPE* out_data, size_t step_len, CTYPE value) {
  for (size_t i = 0; i < step_len; ++i) {
    out_data[i] = value;
  }
}

template <typename CTYPE>
void apply_padding_to_dim(
    KernelRuntimeContext& ctx,
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

  // Do not copy padding beyond the out tensor bounds.
  if (pad_before > 0) {
    size_t numel = 1;
    for (ET_UNUSED const auto i : c10::irange(out_sizes.size())) {
      numel *= out_sizes[i];
    }
    ET_KERNEL_CHECK_MSG(
        ctx,
        numel >= pad_before * out_step_len,
        InvalidArgument,
        /* void */,
        "Out tensor is too small for the requested padding.");
  }
  for (ET_UNUSED const auto i : c10::irange(pad_before)) {
    set_all_to_value(out_data, out_step_len, value);
    out_data += out_step_len;
  }

  // If subsequent dims are not padded, then the whole block of memory can be
  // copied.
  if (dim >= last_padded_dim) {
    size_t copy_len = in_step_len * self_sizes[dim];
    size_t copy_nbytes = copy_len * sizeof(CTYPE);

    if (copy_nbytes > 0) {
      // Check that out_data and self_data do not overlap.
      ET_KERNEL_CHECK_MSG(
          ctx,
          out_data != self_data &&
              ((out_data + copy_len <= self_data) ||
               (self_data + copy_len <= out_data)),
          InvalidArgument,
          /* void */,
          "Out tensor overlaps with the input tensor. This is not supported.");
      memcpy(out_data, self_data, copy_nbytes);
      out_data += copy_len;
      self_data += copy_len;
    }
  }
  // Otherwise, call this function recursively
  else {
    for (ET_UNUSED const auto i : c10::irange(self_sizes[dim])) {
      apply_padding_to_dim(
          ctx,
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

  // Do not copy padding beyond the out tensor bounds.
  if (pad_after > 0) {
    size_t numel = 1;
    for (ET_UNUSED const auto i : c10::irange(out_sizes.size())) {
      numel *= out_sizes[i];
    }
    ET_KERNEL_CHECK_MSG(
        ctx,
        numel >= pad_after * out_step_len,
        InvalidArgument,
        /* void */,
        "Out tensor is too small for the requested padding.");
  }
  for (ET_UNUSED const auto i : c10::irange(pad_after)) {
    set_all_to_value(out_data, out_step_len, value);
    out_data += out_step_len;
  }
}

template <typename CTYPE>
void constant_pad_nd_out_impl(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    IntArrayRef pad,
    CTYPE value_v,
    Tensor& out) {
  const CTYPE* self_data = self.const_data_ptr<CTYPE>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

  size_t ndim = self.dim();

  if (ndim == 0) {
    out_data[0] = self_data[0];
    return;
  }

  int64_t self_sizes[kTensorDimensionLimit];
  int64_t self_strides[kTensorDimensionLimit];
  int64_t out_sizes[kTensorDimensionLimit];
  int64_t out_strides[kTensorDimensionLimit];

  // Collect sizes and strides of input and output tensors and determine the
  // last padded dimension
  size_t last_padded_dim = 0;
  for (const auto i : c10::irange(ndim)) {
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
      ctx,
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
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef pad,
    const Scalar& value,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_constant_pad_args(in, pad, value, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  // resize out tensor for dynamic shapes
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_constant_pad_output(in, pad, out) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType in_type = in.scalar_type();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "constant_pad_nd.out";

  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, op_name, CTYPE, [&]() {
    auto opt_value_casted =
        utils::internal::check_overflow_scalar_cast<CTYPE>(value);
    ET_KERNEL_CHECK(ctx, opt_value_casted.has_value(), InvalidArgument, );
    auto value_casted = opt_value_casted.value();
    constant_pad_nd_out_impl<CTYPE>(ctx, in, pad, value_casted, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
