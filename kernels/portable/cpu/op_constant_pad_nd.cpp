/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <algorithm>
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
    executorch::aten::ArrayRef<executorch::aten::DimOrderType> dim_order,
    const CTYPE* self_data,
    IntArrayRef self_sizes,
    IntArrayRef self_strides,
    CTYPE* out_data,
    CTYPE* out_data_end,
    IntArrayRef out_strides,
    IntArrayRef pad,
    const CTYPE value,
    size_t last_padded_dim,
    size_t dim) {
  if (dim >= ndim) {
    return;
  }

  const size_t logical_dim = dim_order[dim];
  size_t pad_i = ndim - 1 - logical_dim;

  size_t pad_before = 0;
  size_t pad_after = 0;
  if (pad_i < pad.size() / 2) {
    pad_before = std::max<int64_t>(pad[2 * pad_i], 0);
    pad_after = std::max<int64_t>(pad[2 * pad_i + 1], 0);
  }

  size_t out_step_len = out_strides[logical_dim];
  size_t in_step_len = self_strides[logical_dim];

  // Do not copy padding beyond the out tensor bounds.
  // Use division to avoid potential overflow in multiplication.
  if (pad_before > 0) {
    ET_KERNEL_CHECK_MSG(
        ctx,
        out_data <= out_data_end,
        InvalidArgument,
        /* void */,
        "Out data pointer exceeds buffer bounds.");
    size_t remaining = out_data_end - out_data;
    ET_KERNEL_CHECK_MSG(
        ctx,
        out_step_len > 0 && remaining / out_step_len >= pad_before,
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
    size_t copy_len = in_step_len * self_sizes[logical_dim];
    size_t copy_nbytes = copy_len * sizeof(CTYPE);

    if (copy_nbytes > 0) {
      // Bounds check before memcpy
      ET_KERNEL_CHECK_MSG(
          ctx,
          out_data <= out_data_end,
          InvalidArgument,
          /* void */,
          "Out data pointer exceeds buffer bounds.");
      size_t remaining = out_data_end - out_data;
      ET_KERNEL_CHECK_MSG(
          ctx,
          remaining >= copy_len,
          InvalidArgument,
          /* void */,
          "Out tensor is too small for the copy operation.");
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
    }
  }
  // Otherwise, call this function recursively
  else {
    for (const auto i : c10::irange(self_sizes[logical_dim])) {
      apply_padding_to_dim(
          ctx,
          ndim,
          dim_order,
          self_data,
          self_sizes,
          self_strides,
          out_data,
          out_data_end,
          out_strides,
          pad,
          value,
          last_padded_dim,
          dim + 1);

      if (ctx.failure_state() != Error::Ok) {
        return;
      }

      out_data += out_step_len;
      if (i + 1 < self_sizes[logical_dim]) {
        self_data += in_step_len;
      }
    }
  }

  // Do not copy padding beyond the out tensor bounds.
  // Use division to avoid potential overflow in multiplication.
  if (pad_after > 0) {
    ET_KERNEL_CHECK_MSG(
        ctx,
        out_data <= out_data_end,
        InvalidArgument,
        /* void */,
        "Out data pointer exceeds buffer bounds.");
    size_t remaining = out_data_end - out_data;
    ET_KERNEL_CHECK_MSG(
        ctx,
        out_step_len > 0 && remaining / out_step_len >= pad_after,
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
  if (out.numel() == 0) {
    return;
  }

  const CTYPE* self_data = self.const_data_ptr<CTYPE>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

  size_t ndim = self.dim();

  if (ndim == 0) {
    out_data[0] = self_data[0];
    return;
  }

  int64_t self_sizes[kTensorDimensionLimit];
  int64_t self_strides[kTensorDimensionLimit];
  int64_t out_strides[kTensorDimensionLimit];

  // Collect sizes and strides of input and output tensors and determine the
  // last padded dimension
  size_t last_padded_dim = 0;
  size_t input_offset = 0;
  for (const auto i : c10::irange(ndim)) {
    const size_t dim = self.dim_order()[i];
    self_sizes[dim] = self.size(dim);
    self_strides[dim] = self.strides()[dim];
    out_strides[dim] = out.strides()[dim];

    size_t pad_i = ndim - 1 - dim;
    if (pad_i < pad.size() / 2) {
      const int64_t crop_before = -std::min<int64_t>(pad[2 * pad_i], 0);
      const int64_t crop_after = -std::min<int64_t>(pad[2 * pad_i + 1], 0);
      self_sizes[dim] -= crop_before + crop_after;
      input_offset += crop_before * self_strides[dim];
      if (pad[2 * pad_i] != 0 || pad[2 * pad_i + 1] != 0) {
        last_padded_dim = i;
      }
    }
    if (self_sizes[dim] == 0) {
      set_all_to_value(out_data, out.numel(), value_v);
      return;
    }
  }

  IntArrayRef self_sizes_ref(self_sizes, ndim);
  IntArrayRef self_strides_ref(self_strides, ndim);
  IntArrayRef out_strides_ref(out_strides, ndim);

  CTYPE* out_data_end = out_data + out.numel();

  apply_padding_to_dim(
      ctx,
      ndim,
      self.dim_order(),
      self_data + input_offset,
      self_sizes_ref,
      self_strides_ref,
      out_data,
      out_data_end,
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

  const bool has_positive_padding =
      std::any_of(pad.begin(), pad.end(), [](int64_t p) { return p > 0; });
  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, op_name, CTYPE, [&]() {
    // PyTorch ignores the fill value when the operation only crops or copies.
    auto opt_value_casted = utils::internal::check_overflow_scalar_cast<CTYPE>(
        has_positive_padding ? value : Scalar(0));
    ET_KERNEL_CHECK(ctx, opt_value_casted.has_value(), InvalidArgument, );
    auto value_casted = opt_value_casted.value();
    constant_pad_nd_out_impl<CTYPE>(ctx, in, pad, value_casted, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
