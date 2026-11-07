/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <cinttypes>
#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;

namespace {

// self can be in another dim order than out, like the result of nonzero.
template <typename CTYPE>
void copy_self_to_out(const Tensor& in, Tensor& out) {
  if (in.strides() == out.strides()) {
    memcpy(
        out.mutable_data_ptr<CTYPE>(), in.const_data_ptr<CTYPE>(), in.nbytes());
  } else {
    _to_dim_order_copy_impl<CTYPE, CTYPE>(in, out);
  }
}

template <typename CTYPE>
void scatter_src_helper(
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  const int64_t* index_data = index.const_data_ptr<int64_t>();
  const CTYPE* src_data = src.const_data_ptr<CTYPE>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

  copy_self_to_out<CTYPE>(in, out);

  if (dim < 0) {
    dim += nonzero_dim(in);
  }

  // The index and src can be in any dim order, like the result of nonzero, so
  // both are read through their strides. A contiguous index, the usual case,
  // is read in order.
  const auto index_strides = index.strides();
  const auto src_strides = src.strides();
  const bool index_contiguous = is_contiguous_dim_order(
      index.dim_order().data(), index.dim_order().size());
  for (const auto ix : c10::irange(index.numel())) {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    size_t ix_coord[kTensorDimensionLimit];
    indexToCoordinate(index, ix, ix_coord);

    size_t index_ix = ix;
    if (!index_contiguous) {
      index_ix = 0;
      for (const auto i : c10::irange(index.dim())) {
        index_ix += ix_coord[i] * index_strides[i];
      }
    }
    size_t src_ix = 0;
    for (const auto i : c10::irange(src.dim())) {
      src_ix += ix_coord[i] * src_strides[i];
    }

    // @lint-ignore CLANGTIDY facebook-hte-CArray
    size_t out_coord[kTensorDimensionLimit];
    for (const auto i : c10::irange(out.dim())) {
      if (i == dim) {
        out_coord[i] = index_data[index_ix];
      } else {
        out_coord[i] = ix_coord[i];
      }
    }
    size_t out_ix = coordinateToIndex(out, out_coord);

    out_data[out_ix] = src_data[src_ix];
  }
}

template <typename CTYPE, typename CTYPE_VAL>
void scatter_value_helper(
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    CTYPE_VAL val,
    Tensor& out) {
  const int64_t* index_data = index.const_data_ptr<int64_t>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

  copy_self_to_out<CTYPE>(in, out);

  if (dim < 0) {
    dim += nonzero_dim(in);
  }

  // The index can be in any dim order, like the result of nonzero, so it is
  // read through its strides. A contiguous index, the usual case, is read in
  // order.
  const auto index_strides = index.strides();
  const bool index_contiguous = is_contiguous_dim_order(
      index.dim_order().data(), index.dim_order().size());
  for (const auto ix : c10::irange(index.numel())) {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    size_t ix_coord[kTensorDimensionLimit];
    indexToCoordinate(index, ix, ix_coord);

    size_t index_ix = ix;
    if (!index_contiguous) {
      index_ix = 0;
      for (const auto i : c10::irange(index.dim())) {
        index_ix += ix_coord[i] * index_strides[i];
      }
    }

    // @lint-ignore CLANGTIDY facebook-hte-CArray
    size_t out_coord[kTensorDimensionLimit];
    for (const auto i : c10::irange(out.dim())) {
      if (i == dim) {
        out_coord[i] = index_data[index_ix];
      } else {
        out_coord[i] = ix_coord[i];
      }
    }
    size_t out_ix = coordinateToIndex(out, out_coord);

    out_data[out_ix] = static_cast<CTYPE>(val);
  }
}

} // namespace

Tensor& scatter_src_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_scatter_src_args(in, dim, index, src, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  static constexpr auto name = "scatter.src_out";

  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, name, CTYPE, [&]() {
    scatter_src_helper<CTYPE>(in, dim, index, src, out);
  });

  return out;
}

Tensor& scatter_value_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    const Scalar& value,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_scatter_value_args(in, dim, index, value, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  static constexpr auto name = "scatter.value_out";

  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, name, CTYPE, [&]() {
    auto opt_val = utils::internal::check_overflow_scalar_cast<CTYPE>(value);
    ET_KERNEL_CHECK(ctx, opt_val.has_value(), InvalidArgument, );
    auto val = opt_val.value();
    scatter_value_helper<CTYPE>(in, dim, index, val, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
