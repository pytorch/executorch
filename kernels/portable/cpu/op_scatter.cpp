/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cinttypes>
#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {

template <typename CTYPE>
void scatter_src_helper(
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  const CTYPE* in_data = in.const_data_ptr<CTYPE>();
  const long* index_data = index.const_data_ptr<long>();
  const CTYPE* src_data = src.const_data_ptr<CTYPE>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

  memcpy(out_data, in_data, in.nbytes());

  if (dim < 0) {
    dim += nonzero_dim(in);
  }

  for (size_t ix = 0; ix < index.numel(); ++ix) {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    size_t ix_coord[kTensorDimensionLimit];
    indexToCoordinate(index, ix, ix_coord);

    size_t src_ix = coordinateToIndex(src, ix_coord);

    // @lint-ignore CLANGTIDY facebook-hte-CArray
    size_t out_coord[kTensorDimensionLimit];
    for (size_t i = 0; i < out.dim(); ++i) {
      if (i == dim) {
        out_coord[i] = index_data[ix];
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
  const CTYPE* in_data = in.const_data_ptr<CTYPE>();
  const long* index_data = index.const_data_ptr<long>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

  memcpy(out_data, in_data, in.nbytes());

  if (dim < 0) {
    dim += nonzero_dim(in);
  }

  for (size_t ix = 0; ix < index.numel(); ++ix) {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    size_t ix_coord[kTensorDimensionLimit];
    indexToCoordinate(index, ix, ix_coord);

    // @lint-ignore CLANGTIDY facebook-hte-CArray
    size_t out_coord[kTensorDimensionLimit];
    for (size_t i = 0; i < out.dim(); ++i) {
      if (i == dim) {
        out_coord[i] = index_data[ix];
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
    KernelRuntimeContext& context,
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  (void)context;

  ET_KERNEL_CHECK(
      context,
      check_scatter_src_args(in, dim, index, src, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      context,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out);

  constexpr auto name = "scatter.src_out";

  ET_SWITCH_REALHB_TYPES(in.scalar_type(), ctx, name, CTYPE, [&]() {
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

  ScalarType val_type = utils::get_scalar_dtype(value);

  constexpr auto name = "scatter.value_out";

  ET_SWITCH_SCALAR_OBJ_TYPES(val_type, ctx, name, CTYPE_VAL, [&] {
    CTYPE_VAL val;
    utils::extract_scalar(value, &val);

    ET_SWITCH_REALHB_TYPES(in.scalar_type(), ctx, name, CTYPE, [&]() {
      scatter_value_helper<CTYPE>(in, dim, index, val, out);
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
