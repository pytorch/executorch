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

#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {

template <typename CTYPE>
void gather_helper(
    const Tensor& in,
    const Tensor& index,
    Tensor& out,
    int64_t dim) {
  const CTYPE* in_data = in.const_data_ptr<CTYPE>();
  const long* index_data = index.const_data_ptr<long>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

  if (index.dim() == 0) {
    out_data[0] = in_data[index_data[0]];
    return;
  }

  for (size_t ix = 0; ix < index.numel(); ++ix) {
    size_t ix_coord[kTensorDimensionLimit];
    indexToCoordinate(index, ix, ix_coord);

    size_t in_coord[kTensorDimensionLimit];
    for (size_t i = 0; i < out.dim(); ++i) {
      if (i == dim) {
        in_coord[i] = index_data[ix];
      } else {
        in_coord[i] = ix_coord[i];
      }
    }

    size_t in_ix = coordinateToIndex(in, in_coord);
    size_t out_ix = coordinateToIndex(out, ix_coord);

    out_data[out_ix] = in_data[in_ix];
  }
}

} // namespace

Tensor& gather_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_gather_args(in, dim, index, sparse_grad, out),
      InvalidArgument,
      out);

  if (dim < 0) {
    dim += nonzero_dim(in);
  }

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, index.sizes()) == Error::Ok,
      InvalidArgument,
      out);

  constexpr auto name = "gather.out";

  ET_SWITCH_REALHB_TYPES(in.scalar_type(), ctx, name, CTYPE, [&]() {
    gather_helper<CTYPE>(in, index, out, dim);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
