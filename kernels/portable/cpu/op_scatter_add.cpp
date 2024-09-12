/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <cstring>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {

template <typename CTYPE>
void scatter_add_helper(
    const CTYPE* src_data,
    const long* index_data,
    CTYPE* out_data,
    const Tensor& src,
    const Tensor& index,
    Tensor& out,
    int64_t dim) {
  for (size_t ix = 0; ix < index.numel(); ++ix) {
    size_t ix_coord[kTensorDimensionLimit];
    indexToCoordinate(index, ix, ix_coord);

    size_t src_ix = coordinateToIndex(src, ix_coord);

    size_t out_coord[kTensorDimensionLimit];
    for (size_t i = 0; i < out.dim(); ++i) {
      if (i == dim) {
        out_coord[i] = index_data[ix];
      } else {
        out_coord[i] = ix_coord[i];
      }
    }
    size_t out_ix = coordinateToIndex(out, out_coord);

    out_data[out_ix] += src_data[src_ix];
  }
}

} // namespace

Tensor& scatter_add_out(
    KernelRuntimeContext& context,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  (void)context;

  ET_KERNEL_CHECK(
      context,
      check_scatter_add_args(self, dim, index, src, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      context,
      tensors_have_same_dim_order(self, src, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      context, tensor_is_default_dim_order(index), InvalidArgument, out);

  if (dim < 0) {
    dim += nonzero_dim(self);
  }

  ET_KERNEL_CHECK(
      context,
      resize_tensor(out, self.sizes()) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType self_type = self.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(
      Bool, self_type, ctx, "scatter_add.out", CTYPE, [&]() {
        const CTYPE* self_data = self.const_data_ptr<CTYPE>();
        const long* index_data = index.const_data_ptr<long>();
        const CTYPE* src_data = src.const_data_ptr<CTYPE>();
        CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

        memcpy(out_data, self_data, self.nbytes());

        if (index.numel() != 0) {
          if (self.dim() == 0) {
            out_data[0] += nonempty_size(index, 0) * src_data[0];
          } else {
            scatter_add_helper<CTYPE>(
                src_data, index_data, out_data, src, index, out, dim);
          }
        }
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
