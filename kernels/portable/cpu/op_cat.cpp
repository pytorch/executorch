/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& cat_out(
    KernelRuntimeContext& ctx,
    exec_aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  if (dim < 0) {
    dim += out.dim();
  }

  ET_KERNEL_CHECK(ctx, check_cat_args(tensors, dim, out), InvalidArgument, out);

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_cat_out_target_size(tensors, dim, expected_out_size, &expected_out_dim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  // Special handling when all inputs are 1D-empty tensors for aten consistency
  // In that case, just return an 1D-empty tensor without checking dim
  bool all_1d_empty = true;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].numel() != 0 || tensors[i].dim() != 1) {
      all_1d_empty = false;
      break;
    }
  }
  if (all_1d_empty) {
    return out;
  }

  const size_t outer = getLeadingDims(out, dim);
  const size_t dim_stride = getTrailingDims(out, dim);
  const size_t ninputs = tensors.size();

  const auto out_type = out.scalar_type();
  ET_SWITCH_REALHB_TYPES(out_type, ctx, "cat.out", CTYPE_OUT, [&] {
    CTYPE_OUT* out_ptr = out.mutable_data_ptr<CTYPE_OUT>();
    for (size_t i = 0; i < outer; ++i) {
      for (size_t j = 0; j < ninputs; ++j) {
        const auto in_type = tensors[j].scalar_type();
        ET_SWITCH_REALHB_TYPES(in_type, ctx, "cat.out", CTYPE_IN, [&] {
          if (tensors[j].numel() == 0) {
            return;
          }
          size_t inner = tensors[j].size(dim) * dim_stride;
          const CTYPE_IN* const in_ptr =
              tensors[j].const_data_ptr<CTYPE_IN>() + i * inner;

          for (size_t k = 0; k < inner; ++k) {
            out_ptr[k] = static_cast<CTYPE_OUT>(in_ptr[k]);
          }
          out_ptr += inner;
        });
      }
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
