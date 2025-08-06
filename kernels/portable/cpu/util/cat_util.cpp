/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/cat_util.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>

namespace torch::executor::native {

bool check_cat_args(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
  return torch::executor::check_cat_args(tensors, dim, out);
}

void get_cat_out_target_size(
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    executorch::aten::SizesType* out_sizes,
    size_t* out_ndim) {
  torch::executor::get_cat_out_target_size(tensors, dim, out_sizes, out_ndim);
}

Tensor& cat_out_impl(
    KernelRuntimeContext& ctx,
    executorch::aten::ArrayRef<Tensor> tensors,
    int64_t dim,
    Tensor& out) {
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
  const bool out_is_complex =
      executorch::runtime::isComplexType(out.scalar_type());

  if (out_is_complex) {
    // TODO: The current support for complex dtype enforces that input and
    // output tensors have the same dtype. Support mixed dtypes in the future.
    for (size_t i = 0; i < ninputs; ++i) {
      const auto in_type = tensors[i].scalar_type();
      ET_KERNEL_CHECK(ctx, out_type == in_type, InvalidArgument, out);
    }
    ET_SWITCH_COMPLEXH_TYPES(out_type, ctx, "cat.out", CTYPE, [&] {
      CTYPE* out_ptr = out.mutable_data_ptr<CTYPE>();
      for (size_t i = 0; i < outer; ++i) {
        for (size_t j = 0; j < ninputs; ++j) {
          if (tensors[j].numel() == 0) {
            return;
          }
          size_t inner = tensors[j].size(dim) * dim_stride;
          const CTYPE* const in_ptr =
              tensors[j].const_data_ptr<CTYPE>() + i * inner;
          memcpy(out_ptr, in_ptr, inner * sizeof(CTYPE));
          out_ptr += inner;
        }
      }
    });
  } else {
    ET_SWITCH_REALHBBF16_TYPES(out_type, ctx, "cat.out", CTYPE_OUT, [&] {
      CTYPE_OUT* out_ptr = out.mutable_data_ptr<CTYPE_OUT>();
      for (size_t i = 0; i < outer; ++i) {
        for (size_t j = 0; j < ninputs; ++j) {
          const auto in_type = tensors[j].scalar_type();
          ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, "cat.out", CTYPE_IN, [&] {
            if (tensors[j].numel() == 0) {
              return;
            }
            size_t inner = tensors[j].size(dim) * dim_stride;
            const CTYPE_IN* const in_ptr =
                tensors[j].const_data_ptr<CTYPE_IN>() + i * inner;

            if (sizeof(CTYPE_IN) == sizeof(CTYPE_OUT)) {
              memcpy(out_ptr, in_ptr, inner * sizeof(CTYPE_IN));
            } else {
              for (size_t k = 0; k < inner; ++k) {
                out_ptr[k] = static_cast<CTYPE_OUT>(in_ptr[k]);
              }
            }
            out_ptr += inner;
          });
        }
      }
    });
  }
  return out;
}
} // namespace torch::executor::native
