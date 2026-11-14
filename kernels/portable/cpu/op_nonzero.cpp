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

#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using SizesType = executorch::aten::SizesType;

namespace {

void increment_index(size_t* index, const ArrayRef<SizesType> sizes) {
  for (ssize_t i = sizes.size() - 1; i >= 0; --i) {
    index[i]++;
    if (static_cast<ssize_t>(index[i]) == sizes[i]) {
      index[i] = 0;
    } else {
      return;
    }
  }
}

/**
 * Two pass algorithm where we first count the number of non zeros, then resize
 * out to the appropriate size, and then loop again and properly write into out
 */
template <typename CTYPE>
void nonzero(KernelRuntimeContext& ctx, const Tensor& input, Tensor& output) {
  const CTYPE* in_data = input.const_data_ptr<CTYPE>();
  size_t lim = input.numel();
  int32_t num_nonzero = 0;

  // Count number of non zeros
  for (const auto i : c10::irange(lim)) {
    if (in_data[i] != 0) {
      num_nonzero++;
    }
  }

  // resize out
  SizesType out_shape[2] = {
      static_cast<SizesType>(num_nonzero), static_cast<SizesType>(input.dim())};
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(
          output, ArrayRef<executorch::aten::SizesType>(out_shape, 2)) ==
          Error::Ok,
      InvalidArgument, );

  size_t index[kTensorDimensionLimit];
  memset(index, 0, sizeof(index));

  int64_t* out_data = output.mutable_data_ptr<int64_t>();
  // ATen returns nonzero column by column, and exported programs keep that.
  const auto row_stride = output.strides()[0];
  const auto col_stride = output.strides()[1];
  // The input can be in another dim order too, like the result of nonzero.
  const auto in_strides = input.strides();
  const bool in_contiguous = is_contiguous_dim_order(
      input.dim_order().data(), input.dim_order().size());
  size_t out_row = 0;

  // Loop again and this time write the proper indices into out
  for (const auto i : c10::irange(lim)) {
    size_t in_index = i;
    if (!in_contiguous) {
      in_index = 0;
      for (const auto j : c10::irange(input.dim())) {
        in_index += index[j] * in_strides[j];
      }
    }
    if (in_data[in_index] != 0) {
      for (const auto j : c10::irange(input.dim())) {
        out_data[out_row * row_stride + j * col_stride] = index[j];
      }
      out_row++;
    }
    increment_index(index, input.sizes());
  }
}

} // namespace

/**
 * Determines the non zero indices of input.
 * Out is a 2-D tensor where every row is a non zero index of the input.
 */
Tensor& nonzero_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(ctx, check_nonzero_args(in, out), InvalidArgument, out);

  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, "nonzero.out", CTYPE, [&] {
    nonzero<CTYPE>(ctx, in, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
