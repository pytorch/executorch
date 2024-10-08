/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using SizesType = exec_aten::SizesType;

namespace {

void increment_index(size_t* index, const ArrayRef<SizesType> sizes) {
  for (ssize_t i = sizes.size() - 1; i >= 0; --i) {
    index[i]++;
    if (index[i] == sizes[i]) {
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
  for (size_t i = 0; i < lim; ++i) {
    if (in_data[i] != 0) {
      num_nonzero++;
    }
  }

  // resize out
  SizesType out_shape[2] = {
      static_cast<SizesType>(num_nonzero), static_cast<SizesType>(input.dim())};
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(output, ArrayRef<exec_aten::SizesType>(out_shape, 2)) ==
          Error::Ok,
      InvalidArgument, );

  size_t index[kTensorDimensionLimit];
  memset(index, 0, sizeof(index));

  int64_t* out_data = output.mutable_data_ptr<int64_t>();
  size_t out_idx = 0;

  // Loop again and this time write the proper indices into out
  for (size_t i = 0; i < lim; i++) {
    if (in_data[i] != 0) {
      for (size_t j = 0; j < input.dim(); j++) {
        out_data[out_idx++] = index[j];
      }
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

  ET_SWITCH_REAL_TYPES_AND(
      Bool, in.scalar_type(), ctx, "nonzero.out", CTYPE, [&] {
        nonzero<CTYPE>(ctx, in, out);
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
