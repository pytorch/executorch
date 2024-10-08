/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using TensorList = exec_aten::TensorList;

/**
 * Splits the tensor into chunks of size `split_size` along the specified
 * dimension.
 *
 * The last chunk will be smaller if the tensor size along the given dimension
 * dim is not evenly divisible by `split_size`.
 *
 * split_copy.Tensor_out(Tensor input, int split_size, int dim=0, *,
 * Tensor(a!)[] out) -> ()
 */
void split_copy_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    int64_t split_size,
    int64_t dim,
    TensorList out) {
  (void)ctx;
  // Support python-style negative indexing.
  if (dim < 0) {
    dim += input.dim();
  }

  ET_KERNEL_CHECK(
      ctx,
      check_split_copy_args(input, split_size, dim, out),
      InvalidArgument, );

  for (size_t i = 0; i < out.size(); ++i) {
    ET_KERNEL_CHECK(
        ctx, tensors_have_same_dim_order(input, out[i]), InvalidArgument, );
  }

  const size_t leading_dims = getLeadingDims(input, dim);
  const size_t trailing_dims = getTrailingDims(input, dim);
  const size_t step = input.size(dim) * trailing_dims;

  ScalarType in_type = input.scalar_type();
  ScalarType out_type = out[0].scalar_type();

  ET_SWITCH_REAL_TYPES_AND(
      Bool, in_type, ctx, "split_copy.Tensor_out", CTYPE_IN, [&]() {
        ET_SWITCH_REAL_TYPES_AND(
            Bool, out_type, ctx, "split_copy.Tensor_out", CTYPE_OUT, [&]() {
              const CTYPE_IN* input_data = input.const_data_ptr<CTYPE_IN>();
              for (size_t i = 0, e = out.size(); i < e; ++i) {
                size_t out_step = out[i].size(dim) * trailing_dims;
                if (out_step == 0) {
                  continue;
                }
                const CTYPE_IN* src = input_data;
                CTYPE_OUT* dest = out[i].mutable_data_ptr<CTYPE_OUT>();
                for (size_t j = 0; j < leading_dims; ++j) {
                  for (size_t k = 0; k < out_step; ++k) {
                    dest[k] = convert<CTYPE_OUT, CTYPE_IN>(src[k]);
                  }
                  src += step;
                  dest += out_step;
                }
                input_data += out_step;
              }
            });
      });
}

} // namespace native
} // namespace executor
} // namespace torch
