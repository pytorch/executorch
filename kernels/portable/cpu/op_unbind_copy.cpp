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
 * unbind_copy.int_out(Tensor input, int dim=0, *, Tensor(a!)[] out) -> ()
 */
void unbind_copy_int_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    int64_t dim,
    TensorList out) {
  (void)ctx;
  // Support python-style negative indexing.
  if (dim < 0) {
    dim += input.dim();
  }

  ET_KERNEL_CHECK(
      ctx, check_unbind_copy_args(input, dim, out), InvalidArgument, );

  for (int i = 0; i < out.size(); ++i) {
    ET_KERNEL_CHECK(
        ctx, tensors_have_same_dim_order(input, out[i]), InvalidArgument, );
  }

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(input), InvalidArgument, );

  if (input.numel() == 0) {
    return;
  }

  const size_t leading_dims = getLeadingDims(input, dim);
  const size_t trailing_dims = getTrailingDims(input, dim);
  const size_t step = input.size(dim) * trailing_dims;

  ScalarType in_type = input.scalar_type();
  ScalarType out_type = out[0].scalar_type();

  ET_SWITCH_REAL_TYPES_AND(
      Bool, in_type, ctx, "unbind_copy.int_out", CTYPE_IN, [&]() {
        ET_SWITCH_REAL_TYPES_AND(
            Bool, out_type, ctx, "unbind_copy.int_out", CTYPE_OUT, [&]() {
              const CTYPE_IN* const input_data =
                  input.const_data_ptr<CTYPE_IN>();
              for (size_t i = 0, e = out.size(); i < e; ++i) {
                size_t input_offset = i * trailing_dims;
                CTYPE_OUT* const dest = out[i].mutable_data_ptr<CTYPE_OUT>();
                size_t dest_offset = 0;
                for (size_t j = 0; j < leading_dims; ++j) {
                  for (size_t k = 0; k < trailing_dims; ++k) {
                    dest[dest_offset + k] = convert<CTYPE_OUT, CTYPE_IN>(
                        input_data[input_offset + k]);
                  }
                  input_offset += step;
                  dest_offset += trailing_dims;
                }
              }
            });
      });
}

} // namespace native
} // namespace executor
} // namespace torch
