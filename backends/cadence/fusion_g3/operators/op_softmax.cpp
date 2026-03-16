/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fusion_g3/operators/operators.h>

#include <cmath>

#include <xa_nnlib_kernels_api.h>

#include <executorch/backends/cadence/common/xt_macros.h>
#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using ::executorch::aten::ArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;

namespace impl {
namespace G3 {
namespace native {

Tensor& _softmax_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  (void)ctx;

  // Adjust for negative dim
  dim = dim < 0 ? dim + executorch::runtime::nonzero_dim(in) : dim;
#ifdef OP_ARG_CHECK
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensors_have_same_dim_order(in, out),
      InvalidArgument,
      out);
#endif

  int inp_shapes[in.dim()];
  const ArrayRef<Tensor::SizesType> in_size = in.sizes();
  for (int i = 0; i < in.dim(); i++) {
    inp_shapes[i] = in_size[i];
  }

  if ((in.scalar_type() == ScalarType::Float) &&
      (out.scalar_type() == ScalarType::Float)) {
    const float* const inp_data = in.const_data_ptr<float>();
    float* const out_data = out.mutable_data_ptr<float>();
    int axis = dim;
    XT_KERNEL_CHECK(
        ctx,
        out,
        xa_nn_softmax_f32_f32,
        out_data,
        inp_data,
        inp_shapes,
        in.dim(),
        &axis);
  } else {
    ET_KERNEL_CHECK(
        ctx,
        torch::executor::check_softmax_args(in, dim, half_to_float, out),
        InvalidArgument,
        out);

    ET_SWITCH_FLOATH_TYPES(in.scalar_type(), ctx, "_softmax.out", CTYPE, [&]() {
      const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
      CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

      torch::executor::apply_over_dim(
          [in_data, out_data](
              const size_t size, const size_t stride, const size_t base) {
            // calculate max in softmax dim. During softmax computation each
            // value is subtracted by the maximum in value before calling exp
            // to preserve numerical stability.
            const CTYPE max_in = torch::executor::apply_unary_reduce_fn(
                [](const CTYPE val_in, CTYPE val_accum) {
                  return std::max(val_in, val_accum);
                },
                in_data + base,
                size,
                stride);

            const CTYPE temp_sum =
                torch::executor::apply_unary_map_reduce_fn<CTYPE, CTYPE>(
                    [max_in](const CTYPE val_in) {
                      return std::exp(val_in - max_in);
                    },
                    [](const CTYPE mapped_in, CTYPE val_accum) {
                      return val_accum + mapped_in;
                    },
                    in_data + base,
                    size,
                    stride);

            torch::executor::apply_unary_map_fn(
                [max_in, temp_sum](const CTYPE val_in) {
                  return std::exp(val_in - max_in) / temp_sum;
                },
                in_data + base,
                out_data + base,
                size,
                stride);
          },
          in,
          dim);
    });
  }

  return out;
}

} // namespace native
} // namespace G3
} // namespace impl
