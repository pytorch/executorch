/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/kernels/optimized/vec/vec.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

namespace {

/**
 * Fast path of natural exponential function. When no casting is required, CPU
 * vector intrinsics can be used.
 */
template <
    typename CTYPE_IN,
    typename CTYPE_OUT,
    typename std::enable_if<std::is_same<CTYPE_IN, CTYPE_OUT>::value, int>::
        type = 0>
void exp_data(
    const CTYPE_IN* in_data,
    const size_t numel,
    CTYPE_OUT* out_data) {
  using Vec = executorch::vec::Vectorized<CTYPE_IN>;
  executorch::vec::map<CTYPE_IN>(
      [](Vec x) { return x.exp(); }, out_data, in_data, numel);
}

/**
 * Slow path of natural exponential function.
 */
template <
    typename CTYPE_IN,
    typename CTYPE_OUT,
    typename std::enable_if<!std::is_same<CTYPE_IN, CTYPE_OUT>::value, int>::
        type = 0>
void exp_data(
    const CTYPE_IN* in_data,
    const size_t numel,
    CTYPE_OUT* out_data) {
  for (size_t i = 0; i < numel; i++) {
    CTYPE_OUT xi = static_cast<CTYPE_OUT>(in_data[i]);
    out_data[i] = std::exp(xi);
  }
}

} // namespace

Tensor& opt_exp_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  auto error = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize output tensor.");

  ET_SWITCH_REAL_TYPES_AND(
      Bool, in.scalar_type(), ctx, "exp.out", CTYPE_IN, [&] {
        ET_SWITCH_FLOAT_TYPES(
            out.scalar_type(), ctx, "exp.out", CTYPE_OUT, [&] {
              exp_data<CTYPE_IN, CTYPE_OUT>(
                  in.const_data_ptr<CTYPE_IN>(),
                  in.numel(),
                  out.mutable_data_ptr<CTYPE_OUT>());
            });
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
