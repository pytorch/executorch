/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

/**
 * @file
 *
 * NOTE: This file is deprecated: no new code should be added to it, and its
 * contents should be split into per-operator files like op_add.cpp.
 */

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

template <typename T>
using optional = exec_aten::optional<T>;

// kernel for demonstration purpose only

// Kernel implementation provided by user.
// The schema is added by user to PyTorch native function DSL in a yaml file,
// defined in
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/README.md
// @lint-ignore-every CLANGTIDY

namespace {
bool check_linear_scratch_example_args(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias,
    Tensor& out,
    Tensor& scratch) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      input.size(1) == weight.size(1), "Unexpected weight size 1");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      scratch.size(0) == input.size(0), "Unexpected scratch size 0");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      scratch.size(1) == weight.size(0), "Unexpected scratch size 1");

  return true;
}
} // namespace

/*
 * A simple example of using scratch tensor. In this specific case we could also
 * update the out tensor in place to avoid the scratch tensor.
 *
 * linear.scratch_example(Tensor input, Tensor weight, Tensor? bias=None, *,
 *     Tensor(a!) out, Tensor(b!) _scratch_tensor) -> Tensor(a!)
 */
Tensor& linear_scratch_example(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias,
    Tensor& out,
    Tensor& scratch) {
  size_t M, N, K;
  M = input.size(0);
  N = input.size(1);
  K = weight.size(0);

  // TODO: Update to use ET_KERNEL_CHECK when context is available in custom
  // ops.
  ET_CHECK(
      check_linear_scratch_example_args(input, weight, bias, out, scratch));

  // input @ weight -> scratch
  // TODO: does not handle the case that accumulator has different type
  // as input
  // TODO: this is just some inefficient implementation to verify correctness
  if (input.scalar_type() == ScalarType::Float) {
    // only support float32 before D35829540 is landed
    using scalar_t = float;
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < K; ++j) {
        scalar_t* scratch_ptr =
            scratch.mutable_data_ptr<scalar_t>() + (i * K + j);
        *scratch_ptr = 0;
        for (size_t k = 0; k < N; ++k) {
          const scalar_t* const input_ptr =
              input.const_data_ptr<scalar_t>() + (i * N + k);
          // note it's transposed
          // (j,k) element in the (K, N) array
          const scalar_t* const weight_ptr =
              weight.const_data_ptr<scalar_t>() + (j * N + k);
          *scratch_ptr += *input_ptr * *weight_ptr;
        }
      }
    }

    // add the bias
    if (bias.has_value()) {
      ET_CHECK_MSG(K == bias.value().numel(), "Unexpected numel for bias");
      for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
          scalar_t* scratch_ptr =
              scratch.mutable_data_ptr<scalar_t>() + (i * K + j);
          scalar_t* out_ptr = out.mutable_data_ptr<scalar_t>() + (i * K + j);
          scalar_t* bias_ptr = bias.value().mutable_data_ptr<scalar_t>() + j;
          *out_ptr = *scratch_ptr + *bias_ptr;
        }
      }
    }
  }
  return out;
}

Tensor& linear_scratch_example(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias,
    Tensor& out,
    Tensor& scratch) {
  // TODO(larryliu): Add a context arg to the real op function and remove this
  // wrapper
  (void)ctx;
  return linear_scratch_example(input, weight, bias, out, scratch);
}

} // namespace native
} // namespace executor
} // namespace torch
