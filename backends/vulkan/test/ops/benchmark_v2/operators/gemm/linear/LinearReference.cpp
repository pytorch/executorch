// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/GemmReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/linear/LinearReference.h>

namespace vkcompute::shader_profiling::reference {

Tensor
linear_reference(const Json& spec, const TensorMap& tensors, size_t threads) {
  const Tensor* bias =
      spec.at("args").at(2).is_null() ? nullptr : &tensor_arg(spec, 2, tensors);
  return linear_projection(
      tensor_arg(spec, 0, tensors),
      tensor_arg(spec, 1, tensors),
      bias,
      threads);
}

REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(linear, linear_reference);
  REGISTER_OPERATOR_REFERENCE(aten.linear.default, linear);
}

} // namespace vkcompute::shader_profiling::reference
