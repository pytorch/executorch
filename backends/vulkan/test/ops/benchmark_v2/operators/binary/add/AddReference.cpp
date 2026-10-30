// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/binary/add/AddReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/binary/BinaryReference.h>

namespace vkcompute::shader_profiling::reference {
Tensor add_tensor_reference(
    const Json& spec,
    const TensorMap& tensors,
    size_t threads) {
  Json shader_spec = spec;
  shader_spec["bindings"] = {
      spec.at("args").back(), spec.at("args").at(0), spec.at("args").at(1)};
  const Json& alpha = spec.at("args").at(2);
  const double alpha_value = alpha.is_null() ? 1.0
      : alpha.contains("float")              ? alpha.at("float").get<double>()
                                             : alpha.at("int").get<double>();
  shader_spec["push_constants"] = {{{"data", {alpha_value}}}};
  return binary_reference(shader_spec, tensors, "add", threads);
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(add_tensor, add_tensor_reference);
  REGISTER_OPERATOR_REFERENCE(aten.add.Tensor, add_tensor);
}
} // namespace vkcompute::shader_profiling::reference
