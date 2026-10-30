// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>

namespace vkcompute::shader_profiling::reference {
static Tensor clone_reference(const Json& spec, const TensorMap& tensors, size_t) {
  return tensor_arg(spec, 0, tensors);
}
REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(clone, clone_reference);
  REGISTER_OPERATOR_REFERENCE(aten.clone.default, clone);
}
} // namespace vkcompute::shader_profiling::reference
