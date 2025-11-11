/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch {
namespace runtime {

/**
 * The resizing capabilities of a Tensor.
 *
 * The rank of an ExecuTorch Tensors can never change, but shape sometimes can.
 */
enum class TensorShapeDynamism : uint8_t {
  /// Cannot change shape.
  STATIC = 0,
  /// Shape cannot exceed initial capacity.
  DYNAMIC_BOUND = 1,
  /// No restriction on shape and capacity.
  DYNAMIC_UNBOUND = 2,
};

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::TensorShapeDynamism;
} // namespace executor
} // namespace torch
