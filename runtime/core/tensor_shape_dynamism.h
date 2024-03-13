/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace torch {
namespace executor {

/**
 * Rank can never change in executorch. But shape sometimes can.
 * This enum is used to help provide better safety in kernels on what
 * tensors are resizable
 *
 * WARNING: This abstraction is only temporary. The long term vision is that
 * this wont exist in the runtime. Instead the runtime will support a debug mode
 * that allows for similar fail early patterns without having to pay the runtime
 * cost of directly embedding this abstraction in tensor and performing checks
 * against it during resizing. Adding this debug mode is non trivial though so
 * for the short term this abstraction helps us move fast. TODO(jakeszwe):
 * T134528146
 */
enum class TensorShapeDynamism : uint8_t {
  /// Cannot change shape
  STATIC = 0,
  /// shape cannot exceed initial capacity
  DYNAMIC_BOUND = 1,
  /// No restriction on shape and capacity
  DYNAMIC_UNBOUND = 2,
};

} // namespace executor
} // namespace torch
