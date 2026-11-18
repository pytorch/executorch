/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace vkcompute {

// Values of the op_type specialization constant of the binary op and
// tensor-scalar shaders. Must match binary_op.glslh and binary_scalar.glsl.
enum class BinaryOpType : int32_t {
  kAdd = 0,
  kSub = 1,
  kMul = 2,
  kDiv = 3,
  kFloorDivide = 4,
  kPow = 5,
  kMinimum = 6,
  kSwiglu = 7,
  kEq = 8,
  kLt = 9,
  kLe = 10,
  kGt = 11,
  kGe = 12,
  kBitwiseAnd = 13,
  kBitwiseOr = 14,
  // Implemented by the tensor-scalar shader only.
  kNe = 15,
};

} // namespace vkcompute
