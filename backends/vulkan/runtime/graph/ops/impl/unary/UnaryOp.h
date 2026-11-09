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

// Values of the op_type specialization constant of the unary op shaders. Must
// match unary_op.glslh.
enum class UnaryOpType : int32_t {
  kClamp = 0,
  kAbs = 1,
  kCos = 2,
  kExp = 3,
  kGelu = 4,
  kGeluErf = 5,
  kNeg = 6,
  kSigmoid = 7,
  kSin = 8,
  kSqrt = 9,
  kRsqrt = 10,
  kTanh = 11,
  kHardshrink = 12,
  kHardswish = 13,
  kHardsigmoid = 14,
  kLeakyRelu = 15,
  kLog10 = 16,
  kRound = 17,
  kTan = 18,
  kBitwiseNot = 19,
  kRelu = 20,
};

} // namespace vkcompute
