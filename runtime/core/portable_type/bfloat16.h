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
 * The "brain floating-point" type, compatible with c10/util/BFloat16.h from
 * pytorch core.
 *
 * This representation uses 1 bit for the sign, 8 bits for the exponent and 7
 * bits for the mantissa.
 */
struct alignas(2) BFloat16 {
  uint16_t x;
};

} // namespace executor
} // namespace torch
