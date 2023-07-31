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
 * A half-precision floating point type, compatible with c10/util/Half.h from
 * pytorch core.
 */
struct alignas(2) Half {
  uint16_t x;
};

} // namespace executor
} // namespace torch
