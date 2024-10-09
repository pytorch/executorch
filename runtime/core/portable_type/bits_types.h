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
namespace etensor {

/**
 * bits1x8 is an uninterpreted dtype of a tensor with 1 bit (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits1x8 {
  using underlying = uint8_t;
  uint8_t val_;
  bits1x8() = default;
  explicit bits1x8(uint8_t val) : val_(val) {}
};

/**
 * bits2x4 is an uninterpreted dtype of a tensor with 2 bits (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits2x4 {
  using underlying = uint8_t;
  uint8_t val_;
  bits2x4() = default;
  explicit bits2x4(uint8_t val) : val_(val) {}
};

/**
 * bits4x2 is an uninterpreted dtype of a tensor with 4 bits (packed to byte
 * boundary), without any semantics defined.
 */
struct alignas(1) bits4x2 {
  using underlying = uint8_t;
  uint8_t val_;
  bits4x2() = default;
  explicit bits4x2(uint8_t val) : val_(val) {}
};

/**
 * bits8 is an uninterpreted dtype of a tensor with 8 bits, without any
 * semantics defined.
 */
struct alignas(1) bits8 {
  uint8_t val_;
  bits8() = default;
  explicit bits8(uint8_t val) : val_(val) {}
};

/**
 * bits16 is an uninterpreted dtype of a tensor with 16 bits, without any
 * semantics defined.
 */
struct alignas(2) bits16 {
  uint16_t val_;
  bits16() = default;
  explicit bits16(uint16_t val) : val_(val) {}
};

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::bits16;
using ::executorch::runtime::etensor::bits1x8;
using ::executorch::runtime::etensor::bits2x4;
using ::executorch::runtime::etensor::bits4x2;
using ::executorch::runtime::etensor::bits8;
} // namespace executor
} // namespace torch
