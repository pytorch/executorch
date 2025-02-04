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
 * qint8 is for signed 8 bit quantized Tensors
 */
struct alignas(1) qint8 {
  using underlying = int8_t;
  int8_t val_;
  qint8() = default;
  explicit qint8(int8_t val) : val_(val) {}
};

/**
 * quint8 is for unsigned 8 bit quantized Tensors
 */
struct alignas(1) quint8 {
  using underlying = uint8_t;
  uint8_t val_;
  quint8() = default;
  explicit quint8(uint8_t val) : val_(val) {}
};

/**
 * qint32 is for signed 32 bit quantized Tensors
 */
struct alignas(4) qint32 {
  using underlying = int32_t;
  int32_t val_;
  qint32() = default;
  explicit qint32(int32_t val) : val_(val) {}
};

/**
 * quint4x2 is for un-signed 4 bit quantized Tensors that are packed to byte
 * boundary.
 */
struct alignas(1) quint4x2 {
  using underlying = uint8_t;
  uint8_t val_;
  quint4x2() = default;
  explicit quint4x2(uint8_t val) : val_(val) {}
};

/**
 * quint2x4 is for un-signed 2 bit quantized Tensors that are packed to byte
 * boundary.
 */
struct alignas(1) quint2x4 {
  using underlying = uint8_t;
  uint8_t val_;
  quint2x4() = default;
  explicit quint2x4(uint8_t val) : val_(val) {}
};

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::qint32;
using ::executorch::runtime::etensor::qint8;
using ::executorch::runtime::etensor::quint2x4;
using ::executorch::runtime::etensor::quint4x2;
using ::executorch::runtime::etensor::quint8;
} // namespace executor
} // namespace torch
