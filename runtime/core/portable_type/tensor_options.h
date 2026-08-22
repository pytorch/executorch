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
 * Tensor data memory formats supported by ExecuTorch. This concept only exists
 * for compatibility with ATen; use dim_order to describe non-contiguous
 * layouts.
 */
enum class MemoryFormat : int8_t {
  /**
   * Row-major contiguous data.
   */
  Contiguous = 0,
  /**
   * Output tensor format should remain the same as the input tensor format.
   * E.g. if the input tensor is in channels_last format, operator output
   * should be in channels_last format.
   */
  Preserve = 1,
};

/**
 * Tensor data memory layout. This concept only exists for compatibility
 * with ATen.
 */
enum class Layout : int8_t {
  /**
   * The tensor occupies memory densely and indexing is managed through strides.
   * Contrasted with a sparse tensor layout where the memory structure of the
   * data blob will be more complicated and indexing requires larger structures.
   *
   * This is the only layout supported by ExecuTorch.
   */
  Strided = 0,
};
} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::Layout;
using ::executorch::runtime::etensor::MemoryFormat;
} // namespace executor
} // namespace torch
