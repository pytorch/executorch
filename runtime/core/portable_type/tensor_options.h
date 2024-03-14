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
 * Tensor data memory format. This concept only exists for compatibility
 * with ATen.
 */
enum class MemoryFormat : int8_t {
  /**
   * Row-major contiguous data format.
   *
   * This is the only format supported by ExecuTorch. Use dim orders to
   * describe other layouts.
   */
  Contiguous,
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
  Strided,
};
} // namespace executor
} // namespace torch
