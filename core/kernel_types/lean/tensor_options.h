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
   * This is the only format supported by Executorch. Use dim orders to
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
   * This is the only layout supported by Executorch.
   */
  Strided,
};
} // namespace executor
} // namespace torch
