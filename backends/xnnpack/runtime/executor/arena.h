#pragma once

#include <executorch/runtime/core/error.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace executorch::backends::xnnpack::executor {

/*
 * Provides a block of growable, contiguous memory.
 */
struct Arena {
  std::unique_ptr<uint8_t[]> buffer;
  size_t size = 0;

  inline void* data() {
    return buffer.get();
  }

  // Grows the arena to at least `new_size` bytes. The arena is
  // never shrunk. Re-allocation does not preserve existing contents.
  // On allocation failure the arena is left unchanged.
  runtime::Error resize(size_t new_size);
};

} // namespace executorch::backends::xnnpack::executor
