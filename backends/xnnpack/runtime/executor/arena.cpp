#include <executorch/backends/xnnpack/runtime/executor/arena.h>

namespace executorch::backends::xnnpack::executor {

runtime::Error Arena::resize(size_t new_size) {
  if (new_size <= size) {
    return runtime::Error::Ok;
  }
  auto* new_buffer = new (std::nothrow) uint8_t[new_size];
  if (new_buffer == nullptr) {
    return runtime::Error::MemoryAllocationFailed;
  }
  buffer.reset(new_buffer);
  size = new_size;
  return runtime::Error::Ok;
}

} // namespace executorch::backends::xnnpack::executor
