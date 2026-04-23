#include <executorch/backends/xnnpack/runtime/executor/arena.h>

namespace executorch::backends::xnnpack::executor {

void Arena::resize(size_t new_size) {
    if (new_size <= size) { return; }
    buffer = std::make_unique<uint8_t[]>(new_size);
    size = new_size;
}

}
