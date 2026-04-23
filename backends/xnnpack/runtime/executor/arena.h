#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace executorch::backends::xnnpack::executor {

struct Arena {
    std::unique_ptr<uint8_t[]> buffer;
    size_t size = 0;

    inline void* data() { return buffer.get(); }
    void resize(size_t new_size);
};

}
