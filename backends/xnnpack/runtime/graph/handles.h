#pragma once

#include <cstdint>
#include <vector>

namespace executorch::backends::xnnpack::graph {

using InputHandle = uint32_t;
using NodeHandle = uint32_t;
using OutputHandle = uint32_t;
using SymIntHandle = uint32_t;

struct ValueHandle {
    uint16_t node;
    uint16_t output = 0;

    bool operator==(const ValueHandle& o) const {
        return node == o.node && output == o.output;
    }

    static constexpr ValueHandle null() {
        return { UINT16_MAX, UINT16_MAX };
    }
    bool is_null() const { return node == UINT16_MAX && output == UINT16_MAX; }
};

using ValueHandles = std::vector<ValueHandle>;

}
