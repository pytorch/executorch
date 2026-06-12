#pragma once

#include <cstdint>
#include <vector>

namespace executorch::backends::xnnpack::graph {

using InputHandle = uint32_t;
using NodeHandle = uint32_t;
using OutputHandle = uint32_t;
using SymIntHandle = uint32_t;

struct ValueHandle {
  uint32_t node;
  uint32_t output = 0;

  bool operator==(const ValueHandle& o) const {
    return node == o.node && output == o.output;
  }

  static constexpr ValueHandle null() {
    return {UINT32_MAX, UINT32_MAX};
  }
  bool is_null() const {
    return node == UINT32_MAX && output == UINT32_MAX;
  }
};

using ValueHandles = std::vector<ValueHandle>;

} // namespace executorch::backends::xnnpack::graph
