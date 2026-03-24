#pragma once

#include <executorch/backends/xnnpack/runtime/core/span.h>
#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>

#include <cstdint>
#include <optional>

namespace executorch::backends::xnnpack::executor {

struct ShapeBound {
    uint64_t min = 1;
    std::optional<uint64_t> max = {};
};

struct ShapeEnv {
    std::vector<ShapeBound> specialized_bounds;
    std::vector<ShapeBound> unspecialized_bounds;

    ShapeEnv() = default;
    ShapeEnv(uint32_t num_symints);

    bool specialize(core::Span<const graph::TensorSpec> specs, core::Span<core::Tensor> values);
};

}
