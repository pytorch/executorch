#include <executorch/backends/xnnpack/runtime/executor/shape_env.h>

namespace executorch::backends::xnnpack::executor {

ShapeEnv::ShapeEnv(uint32_t num_symints) : specialized_bounds(num_symints), unspecialized_bounds(num_symints) {

}

bool ShapeEnv::specialize(core::Span<const graph::TensorSpec> specs, core::Span<core::Tensor> values) {
    for (auto& b : specialized_bounds) {
        b.min = 1;
        b.max = {};
    }

    if (specs.size() != values.size()) { return false; }

    for (size_t i = 0; i < specs.size(); i++) {
        auto& spec = specs[i];
        auto& tensor = values[i];

        if (spec.sizes.size() != tensor.sizes.size()) { return false; }

        for (size_t d = 0; d < spec.sizes.size(); d++) {
            auto& dim = spec.sizes[d];
            auto concrete = static_cast<int64_t>(tensor.sizes[d]);

            if (dim.is_constant()) {
                if (dim.offset != concrete) { return false; }
                continue;
            }

            if (dim.coeffs.size() == 1 && dim.coeffs[0].coefficient == 1) {
                auto sym = dim.coeffs[0].sym;
                auto solved = static_cast<uint64_t>(concrete - dim.offset);
                auto& bound = specialized_bounds[sym];
                bound.min = std::max(bound.min, solved);
                bound.max = bound.max ? std::min(*bound.max, solved) : solved;
                if (bound.min > *bound.max) { return false; }
                continue;
            }
        }
    }

    return true;
}

}
