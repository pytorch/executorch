#include <executorch/backends/xnnpack/runtime/executor/shape_env.h>

namespace executorch::backends::xnnpack::executor {

using executorch::runtime::Span;

ShapeEnv::ShapeEnv(uint32_t num_symints) : bounds(num_symints) {}

runtime::Error ShapeEnv::specialize(
    Span<const graph::TensorSpec> specs,
    Span<core::Tensor> values) {
  for (auto& b : bounds) {
    b.min = 1;
    b.max = {};
  }

  if (specs.size() != values.size()) {
    return runtime::Error::InvalidArgument;
  }

  for (size_t i = 0; i < specs.size(); i++) {
    auto& spec = specs[i];
    auto& tensor = values[i];

    if (spec.sizes.size() != tensor.sizes.size()) {
      return runtime::Error::InvalidArgument;
    }

    for (size_t d = 0; d < spec.sizes.size(); d++) {
      auto& dim = spec.sizes[d];
      auto concrete = static_cast<int64_t>(tensor.sizes[d]);

      if (dim.is_constant()) {
        if (dim.offset != concrete) {
          return runtime::Error::InvalidArgument;
        }
        continue;
      }

      if (dim.coeffs.size() == 1 && dim.coeffs[0].coefficient == 1) {
        auto sym = dim.coeffs[0].sym;
        if (sym >= bounds.size()) {
          // Spec references a symint outside this env's range.
          return runtime::Error::Internal;
        }
        // Solve sym = concrete - offset. A dim is always >= 1, so a value
        // smaller than the offset means the concrete shape can't satisfy
        // the spec.
        int64_t solved_signed = concrete - dim.offset;
        if (solved_signed < 1) {
          return runtime::Error::InvalidArgument;
        }
        auto solved = static_cast<uint64_t>(solved_signed);
        auto& bound = bounds[sym];
        // `min` accumulates the largest solved value and `max` the smallest;
        // if any two occurrences disagree, min > max flags the contradiction.
        bound.min = std::max(bound.min, solved);
        bound.max = bound.max ? std::min(*bound.max, solved) : solved;
        if (bound.min > *bound.max) {
          return runtime::Error::InvalidArgument;
        }
        continue;
      }
    }
  }

  return runtime::Error::Ok;
}

} // namespace executorch::backends::xnnpack::executor
