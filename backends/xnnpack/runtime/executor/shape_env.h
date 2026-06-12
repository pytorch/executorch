#pragma once

#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/span.h>

#include <cstdint>
#include <optional>

namespace executorch::backends::xnnpack::executor {

/*
 * Specifies the lower and optional upper bounds for a shape value.
 * When max is empty, the value can be arbitrarily large.
 */
struct ShapeBound {
  uint64_t min = 1;
  std::optional<uint64_t> max = {};
};

/*
 * Tracks symint values and provides logic to specialize symints
 * based on concrete inputs. This class implements a restricted
 * subset of the PyTorch ShapeEnv logic.
 */
struct ShapeEnv {
  /*
   * Symint bounds, solved from concrete inputs by specialize(). For
   * example, if an input tensor is [1, s0] and given concrete
   * shape [1, 10], then s0 is known to be 10.
   */
  std::vector<ShapeBound> bounds;

  ShapeEnv() = default;
  ShapeEnv(uint32_t num_symints);

  /*
   * Specialize the bounds for a given set of concrete tensors, solving for the
   * symints that appear in the specs. Each call resets the bounds.
   */
  runtime::Error specialize(
      runtime::Span<const graph::TensorSpec> specs,
      runtime::Span<core::Tensor> values);
};

} // namespace executorch::backends::xnnpack::executor
