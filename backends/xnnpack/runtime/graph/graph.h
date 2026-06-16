#pragma once

#include <executorch/backends/xnnpack/runtime/core/variant_util.h>
#include <executorch/backends/xnnpack/runtime/graph/handles.h>
#include <executorch/backends/xnnpack/runtime/graph/node.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>
#include <executorch/runtime/core/error.h>
#include <vector>

namespace executorch::backends::xnnpack::graph {

/*
 * Describes a computational graph.
 */
struct Graph {
  std::vector<TensorSpec> input_specs;
  std::vector<Node> nodes;
  std::vector<ValueHandle> outputs;

  /* Clean up nodes marked as dead. */
  [[nodiscard]] runtime::Error compact_nodes();

  /* Returns the number of symints referenced in the graph. */
  uint32_t symint_count() const;

  /* Regenerate user metadata on nodes. */
  void update_users();

  /* Retrieve the output specifier for a given node handle. */
  inline OutputSpec get_output_spec_for_node(NodeHandle node) const {
    return std::visit(
        overloaded{
            [&](const InputNode& n) -> OutputSpec {
              return input_specs.at(n.input);
            },
            [](const ConstantNode& n) -> OutputSpec {
              TensorSpec spec;
              spec.dtype = n.tensor->dtype;
              spec.sizes.reserve(n.tensor->sizes.size());
              for (auto s : n.tensor->sizes) {
                spec.sizes.push_back(
                    DimSizeSpec::constant(static_cast<int64_t>(s)));
              }
              spec.quant_params = n.quant_params;
              return spec;
            },
            [](const CallOperatorNode& n) -> OutputSpec {
              return n.output_specs;
            },
            [](const CallSubgraphNode& n) -> OutputSpec {
              return n.output_specs;
            },
        },
        nodes[node].value);
  }

  /* Retrieve the tensor spec for a given value handle. */
  inline TensorSpec get_tensor_spec(ValueHandle vh) const {
    auto spec = get_output_spec_for_node(vh.node);
    return std::visit(
        overloaded{
            [](const TensorSpec& s) -> TensorSpec { return s; },
            [&](const std::vector<TensorSpec>& v) -> TensorSpec {
              return v.at(vh.output);
            },
        },
        spec);
  }
};

} // namespace executorch::backends::xnnpack::graph
