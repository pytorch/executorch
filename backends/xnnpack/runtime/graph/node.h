#pragma once

#include <executorch/backends/xnnpack/runtime/core/variant_util.h>
#include <executorch/backends/xnnpack/runtime/graph/handles.h>
#include <executorch/backends/xnnpack/runtime/graph/operator.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>

#include <executorch/backends/xnnpack/runtime/core/tensor.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace executorch::backends::xnnpack::graph {

enum class NodeFlags : uint8_t {
  None = 0,
  Dead = (1 << 0), // This node is marked for deletion.
  UseXnnpack = (1 << 1), // Sub-delegate this node to XNNPACK.
  PassInternal1 = (1 << 2), // Pass-internal usage.
};

inline NodeFlags operator|(NodeFlags a, NodeFlags b) {
  return static_cast<NodeFlags>(
      static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

inline NodeFlags operator&(NodeFlags a, NodeFlags b) {
  return static_cast<NodeFlags>(
      static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

inline NodeFlags operator~(NodeFlags a) {
  return static_cast<NodeFlags>(~static_cast<uint8_t>(a));
}

inline NodeFlags& operator|=(NodeFlags& a, NodeFlags b) {
  a = a | b;
  return a;
}

inline NodeFlags& operator&=(NodeFlags& a, NodeFlags b) {
  a = a & b;
  return a;
}

using ConstantArg = std::variant<int64_t, double, std::vector<int64_t>>;
using OutputSpec = std::variant<TensorSpec, std::vector<TensorSpec>>;

struct Graph;

/*
 * A node that represents a call into an inner graph. This is primarily
 * used for delegation of subgraphs to run with XNNPACK.
 */
struct CallSubgraphNode {
  std::vector<ValueHandle> args;
  OutputSpec output_specs;
  std::unique_ptr<Graph> subgraph;
};

/*
 * A node that represents a top-level graph input.
 */
struct InputNode {
  InputHandle input;

  bool operator==(const InputNode& o) const {
    return input == o.input;
  }
};

/*
 * A node that represents a constant value.
 */
struct ConstantNode {
  std::shared_ptr<const core::Tensor> tensor;
  std::optional<core::QuantParams> quant_params;

  bool operator==(const ConstantNode& o) const {
    return tensor == o.tensor && quant_params == o.quant_params;
  }
};

/*
 * A node that represents the result of a operator invocation.
 */
struct CallOperatorNode {
  std::vector<ValueHandle> args;
  Operator op;
  OutputSpec output_specs;
  std::vector<ConstantArg> constant_args;

  // Fused output activation bounds (e.g. a ReLU/HardTanh folded into this op).
  // Defaults are a no-op clamp.
  float output_min = -INFINITY;
  float output_max = INFINITY;

  bool operator==(const CallOperatorNode& o) const {
    return args == o.args && op == o.op && output_specs == o.output_specs &&
        constant_args == o.constant_args && output_min == o.output_min &&
        output_max == o.output_max;
  }
};

using NodeVariant =
    std::variant<InputNode, ConstantNode, CallOperatorNode, CallSubgraphNode>;

/*
 * A
 */
struct Node {
  NodeVariant value;

  /* A list of nodes that directly consume the output of this node. */
  std::vector<NodeHandle> users;

  /*
   * An opaque field used internally by various passes. This field is not
   * guaranteed to be preserved across passes and should only be used as
   * internal, temporary state for graph transformations.
   */
  uint32_t tag = 0;

  NodeFlags flags = NodeFlags::None;

  /* Returns a list of value handles this node takes as inputs. */
  const std::vector<ValueHandle>& get_args() const {
    return std::visit(
        overloaded{
            [](const InputNode&) -> const std::vector<ValueHandle>& {
              static const std::vector<ValueHandle> empty;
              return empty;
            },
            [](const ConstantNode&) -> const std::vector<ValueHandle>& {
              static const std::vector<ValueHandle> empty;
              return empty;
            },
            [](const CallOperatorNode& n) -> const std::vector<ValueHandle>& {
              return n.args;
            },
            [](const CallSubgraphNode& n) -> const std::vector<ValueHandle>& {
              return n.args;
            },
        },
        value);
  }

  /* Returns the number of output values produced by this node. */
  uint32_t output_count() const {
    return std::visit(
        overloaded{
            [](const InputNode&) -> uint32_t { return 1; },
            [](const ConstantNode&) -> uint32_t { return 1; },
            [](const CallOperatorNode& n) -> uint32_t {
              return std::visit(
                  overloaded{
                      [](const TensorSpec&) -> uint32_t { return 1; },
                      [](const std::vector<TensorSpec>& v) -> uint32_t {
                        return static_cast<uint32_t>(v.size());
                      },
                  },
                  n.output_specs);
            },
            [](const CallSubgraphNode& n) -> uint32_t {
              return std::visit(
                  overloaded{
                      [](const TensorSpec&) -> uint32_t { return 1; },
                      [](const std::vector<TensorSpec>& v) -> uint32_t {
                        return static_cast<uint32_t>(v.size());
                      },
                  },
                  n.output_specs);
            },
        },
        value);
  }
};

} // namespace executorch::backends::xnnpack::graph
