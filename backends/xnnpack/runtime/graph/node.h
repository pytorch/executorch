#pragma once

#include <executorch/backends/xnnpack/runtime/core/variant_util.h>
#include <executorch/backends/xnnpack/runtime/graph/handles.h>
#include <executorch/backends/xnnpack/runtime/graph/operator.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>

#include <executorch/backends/xnnpack/runtime/core/tensor.h>

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace executorch::backends::xnnpack::graph {

enum class NodeFlags : uint8_t {
    None = 0,
    UseXnnpack = (1 << 1),
    PassInternal1 = (1 << 2),
    Dead = (1 << 3),
};

inline NodeFlags operator|(NodeFlags a, NodeFlags b) {
    return static_cast<NodeFlags>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

inline NodeFlags operator&(NodeFlags a, NodeFlags b) {
    return static_cast<NodeFlags>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
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

struct CallSubgraphNode {
    std::vector<ValueHandle> args;
    OutputSpec output_specs;
    std::unique_ptr<Graph> subgraph;
};

struct InputNode {
    InputHandle input;

    bool operator==(const InputNode& o) const {
        return input == o.input;
    }
};

struct ConstantNode {
    std::shared_ptr<const core::Tensor> tensor;
    std::optional<core::QuantParams> quant_params;
};

struct CallOperatorNode {
    std::vector<ValueHandle> args;
    Operator op;
    OutputSpec output_specs;
    std::vector<ConstantArg> constant_args;

    bool operator==(const CallOperatorNode& o) const {
        return args == o.args && op == o.op
            && output_specs == o.output_specs
            && constant_args == o.constant_args;
    }
};

using NodeVariant = std::variant<InputNode, ConstantNode, CallOperatorNode, CallSubgraphNode>;

struct Node {
    NodeVariant value;
    std::vector<NodeHandle> users;
    uint32_t tag = 0;
    NodeFlags flags = NodeFlags::None;

    const std::vector<ValueHandle>& get_args() const {
        return std::visit(overloaded {
            [](const InputNode&) -> const std::vector<ValueHandle>& {
                static const std::vector<ValueHandle> empty;
                return empty;
            },
            [](const ConstantNode&) -> const std::vector<ValueHandle>& {
                static const std::vector<ValueHandle> empty;
                return empty;
            },
            [](const CallOperatorNode& n) -> const std::vector<ValueHandle>& { return n.args; },
            [](const CallSubgraphNode& n) -> const std::vector<ValueHandle>& { return n.args; },
        }, value);
    }

    uint32_t output_count() const {
        return std::visit(overloaded {
            [](const InputNode&) -> uint32_t { return 1; },
            [](const ConstantNode&) -> uint32_t { return 1; },
            [](const CallOperatorNode& n) -> uint32_t {
                return std::visit(overloaded {
                    [](const TensorSpec&) -> uint32_t { return 1; },
                    [](const std::vector<TensorSpec>& v) -> uint32_t { return v.size(); },
                }, n.output_specs);
            },
            [](const CallSubgraphNode& n) -> uint32_t {
                return std::visit(overloaded {
                    [](const TensorSpec&) -> uint32_t { return 1; },
                    [](const std::vector<TensorSpec>& v) -> uint32_t { return v.size(); },
                }, n.output_specs);
            },
        }, value);
    }
};

}
