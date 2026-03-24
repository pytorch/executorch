#pragma once

#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/backends/xnnpack/runtime/graph/handles.h>
#include <executorch/backends/xnnpack/runtime/graph/operator.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace executorch::backends::xnnpack::graph {

class GraphBuilder {
public:
    Graph build();
    ValueHandle createInput(TensorSpec spec);
    ValueHandle createConstant(
        std::shared_ptr<const core::Tensor> tensor,
        std::optional<core::QuantParams> quant_params = std::nullopt);
    ValueHandle createOperator(Operator op, TensorSpec output_spec, ValueHandles args);
    ValueHandle createOperator(Operator op, TensorSpec output_spec, ValueHandles args,
        std::vector<ConstantArg> constant_args);
    ValueHandle createOperatorM(Operator op, std::vector<TensorSpec> output_specs, ValueHandles args);
    OutputHandle createOutput(ValueHandle handle);
    ValueHandle createSymInt();

    template <class... Ts>
    ValueHandle createOperator(Operator op, TensorSpec output_spec, Ts... ts) {
        return createOperator(op, output_spec, ValueHandles{std::forward<Ts>(ts)...});
    }
    template <class... Ts>
    ValueHandle createOperatorM(Operator op, std::vector<TensorSpec> output_specs, Ts... ts) {
        return createOperatorM(op, output_specs, ValueHandles{std::forward<Ts>(ts)...});
    }

private:
    std::vector<TensorSpec> input_specs_;
    std::vector<Node> nodes_;
    std::vector<ValueHandle> outputs_;
    uint32_t next_input_ = 0;
    uint32_t next_sym_int_ = 0;
};

}
