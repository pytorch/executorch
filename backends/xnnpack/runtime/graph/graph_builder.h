#pragma once

#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/backends/xnnpack/runtime/graph/handles.h>
#include <executorch/backends/xnnpack/runtime/graph/operator.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace executorch::backends::xnnpack::graph {

/*
 * A builder class for graph::Graph.
 */
class GraphBuilder {
 public:
  Graph build();

  /* Add an input node to the graph. */
  ValueHandle createInput(TensorSpec spec);

  /* Add a constant node to the graph. */
  ValueHandle createConstant(
      std::shared_ptr<const core::Tensor> tensor,
      std::optional<core::QuantParams> quant_params = std::nullopt);

  /* Add an operator node to the graph. */
  ValueHandle
  createOperator(Operator op, TensorSpec output_spec, ValueHandles args);

  /* Add an operator node to the graph. */
  ValueHandle createOperator(
      Operator op,
      TensorSpec output_spec,
      ValueHandles args,
      std::vector<ConstantArg> constant_args,
      float output_min = -INFINITY,
      float output_max = INFINITY);

  /* Add a multi-output operator node to the graph. */
  ValueHandle createOperatorM(
      Operator op,
      std::vector<TensorSpec> output_specs,
      ValueHandles args);

  /* Add an output node to the graph. */
  OutputHandle createOutput(ValueHandle handle);

  /* Allocate a fresh symbolic int and return its handle. */
  SymIntHandle createSymInt();

  template <class... Ts>
  ValueHandle createOperator(Operator op, TensorSpec output_spec, Ts... ts) {
    return createOperator(
        op, output_spec, ValueHandles{std::forward<Ts>(ts)...});
  }
  template <class... Ts>
  ValueHandle
  createOperatorM(Operator op, std::vector<TensorSpec> output_specs, Ts... ts) {
    return createOperatorM(
        op, output_specs, ValueHandles{std::forward<Ts>(ts)...});
  }

 private:
  std::vector<TensorSpec> input_specs_;
  std::vector<Node> nodes_;
  std::vector<ValueHandle> outputs_;
  uint32_t next_input_ = 0;
  uint32_t next_sym_int_ = 0;
};

} // namespace executorch::backends::xnnpack::graph
