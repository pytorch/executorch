#pragma once

#include <executorch/backends/xnnpack/runtime/operators/operator.h>
#include <executorch/backends/xnnpack/runtime/plan/xnn_subgraph.h>
#include <executorch/runtime/core/result.h>

#include <memory>
#include <variant>
#include <vector>

namespace executorch::backends::xnnpack::graph {
struct Graph;
}

namespace executorch::backends::xnnpack::plan {

using ValueSlot = uint32_t;

/* Run an operator with the given inputs and outputs. */
struct RunOperatorStep {
  std::unique_ptr<operators::Operator> op;
  std::vector<ValueSlot> input_slots;
  std::vector<ValueSlot> output_slots;
};

/* Run an subgraph delegated to XNNPACK. */
struct RunXnnSubgraphStep {
  XnnRuntime runtime;
  std::vector<ValueSlot> external_value_slots;
  uint32_t num_external_inputs = 0;
};

using PlanStep = std::variant<RunOperatorStep, RunXnnSubgraphStep>;

/*
 * Describes the planned execution steps for a compiled graph.
 */
struct ExecutionPlan {
  std::vector<PlanStep> steps;
};

/*
 * Build an execution plan from a model graph. This pre-processes the
 * graph into a format suitable for efficient execution.
 */
runtime::Result<ExecutionPlan> create_execution_plan(graph::Graph& graph);

} // namespace executorch::backends::xnnpack::plan
