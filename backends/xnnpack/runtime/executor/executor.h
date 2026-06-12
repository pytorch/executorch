#pragma once

#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/executor/arena.h>
#include <executorch/backends/xnnpack/runtime/executor/shape_env.h>
#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/backends/xnnpack/runtime/plan/execution_plan.h>
#include <executorch/backends/xnnpack/runtime/plan/memory_plan.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <vector>

namespace executorch::backends::xnnpack::executor {

struct Executor {
  // Owns the graph (and its constant tensors) for the executor's lifetime.
  // XNNPACK keeps pointers into unpacked constant data (e.g. PReLU slopes), so
  // this must outlive `plan`'s runtimes. Declared first => destroyed last.
  graph::Graph graph;
  Arena arena;
  std::vector<graph::TensorSpec> input_specs;
  std::vector<plan::ValueSlot> input_slots;
  plan::MemoryPlan memory_plan;
  plan::ExecutionPlan plan;
  ShapeEnv shape_env;
  std::vector<plan::ValueSlot> output_slots;
  std::vector<core::Tensor> values;

  runtime::Result<std::vector<core::Tensor>> run(
      runtime::Span<core::Tensor> inputs);
  runtime::Error run_step(size_t step_idx, const plan::PlanStep& step);
  runtime::Error setup_xnn_step(const plan::RunXnnSubgraphStep& xnn);
  runtime::Error update_xnn_output_shapes(const plan::RunXnnSubgraphStep& xnn);
  runtime::Error update_planned_memory(runtime::Span<core::Tensor> inputs);

  static runtime::Result<Executor> build(graph::Graph& graph);
};

} // namespace executorch::backends::xnnpack::executor
