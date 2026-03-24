#pragma once

#include <executorch/backends/xnnpack/runtime/core/span.h>
#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/executor/arena.h>
#include <executorch/backends/xnnpack/runtime/executor/shape_env.h>
#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/backends/xnnpack/runtime/plan/execution_plan.h>
#include <executorch/backends/xnnpack/runtime/plan/memory_plan.h>

#include <vector>

namespace executorch::backends::xnnpack::executor {

struct Executor {
    Arena arena;
    std::vector<graph::TensorSpec> input_specs;
    plan::MemoryPlan memory_plan;
    plan::ExecutionPlan plan;
    ShapeEnv shape_env;
    std::vector<plan::ValueSlot> output_slots;
    std::vector<core::Tensor> values;

    std::vector<core::Tensor> run(core::Span<core::Tensor> inputs);
    void run_step(const plan::PlanStep& step);
    void update_planned_memory(core::Span<core::Tensor> inputs);

    static Executor build(graph::Graph& graph);
};

}
