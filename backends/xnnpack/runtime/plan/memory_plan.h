#pragma once

#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/executor/shape_env.h>
#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>
#include <executorch/backends/xnnpack/runtime/plan/execution_plan.h>

#include <variant>
#include <vector>

namespace executorch::backends::xnnpack::plan {

struct ArenaAllocation {
    size_t offset;
    size_t size;
};
struct DynamicAllocation {};
struct ExternalAllocation {};

using AllocationInfo = std::variant<ArenaAllocation, DynamicAllocation, ExternalAllocation>;

struct MemoryPlan {
    size_t arena_size;
    std::vector<AllocationInfo> value_allocations;
    std::vector<graph::TensorSpec> value_specs;

    void replan(const executor::ShapeEnv& shape_env);
};

MemoryPlan create_memory_plan(graph::Graph& graph, ExecutionPlan& execution_plan);

}
