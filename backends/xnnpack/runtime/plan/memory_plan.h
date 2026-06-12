#pragma once

#include <executorch/backends/xnnpack/runtime/core/tensor.h>
#include <executorch/backends/xnnpack/runtime/executor/shape_env.h>
#include <executorch/backends/xnnpack/runtime/graph/graph.h>
#include <executorch/backends/xnnpack/runtime/graph/tensor_spec.h>
#include <executorch/backends/xnnpack/runtime/plan/execution_plan.h>
#include <executorch/runtime/core/error.h>

#include <variant>
#include <vector>

namespace executorch::backends::xnnpack::plan {

/* Describes an allocation planned in the primary memory arena. */
struct ArenaAllocation {
  size_t offset;
  size_t size;
};
/*
 * Describes a standalone dynamic allocation. This memory is not
 * overlapped or otherwise memory planned.
 */
struct DynamicAllocation {};
/*
 * Describes externally-owned memory.
 */
struct ExternalAllocation {};

using AllocationInfo =
    std::variant<ArenaAllocation, DynamicAllocation, ExternalAllocation>;

/*
 * Describes the arena range and/or allocation strategy for each
 * value in an execution plan.
 */
struct MemoryPlan {
  size_t arena_size;
  std::vector<AllocationInfo> value_allocations;
  std::vector<graph::TensorSpec> value_specs;

  runtime::Error replan(const executor::ShapeEnv& shape_env);
};

MemoryPlan create_memory_plan(
    graph::Graph& graph,
    ExecutionPlan& execution_plan);

} // namespace executorch::backends::xnnpack::plan
