#pragma once

#include <executorch/backends/xnnpack/runtime/operators/operator.h>
#include <executorch/backends/xnnpack/runtime/plan/xnn_subgraph.h>

#include <memory>
#include <variant>
#include <vector>

namespace executorch::backends::xnnpack::graph { struct Graph; }

namespace executorch::backends::xnnpack::plan {

using ValueSlot = uint32_t;

struct RunOperatorStep {
    std::unique_ptr<operators::Operator> op;
    std::vector<ValueSlot> input_slots;
    std::vector<ValueSlot> output_slots;
};

struct RunXnnSubgraphStep {
    XnnRuntime runtime;
    std::vector<ValueSlot> external_value_slots;
};

using PlanStep = std::variant<RunOperatorStep, RunXnnSubgraphStep>;

struct ExecutionPlan {
    std::vector<PlanStep> steps;
};

ExecutionPlan create_execution_plan(graph::Graph& graph);

}
