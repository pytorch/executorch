#include <executorch/backends/xnnpack/runtime/plan/execution_plan.h>
#include <executorch/backends/xnnpack/runtime/core/span.h>
#include <executorch/backends/xnnpack/runtime/operators/operator.h>
#include <executorch/backends/xnnpack/runtime/plan/nhwc_rewrite.h>
#include <executorch/backends/xnnpack/runtime/plan/partition.h>
#include <executorch/backends/xnnpack/runtime/plan/schedule.h>
#include <executorch/backends/xnnpack/runtime/plan/xnn_subgraph.h>

namespace executorch::backends::xnnpack::plan {

using namespace graph;

namespace {

    uint32_t assign_value_slots(
        graph::Graph& graph, core::Span<const NodeHandle> linear_schedule) {
        uint32_t next_slot = 0;
        for (auto nh : linear_schedule) {
            graph.nodes[nh].tag = next_slot;
            next_slot += graph.nodes[nh].output_count();
        }
        return next_slot;
    }

    std::vector<PlanStep> create_plan_steps(
        const graph::Graph& graph, core::Span<const NodeHandle> linear_schedule) {
        std::vector<PlanStep> steps;
        steps.reserve(linear_schedule.size());

        for (auto node_handle : linear_schedule) {
            auto& node = graph.nodes[node_handle];

            std::visit(overloaded {
                [&steps, &node, &graph](const CallSubgraphNode& n) {
                    std::vector<ValueSlot> external_value_slots;
                    external_value_slots.reserve(
                        n.args.size() + node.output_count());

                    for (const auto& arg : n.args) {
                        auto slot = graph.nodes[arg.node].tag + arg.output;
                        external_value_slots.push_back(slot);
                    }

                    for (uint32_t i = 0; i < node.output_count(); i++) {
                        external_value_slots.push_back(node.tag + i);
                    }

                    RunXnnSubgraphStep step;
                    step.runtime = compile_xnn_subgraph(
                        *n.subgraph,
                        nullptr
                    );
                    step.external_value_slots = std::move(external_value_slots);
                    steps.push_back(std::move(step));
                },
                [](const InputNode&) {},
                [](const ConstantNode&) {},
                [&steps, &node, &graph](const CallOperatorNode& n) {
                    std::vector<ValueSlot> input_slots;
                    input_slots.reserve(n.args.size());
                    for (const auto& arg : n.args) {
                        if (arg.is_null()) continue;
                        input_slots.push_back(graph.nodes[arg.node].tag + arg.output);
                    }

                    std::vector<ValueSlot> output_slots;
                    for (uint32_t i = 0; i < node.output_count(); i++) {
                        output_slots.push_back(node.tag + i);
                    }

                    auto op = operators::create_operator(n.op);
                    op->setup(n.constant_args);

                    RunOperatorStep step;
                    step.op = std::move(op);
                    step.input_slots = std::move(input_slots);
                    step.output_slots = std::move(output_slots);
                    steps.push_back(std::move(step));
                }
            }, node.value);
        }

        return steps;
    }
}

ExecutionPlan create_execution_plan(graph::Graph& graph) {
    rewrite_nhwc(graph);
    partition_xnn_subgraphs(graph);

    auto linear_schedule = schedule(graph);
    auto num_value_slots = assign_value_slots(graph, linear_schedule);
    (void)num_value_slots;

    auto plan_steps = create_plan_steps(graph, linear_schedule);
    ExecutionPlan plan;
    plan.steps = std::move(plan_steps);

    return plan;
}

}
