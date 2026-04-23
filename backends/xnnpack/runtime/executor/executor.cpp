#include <executorch/backends/xnnpack/runtime/executor/executor.h>

#include <executorch/backends/xnnpack/runtime/core/variant_util.h>
#include <executorch/backends/xnnpack/runtime/plan/execution_plan.h>

#include <cassert>
#include <cstdlib>
#include <xnnpack.h>

namespace executorch::backends::xnnpack::executor {

std::vector<core::Tensor> Executor::run(core::Span<core::Tensor> inputs) {
    update_planned_memory(inputs);

    for (const auto& step : plan.steps) {
        run_step(step);
    }

    std::vector<core::Tensor> outputs;
    outputs.reserve(output_slots.size());
    for (auto slot : output_slots) {
        auto& val = values[slot];
        core::Tensor t;
        t.dtype = val.dtype;
        t.sizes = val.sizes;
        t.storage.data = val.storage.data;
        t.storage.size_in_bytes = val.storage.size_in_bytes;
        t.storage.owner = core::StorageOwner::External;
        outputs.push_back(std::move(t));
    }
    return outputs;
}

void Executor::run_step(const plan::PlanStep& step) {
    std::visit(overloaded {
        [&](const plan::RunOperatorStep& s) {
            std::vector<core::Tensor*> inputs;
            inputs.reserve(s.input_slots.size());
            for (auto slot : s.input_slots) {
                inputs.push_back(&values[slot]);
            }

            std::vector<core::Tensor*> outputs;
            outputs.reserve(s.output_slots.size());
            for (auto slot : s.output_slots) {
                outputs.push_back(&values[slot]);
            }

            s.op->execute(inputs, outputs);
        },
        [&](const plan::RunXnnSubgraphStep& s) {
            auto status = xnn_invoke_runtime(s.runtime.get());
            assert(status == xnn_status_success);
        }
    }, step);
}

void Executor::update_planned_memory(core::Span<core::Tensor> inputs) {
    if (!shape_env.specialize(input_specs, inputs)) {
        abort();
    }

    memory_plan.replan(shape_env);
    arena.resize(memory_plan.arena_size);

    for (size_t i = 0; i < inputs.size(); i++) {
        values[i].sizes = inputs[i].sizes;
        values[i].storage.data = inputs[i].storage.data;
        values[i].storage.size_in_bytes = inputs[i].storage.size_in_bytes;
    }

    assert(memory_plan.value_allocations.size() == values.size());
    for (auto i = 0u; i < values.size(); i++) {
        std::visit(overloaded {
            [&](plan::ArenaAllocation& alloc) {
                auto& storage = values[i].storage;
                assert(storage.owner == core::StorageOwner::Arena);
                storage.data = static_cast<uint8_t*>(arena.data()) + alloc.offset;
                storage.size_in_bytes = alloc.size;
            },
            [&](plan::DynamicAllocation& alloc) {
                assert(values[i].storage.owner == core::StorageOwner::Self);
            },
            [&](plan::ExternalAllocation&) {
                assert(values[i].storage.owner == core::StorageOwner::External);
            }
        }, memory_plan.value_allocations.at(i));
    }

    for (size_t i = 0; i < values.size(); i++) {
        if (std::holds_alternative<plan::ArenaAllocation>(memory_plan.value_allocations[i])) {
            auto& spec = memory_plan.value_specs[i];
            values[i].sizes.resize(spec.sizes.size());
            for (size_t d = 0; d < spec.sizes.size(); d++) {
                auto& dim = spec.sizes[d];
                int64_t size = dim.offset;
                for (auto& term : dim.coeffs) {
                    auto& bound = shape_env.specialized_bounds[term.sym];
                    size += term.coefficient * static_cast<int64_t>(*bound.max);
                }
                values[i].sizes[d] = static_cast<uint64_t>(size);
            }
        }
    }

    for (auto& step : plan.steps) {
        auto* op_step = std::get_if<plan::RunOperatorStep>(&step);
        if (!op_step) continue;

        std::vector<graph::TensorSpec> input_specs;
        input_specs.reserve(op_step->input_slots.size());
        for (auto slot : op_step->input_slots) {
            input_specs.push_back(memory_plan.value_specs[slot]);
        }
        op_step->op->reshape(input_specs);
    }

    for (auto& step : plan.steps) {
        auto* xnn = std::get_if<plan::RunXnnSubgraphStep>(&step);
        if (!xnn) continue;

        auto* rt = xnn->runtime.get();

        for (uint32_t eid = 0; eid < xnn->external_value_slots.size(); eid++) {
            auto slot = xnn->external_value_slots[eid];
            auto& tensor = values[slot];
            std::vector<size_t> dims(tensor.sizes.begin(), tensor.sizes.end());
            xnn_reshape_external_value(rt, eid, dims.size(), dims.data());
        }

        xnn_reshape_runtime(rt);

        std::vector<xnn_external_value> externals(xnn->external_value_slots.size());
        for (uint32_t eid = 0; eid < xnn->external_value_slots.size(); eid++) {
            auto slot = xnn->external_value_slots[eid];
            externals[eid].id = eid;
            externals[eid].data = values[slot].storage.data;
        }
        xnn_setup_runtime_v2(rt, externals.size(), externals.data());
    }
}

Executor Executor::build(graph::Graph& graph) {
    auto init_status = xnn_initialize(nullptr);
    assert(init_status == xnn_status_success);

    auto execution_plan = plan::create_execution_plan(graph);
    auto memory_plan = plan::create_memory_plan(graph, execution_plan);

    std::vector<plan::ValueSlot> output_slots;
    output_slots.reserve(graph.outputs.size());
    for (auto& vh : graph.outputs) {
        output_slots.push_back(graph.nodes[vh.node].tag + vh.output);
    }

    auto num_slots = memory_plan.value_allocations.size();
    std::vector<core::Tensor> values(num_slots);
    for (size_t i = 0; i < num_slots; i++) {
        values[i].dtype = memory_plan.value_specs[i].dtype;
        if (std::holds_alternative<plan::ArenaAllocation>(memory_plan.value_allocations[i])) {
            values[i].storage.owner = core::StorageOwner::Arena;
        }
    }

    for (size_t n = 0; n < graph.nodes.size(); n++) {
        auto* cn = std::get_if<graph::ConstantNode>(&graph.nodes[n].value);
        if (!cn) continue;
        auto slot = graph.nodes[n].tag;
        values[slot].sizes = cn->tensor->sizes;
        values[slot].storage.data = const_cast<void*>(
            static_cast<const void*>(cn->tensor->storage.data));
        values[slot].storage.size_in_bytes = cn->tensor->storage.size_in_bytes;
    }

    Executor exec;
    exec.input_specs = graph.input_specs;
    exec.memory_plan = std::move(memory_plan);
    exec.plan = std::move(execution_plan);
    exec.shape_env = ShapeEnv(graph.symint_count());
    exec.output_slots = std::move(output_slots);
    exec.values = std::move(values);
    return std::move(exec);
}

}
