#include <executorch/backends/xnnpack/runtime/plan/memory_plan.h>

#include <cstdlib>
#include <variant>

namespace executorch::backends::xnnpack::plan {

using namespace graph;

namespace {

uint64_t resolve_dim_upper(const DimSizeSpec& dim,
                           const executor::ShapeEnv& shape_env) {
    int64_t result = dim.offset;
    for (auto& term : dim.coeffs) {
        auto& bound = shape_env.specialized_bounds[term.sym];
        if (!bound.max) { abort(); }
        result += term.coefficient * static_cast<int64_t>(*bound.max);
    }
    return static_cast<uint64_t>(result);
}

std::vector<uint64_t> resolve_sizes_upper(const TensorSpec& spec,
                                           const executor::ShapeEnv& shape_env) {
    std::vector<uint64_t> sizes;
    sizes.reserve(spec.sizes.size());
    for (auto& dim : spec.sizes) {
        sizes.push_back(resolve_dim_upper(dim, shape_env));
    }
    return sizes;
}

} // namespace

MemoryPlan create_memory_plan(Graph& graph, ExecutionPlan& execution_plan) {
    uint32_t num_slots = 0;
    for (auto& node : graph.nodes) {
        uint32_t end = node.tag + node.output_count();
        if (end > num_slots) num_slots = end;
    }

    std::vector<AllocationInfo> allocations(num_slots);
    std::vector<TensorSpec> specs(num_slots);

    for (uint32_t n = 0; n < graph.nodes.size(); n++) {
        auto& node = graph.nodes[n];
        uint32_t base_slot = node.tag;

        for (uint32_t o = 0; o < node.output_count(); o++) {
            uint32_t slot = base_slot + o;
            ValueHandle vh {
                static_cast<uint16_t>(n),
                static_cast<uint16_t>(o)
            };
            specs[slot] = graph.get_tensor_spec(vh);

            if (std::holds_alternative<InputNode>(node.value)
                || std::holds_alternative<ConstantNode>(node.value)) {
                allocations[slot] = ExternalAllocation {};
            } else {
                ArenaAllocation a;
                a.offset = 0;
                a.size = 0;
                allocations[slot] = a;
            }
        }
    }

    MemoryPlan mp;
    mp.arena_size = 0;
    mp.value_allocations = std::move(allocations);
    mp.value_specs = std::move(specs);
    return mp;
}

void MemoryPlan::replan(const executor::ShapeEnv& shape_env) {
    size_t arena_offset = 0;

    for (size_t i = 0; i < value_allocations.size(); i++) {
        if (auto* arena = std::get_if<ArenaAllocation>(&value_allocations[i])) {
            auto concrete_sizes = resolve_sizes_upper(value_specs[i], shape_env);
            size_t size = core::compute_storage_size(concrete_sizes, value_specs[i].dtype);
            arena->offset = arena_offset;
            arena->size = size;
            arena_offset += size;
        }
    }

    arena_size = arena_offset;
}

}
