#include <executorch/backends/xnnpack/runtime/executor/executor.h>

#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/core/variant_util.h>
#include <executorch/backends/xnnpack/runtime/plan/execution_plan.h>

#include <executorch/runtime/platform/log.h>

#include <xnnpack.h>
#include <cassert>
#include <chrono>
#include <cstdlib>

namespace executorch::backends::xnnpack::executor {

using executorch::runtime::Span;

runtime::Result<std::vector<core::Tensor>> Executor::run(
    Span<core::Tensor> inputs) {
  auto t_mem_start = std::chrono::steady_clock::now();
  ET_CHECK_OK_OR_RETURN_ERROR(update_planned_memory(inputs));
  auto t_mem_end = std::chrono::steady_clock::now();
  ET_LOG(
      Info,
      "update_planned_memory %lldus",
      (long long)std::chrono::duration_cast<std::chrono::microseconds>(
          t_mem_end - t_mem_start)
          .count());

  for (size_t si = 0; si < plan.steps.size(); si++) {
    ET_CHECK_OK_OR_RETURN_ERROR(run_step(si, plan.steps[si]));
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

runtime::Error Executor::run_step(size_t step_idx, const plan::PlanStep& step) {
  runtime::Error err = runtime::Error::Ok;
  std::visit(
      overloaded{
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

            auto t0 = std::chrono::steady_clock::now();
            s.op->execute(
                {inputs.data(), inputs.size()},
                {outputs.data(), outputs.size()});
            auto t1 = std::chrono::steady_clock::now();
            auto us =
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                    .count();
            ET_LOG(Info, "OpStep[%zu] %lldus", step_idx, (long long)us);
          },
          [&](const plan::RunXnnSubgraphStep& s) {
            auto t0 = std::chrono::steady_clock::now();
            err = setup_xnn_step(s);
            if (err != runtime::Error::Ok) {
              return;
            }
            auto t1 = std::chrono::steady_clock::now();
            auto status = xnn_invoke_runtime(s.runtime.get());
            if (status != xnn_status_success) {
              ET_LOG(
                  Error,
                  "xnn_invoke_runtime failed: 0x%x",
                  (unsigned int)status);
              err = runtime::Error::Internal;
              return;
            }
            auto t2 = std::chrono::steady_clock::now();
            err = update_xnn_output_shapes(s);
            if (err != runtime::Error::Ok) {
              return;
            }
            auto setup_us =
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                    .count();
            auto invoke_us =
                std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                    .count();
            ET_LOG(
                Info,
                "XnnStep setup=%lldus invoke=%lldus",
                (long long)setup_us,
                (long long)invoke_us);
          }},
      step);
  return err;
}

runtime::Error Executor::setup_xnn_step(const plan::RunXnnSubgraphStep& xnn) {
  auto* rt = xnn.runtime.get();

  for (uint32_t eid = 0; eid < xnn.num_external_inputs; eid++) {
    auto slot = xnn.external_value_slots[eid];
    auto& tensor = values[slot];
    std::vector<size_t> dims(tensor.sizes.begin(), tensor.sizes.end());
    auto status = xnn_reshape_external_value(rt, eid, dims.size(), dims.data());
    ET_CHECK_OR_RETURN_ERROR(
        status == xnn_status_success,
        Internal,
        "xnn_reshape_external_value failed: 0x%x",
        (unsigned int)status);
  }

  ET_CHECK_OR_RETURN_ERROR(
      xnn_reshape_runtime(rt) == xnn_status_success,
      Internal,
      "xnn_reshape_runtime failed");

  std::vector<xnn_external_value> externals(xnn.external_value_slots.size());
  for (uint32_t eid = 0; eid < xnn.external_value_slots.size(); eid++) {
    auto slot = xnn.external_value_slots[eid];
    externals[eid].id = eid;
    externals[eid].data = values[slot].storage.data;
  }
  ET_CHECK_OR_RETURN_ERROR(
      xnn_setup_runtime_v2(rt, externals.size(), externals.data()) ==
          xnn_status_success,
      Internal,
      "xnn_setup_runtime_v2 failed");
  return runtime::Error::Ok;
}

runtime::Error Executor::update_xnn_output_shapes(
    const plan::RunXnnSubgraphStep& xnn) {
  auto* rt = xnn.runtime.get();

  for (uint32_t eid = 0; eid < xnn.external_value_slots.size(); eid++) {
    auto slot = xnn.external_value_slots[eid];
    size_t num_dims = 0;
    size_t dims[XNN_MAX_TENSOR_DIMS];
    ET_CHECK_OR_RETURN_ERROR(
        xnn_get_external_value_shape(rt, eid, &num_dims, dims) ==
            xnn_status_success,
        Internal,
        "xnn_get_external_value_shape failed");

    values[slot].sizes.resize(num_dims);
    for (size_t d = 0; d < num_dims; d++) {
      values[slot].sizes[d] = dims[d];
    }
    size_t nbytes = core::byte_stride(values[slot].dtype);
    for (size_t d = 0; d < num_dims; d++) {
      nbytes *= dims[d];
    }
    values[slot].storage.size_in_bytes = nbytes;
  }
  return runtime::Error::Ok;
}

runtime::Error Executor::update_planned_memory(Span<core::Tensor> inputs) {
  ET_CHECK_OK_OR_RETURN_ERROR(
      shape_env.specialize({input_specs.data(), input_specs.size()}, inputs));

  ET_CHECK_OK_OR_RETURN_ERROR(memory_plan.replan(shape_env));
  ET_CHECK_OK_OR_RETURN_ERROR(arena.resize(memory_plan.arena_size));

  for (size_t i = 0; i < inputs.size(); i++) {
    auto slot = input_slots[i];
    values[slot].sizes = inputs[i].sizes;
    values[slot].storage.data = inputs[i].storage.data;
    values[slot].storage.size_in_bytes = inputs[i].storage.size_in_bytes;
  }

  assert(memory_plan.value_allocations.size() == values.size());
  for (auto i = 0u; i < values.size(); i++) {
    std::visit(
        overloaded{
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
            }},
        memory_plan.value_allocations.at(i));
  }

  for (size_t i = 0; i < values.size(); i++) {
    if (std::holds_alternative<plan::ArenaAllocation>(
            memory_plan.value_allocations[i])) {
      auto& spec = memory_plan.value_specs[i];
      values[i].sizes.resize(spec.sizes.size());
      for (size_t d = 0; d < spec.sizes.size(); d++) {
        auto& dim = spec.sizes[d];
        int64_t size = dim.offset;
        for (auto& term : dim.coeffs) {
          auto& bound = shape_env.bounds[term.sym];
          size += term.coefficient * static_cast<int64_t>(*bound.max);
        }
        values[i].sizes[d] = static_cast<uint64_t>(size);
      }
    }
  }

  for (auto& step : plan.steps) {
    auto* op_step = std::get_if<plan::RunOperatorStep>(&step);
    if (!op_step)
      continue;

    std::vector<graph::TensorSpec> input_specs;
    input_specs.reserve(op_step->input_slots.size());
    for (auto slot : op_step->input_slots) {
      input_specs.push_back(memory_plan.value_specs[slot]);
    }
    op_step->op->reshape({input_specs.data(), input_specs.size()});
  }

  return runtime::Error::Ok;
}

runtime::Result<Executor> Executor::build(graph::Graph& graph) {
  auto t_build_start = std::chrono::steady_clock::now();

  auto init_status = xnn_initialize(nullptr);
  ET_CHECK_OR_RETURN_ERROR(
      init_status == xnn_status_success,
      Internal,
      "Failed to initialize XNNPACK: 0x%x",
      (unsigned int)init_status);

  auto t0 = std::chrono::steady_clock::now();
  ET_UNWRAP(execution_plan, plan::create_execution_plan(graph));
  auto t1 = std::chrono::steady_clock::now();
  auto memory_plan = plan::create_memory_plan(graph, execution_plan);
  auto t2 = std::chrono::steady_clock::now();

  std::vector<plan::ValueSlot> output_slots;
  output_slots.reserve(graph.outputs.size());
  for (auto& vh : graph.outputs) {
    output_slots.push_back(graph.nodes[vh.node].tag + vh.output);
  }

  auto num_slots = memory_plan.value_allocations.size();
  std::vector<core::Tensor> values(num_slots);
  for (size_t i = 0; i < num_slots; i++) {
    values[i].dtype = memory_plan.value_specs[i].dtype;
    if (std::holds_alternative<plan::ArenaAllocation>(
            memory_plan.value_allocations[i])) {
      values[i].storage.owner = core::StorageOwner::Arena;
    }
  }

  for (size_t n = 0; n < graph.nodes.size(); n++) {
    auto* cn = std::get_if<graph::ConstantNode>(&graph.nodes[n].value);
    if (!cn)
      continue;
    auto slot = graph.nodes[n].tag;
    values[slot].sizes = cn->tensor->sizes;
    values[slot].storage.data =
        const_cast<void*>(static_cast<const void*>(cn->tensor->storage.data));
    values[slot].storage.size_in_bytes = cn->tensor->storage.size_in_bytes;
  }

  auto t3 = std::chrono::steady_clock::now();

  // Let operators pre-process constant tensors (e.g., pack weights).
  for (auto& step : execution_plan.steps) {
    auto* op_step = std::get_if<plan::RunOperatorStep>(&step);
    if (!op_step)
      continue;

    std::vector<core::Tensor*> inputs;
    for (auto slot : op_step->input_slots)
      inputs.push_back(&values[slot]);

    std::vector<core::Tensor*> outputs;
    for (auto slot : op_step->output_slots)
      outputs.push_back(&values[slot]);

    op_step->op->prepare(
        {inputs.data(), inputs.size()}, {outputs.data(), outputs.size()});
  }

  auto t4 = std::chrono::steady_clock::now();
  ET_LOG(
      Info,
      "Executor::build create_execution_plan=%lldms create_memory_plan=%lldms "
      "setup_values=%lldms prepare=%lldms",
      (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
          .count(),
      (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
          .count(),
      (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2)
          .count(),
      (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3)
          .count());

  std::vector<plan::ValueSlot> input_slots(graph.input_specs.size());
  for (size_t n = 0; n < graph.nodes.size(); n++) {
    auto* in = std::get_if<graph::InputNode>(&graph.nodes[n].value);
    if (!in)
      continue;
    input_slots[in->input] = graph.nodes[n].tag;
  }

  Executor exec;
  exec.input_specs = graph.input_specs;
  exec.input_slots = std::move(input_slots);
  exec.memory_plan = std::move(memory_plan);
  exec.plan = std::move(execution_plan);
  exec.shape_env = ShapeEnv(graph.symint_count());
  exec.output_slots = std::move(output_slots);
  exec.values = std::move(values);
  // Keep the graph (and thus all constant tensor storage) alive for the
  // executor's lifetime; XNNPACK references unpacked constant data directly.
  exec.graph = std::move(graph);
  return std::move(exec);
}

} // namespace executorch::backends::xnnpack::executor
