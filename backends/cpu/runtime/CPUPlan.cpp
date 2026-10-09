// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/CPUPlan.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace executorch::backends::cpu {
using namespace executorch::runtime;
namespace {
std::string json_string(std::string_view text) {
  std::string output = "\"";
  for (const auto character : text) {
    if (character == '\\' || character == '"') {
      output += '\\';
    }
    if (static_cast<unsigned char>(character) < 0x20) {
      output += '?';
    } else {
      output += character;
    }
  }
  output += '"';
  return output;
}

} // namespace

CPUPlan::CPUPlan(
    const ptn::Graph& graph,
    std::vector<Buffer>& buffers,
    RuntimeConfiguration configuration,
    ExecutionContext execution)
    : graph_(graph),
      buffers_(buffers),
      configuration_(std::move(configuration)),
      execution_(execution) {}

CPUPlan::~CPUPlan() {
  steps_.clear();
  providers_.clear();
}

Error CPUPlan::prepare(MemoryAllocator& allocator) {
  ET_CHECK_OR_RETURN_ERROR(
      !preparation_started_ && requirements_.size() == graph_.values.size(),
      InvalidState,
      "CPU plan must select before preparation");
  preparation_started_ = true;
  for (const auto& step : steps_) {
    for (auto id : step.region.inputs) {
      const auto& value = graph_.value(id);
      if (value.role == ptn::ValueRole::Parameter ||
          value.role == ptn::ValueRole::ConstantTensor ||
          value.role == ptn::ValueRole::Buffer) {
        ET_CHECK_OR_RETURN_ERROR(
            buffers_.at(id).accepts(requirements_.at(id)),
            InvalidArgument,
            "CPU constant %s before preparation: required read=%zu write=%zu alignment=%zu; available read=%zu write=%zu alignment=%zu",
            value.name.c_str(),
            requirements_[id].readable_bytes,
            requirements_[id].writable_bytes,
            requirements_[id].alignment,
            buffers_[id].readable_bytes,
            buffers_[id].writable_bytes,
            buffers_[id].alignment);
      }
    }
  }
  BufferRequirements scratch;
  for (const auto& step : steps_) {
    scratch.merge(step.storage.scratch);
  }
  if (scratch.readable_bytes || scratch.writable_bytes) {
    auto storage = allocate_buffer(allocator, scratch);
    if (!storage.ok()) {
      return storage.error();
    }
    scratch_ = storage.get();
    scratch_.owner = StorageOwner::Scratch;
  }
  auto error = allocate_activations(allocator);
  if (error != Error::Ok) {
    return error;
  }
  PreparationContext preparation{
      graph_, buffers_, allocator, private_constant_bytes_, execution_};
  for (auto& step : steps_) {
    auto compiled = step.implementation->compile(step.region, preparation);
    if (!compiled.ok()) {
      return compiled.error();
    }
    step.executable = std::move(compiled.get());
  }
  for (auto& provider : providers_) {
    error = provider->finish();
    if (error != Error::Ok) {
      return error;
    }
  }
  for (auto& step : steps_) {
    error = step.executable->reshape();
    if (error != Error::Ok) {
      return error;
    }
  }
  prepared_ = true;
  return Error::Ok;
}

Error CPUPlan::execute(const ExecutionContext& context) {
  ET_CHECK_OR_RETURN_ERROR(prepared_, InvalidState, "CPU plan is not prepared");
  ET_CHECK_OR_RETURN_ERROR(
      context.threads == execution_.threads &&
          context.avx2_fma == execution_.avx2_fma &&
          context.threadpool == execution_.threadpool,
      NotSupported,
      "CPU execution properties changed; reload the static method");
  for (auto& step : steps_) {
    for (const auto& value : step.storage.values) {
      ET_CHECK_OR_RETURN_ERROR(
          buffers_.at(value.id).accepts(requirements_.at(value.id)),
          InvalidArgument,
          "CPU binding %s: required read=%zu write=%zu alignment=%zu; available read=%zu write=%zu alignment=%zu",
          graph_.value(value.id).name.c_str(),
          requirements_[value.id].readable_bytes,
          requirements_[value.id].writable_bytes,
          requirements_[value.id].alignment,
          buffers_[value.id].readable_bytes,
          buffers_[value.id].writable_bytes,
          buffers_[value.id].alignment);
    }
    const auto error = step.executable->bind(scratch_);
    if (error != Error::Ok) {
      return error;
    }
  }
  for (auto& step : steps_) {
    const auto error = step.executable->run(context);
    if (error != Error::Ok) {
      return error;
    }
  }
  return Error::Ok;
}

Error CPUPlan::select() {
  const auto& configuration = configuration_;
  const auto& graph = graph_;
  ET_CHECK_OR_RETURN_ERROR(
      !configuration.force || !configuration.preferences.empty(),
      InvalidArgument,
      "CPU force requires at least one preference");
  for (const auto& preference : configuration.preferences) {
    ET_CHECK_OR_RETURN_ERROR(
        !preference.provider.empty(),
        InvalidArgument,
        "CPU preferences require a provider name");
  }
  ET_CHECK_OR_RETURN_ERROR(
      providers_.empty(), InvalidState, "CPU plan already selected");
  providers_.reserve(configuration.providers.size());
  implementations_.reserve(configuration.providers.size());
  steps_.reserve(graph.schedule.size());
  std::vector<bool> preferences_found(configuration.preferences.size(), false);
  for (auto factory : configuration.providers) {
    ET_CHECK_OR_RETURN_ERROR(
        factory, InvalidArgument, "Null CPU provider factory");
    auto provider = factory();
    ET_CHECK_OR_RETURN_ERROR(provider, InvalidArgument, "Null CPU provider");
    implementations_.push_back(provider->implementations());
    for (size_t index = 0; index < configuration.preferences.size(); ++index) {
      const auto& preference = configuration.preferences[index];
      if (provider->name() == preference.provider) {
        preferences_found.at(index) = preferences_found.at(index) ||
            preference.implementation.empty() ||
            std::any_of(implementations_.back().begin(),
                        implementations_.back().end(),
                        [&](const auto* implementation) {
                          return implementation->name() ==
                              preference.implementation;
                        });
      }
    }
    providers_.push_back(std::move(provider));
  }
  for (size_t index = 0; index < preferences_found.size(); ++index) {
    ET_CHECK_OR_RETURN_ERROR(
        preferences_found.at(index),
        InvalidArgument,
        "Requested CPU provider/implementation is not linked: %s/%s",
        configuration.preferences[index].provider.c_str(),
        configuration.preferences[index].implementation.c_str());
  }
  std::vector<size_t> node_regions(graph.nodes.size(), SIZE_MAX);
  bool preferred_matched = false;
  for (auto id : graph.schedule) {
    const auto& node = graph.node(id);
    if (!node.is_call()) {
      continue;
    }
    for (const auto& input : node.inputs) {
      ET_CHECK_OR_RETURN_ERROR(
          !input.mutated,
          NotSupported,
          "CPU static delegate rejects mutation: %s",
          node.target.c_str());
    }
    for (const auto& output : node.outputs) {
      ET_CHECK_OR_RETURN_ERROR(
          output.kind == ptn::OutputValueKind::Tensor,
          NotSupported,
          "CPU static delegate requires tensor outputs: %s",
          node.target.c_str());
    }
    KernelProvider* selected_provider = nullptr;
    KernelImplementation* selected = nullptr;
    KernelProvider* tied_provider = nullptr;
    KernelImplementation* tied = nullptr;
    int priority = std::numeric_limits<int>::min();
    size_t selected_rank = configuration.preferences.size();
    for (size_t provider_index = 0; provider_index < providers_.size();
         ++provider_index) {
      auto& provider = providers_[provider_index];
      for (auto* implementation : implementations_[provider_index]) {
        size_t rank = 0;
        for (; rank < configuration.preferences.size(); ++rank) {
          const auto& preference = configuration.preferences[rank];
          if (provider->name() == preference.provider &&
              (preference.implementation.empty() ||
               implementation->name() == preference.implementation)) {
            break;
          }
        }
        const auto support = implementation->supports(node, graph, execution_);
        if (!decisions_.empty()) {
          decisions_ += ',';
        }
        decisions_ += "{\"node\":" + json_string(node.name) +
            ",\"provider\":" + json_string(provider->name()) +
            ",\"implementation\":" + json_string(implementation->name()) +
            ",\"preference_rank\":" +
            (rank < configuration.preferences.size() ? std::to_string(rank)
                                                     : "null") +
            ",\"eligible\":" + (support.supported ? "true" : "false") +
            ",\"reason\":" + json_string(support.reason) + "}";
        if (!support.supported) {
          continue;
        }
        if (selected && rank == selected_rank &&
            implementation->baseline_priority() == priority) {
          tied_provider = provider.get();
          tied = implementation;
          continue;
        }
        if (!selected || rank < selected_rank ||
            (rank == selected_rank &&
             implementation->baseline_priority() > priority)) {
          tied = nullptr;
          selected_provider = provider.get();
          selected = implementation;
          priority = implementation->baseline_priority();
          selected_rank = rank;
        }
      }
    }
    if (!selected || !selected_provider) {
      ET_LOG(
          Error,
          "No linked CPU implementation accepts semantic operator %s (%s)",
          node.target.c_str(),
          node.name.c_str());
      return Error::NotSupported;
    }
    if (tied && tied_provider) {
      ET_LOG(
          Error,
          "CPU implementations %.*s/%.*s and %.*s/%.*s tie for %s; add a preference",
          static_cast<int>(selected_provider->name().size()),
          selected_provider->name().data(),
          static_cast<int>(selected->name().size()),
          selected->name().data(),
          static_cast<int>(tied_provider->name().size()),
          tied_provider->name().data(),
          static_cast<int>(tied->name().size()),
          tied->name().data(),
          node.name.c_str());
      return Error::InvalidArgument;
    }
    preferred_matched |= selected_rank < configuration.preferences.size();
    bool connected = false;
    for (auto value : node.input_value_ids()) {
      const auto producer = graph.value(value).producer_id;
      if (producer == ptn::kInvalid || !graph.node(producer).is_call()) {
        continue;
      }
      ET_CHECK_OR_RETURN_ERROR(
          node_regions.at(producer) != SIZE_MAX,
          InvalidProgram,
          "CPU graph schedule is not topological: %s",
          node.name.c_str());
      connected |=
          !steps_.empty() && node_regions.at(producer) == steps_.size() - 1;
    }
    // Consecutive connected schedule intervals cannot introduce a contraction
    // cycle.
    if (steps_.empty() || steps_.back().implementation != selected ||
        !selected->accepts_regions() || !connected) {
      steps_.push_back({selected_provider, selected, {}, nullptr, {}});
    }
    steps_.back().region.nodes.push_back(id);
    node_regions.at(id) = steps_.size() - 1;
  }
  ET_CHECK_OR_RETURN_ERROR(
      !configuration.force || preferred_matched,
      NotSupported,
      "Forced CPU preferences match no operation");
  const std::unordered_set<ptn::ValueId> graph_outputs(
      graph.output_ids.begin(), graph.output_ids.end());
  for (size_t index = 0; index < steps_.size(); ++index) {
    auto& step = steps_[index];
    std::unordered_set<ptn::ValueId> inputs;
    std::unordered_set<ptn::ValueId> outputs;
    for (auto id : step.region.nodes) {
      const auto& node = graph.node(id);
      for (auto input : node.input_value_ids()) {
        const auto producer = graph.value(input).producer_id;
        if (producer == ptn::kInvalid || node_regions.at(producer) != index) {
          inputs.insert(input);
        }
      }
      for (const auto& output : node.outputs) {
        const auto value = output.value_id;
        bool exposed = !step.implementation->accepts_regions() ||
            graph_outputs.count(value);
        for (auto consumer : graph.value(value).consumer_ids) {
          exposed |= node_regions.at(consumer) != index;
        }
        if (exposed) {
          outputs.insert(value);
        }
      }
    }
    step.region.inputs.assign(inputs.begin(), inputs.end());
    step.region.outputs.assign(outputs.begin(), outputs.end());
    std::sort(step.region.inputs.begin(), step.region.inputs.end());
    std::sort(step.region.outputs.begin(), step.region.outputs.end());
  }
  requirements_.resize(graph.values.size());
  for (size_t id = 0; id < graph.values.size(); ++id) {
    auto bytes = tensor_bytes(graph.values[id]);
    if (!bytes.ok()) {
      return bytes.error();
    }
    auto minimum = tensor_requirements(bytes.get(), false);
    if (!minimum.ok()) {
      return minimum.error();
    }
    requirements_[id] = minimum.get();
  }
  for (auto& step : steps_) {
    auto storage =
        step.implementation->requirements(step.region, graph, execution_);
    if (!storage.ok()) {
      return storage.error();
    }
    step.storage = std::move(storage.get());
    for (auto* ids : {&step.region.inputs, &step.region.outputs}) {
      for (auto id : *ids) {
        ET_CHECK_OR_RETURN_ERROR(
            std::any_of(
                step.storage.values.begin(),
                step.storage.values.end(),
                [id](const auto& value) { return value.id == id; }),
            InvalidArgument,
            "CPU provider omitted a boundary storage requirement: %s",
            graph.value(id).name.c_str());
      }
    }
    ET_CHECK_OR_RETURN_ERROR(
        step.storage.scratch.valid(),
        InvalidArgument,
        "Invalid CPU scratch requirement");
    for (const auto& value : step.storage.values) {
      ET_CHECK_OR_RETURN_ERROR(
          value.id < requirements_.size() && value.buffer.valid(),
          InvalidArgument,
          "Invalid CPU storage requirement");
      requirements_[value.id].merge(value.buffer);
    }
    for (auto id : step.region.outputs) {
      auto bytes = tensor_bytes(graph.value(id));
      if (!bytes.ok()) {
        return bytes.error();
      }
      requirements_[id].writable_bytes =
          std::max(requirements_[id].writable_bytes, bytes.get());
    }
  }
  return Error::Ok;
}

Error CPUPlan::allocate_activations(MemoryAllocator& allocator) {
  const auto& graph = graph_;
  std::vector<size_t> first(graph.values.size(), SIZE_MAX);
  std::vector<size_t> last(graph.values.size(), 0);
  for (auto id : graph.input_ids) {
    first.at(id) = 0;
    auto bytes = tensor_bytes(graph.value(id));
    if (!bytes.ok()) {
      return bytes.error();
    }
    requirements_[id].writable_bytes =
        std::max(requirements_[id].writable_bytes, bytes.get());
  }
  for (size_t index = 0; index < steps_.size(); ++index) {
    for (auto id : steps_[index].region.outputs) {
      first.at(id) = std::min(first.at(id), index);
    }
    for (auto id : steps_[index].region.inputs) {
      last.at(id) = std::max(last.at(id), index);
    }
  }
  for (auto id : graph.output_ids) {
    last.at(id) = steps_.size();
  }
  const size_t alignment = std::accumulate(
      requirements_.begin(),
      requirements_.end(),
      kBufferAlignment,
      [](size_t current, const auto& requirement) {
        return std::max(current, requirement.alignment);
      });
  struct Interval {
    ptn::ValueId id;
    size_t first;
    size_t last;
    size_t capacity;
  };
  std::vector<Interval> intervals;
  intervals.reserve(graph.values.size());
  for (size_t id = 0; id < graph.values.size(); ++id) {
    if (buffers_[id].data || first.at(id) == SIZE_MAX) {
      continue;
    }
    const size_t bytes = std::max(
        requirements_[id].readable_bytes, requirements_[id].writable_bytes);
    ET_CHECK_OR_RETURN_ERROR(
        bytes <= SIZE_MAX - (alignment - 1),
        InvalidProgram,
        "CPU arena tensor size overflow");
    const size_t capacity = (bytes + alignment - 1) & ~(alignment - 1);
    intervals.push_back(
        {static_cast<ptn::ValueId>(id),
         first.at(id),
         std::max(first.at(id), last.at(id)),
         capacity});
  }
  std::sort(
      intervals.begin(), intervals.end(), [](const auto& a, const auto& b) {
        return a.first == b.first ? a.capacity > b.capacity : a.first < b.first;
      });
  struct Block {
    size_t offset;
    size_t capacity;
    size_t last;
  };
  std::vector<Block> blocks;
  blocks.reserve(intervals.size());
  std::vector<size_t> offsets(graph.values.size(), SIZE_MAX);
  for (const auto& interval : intervals) {
    auto best = blocks.end();
    for (auto candidate = blocks.begin(); candidate != blocks.end();
         ++candidate) {
      if (candidate->last < interval.first &&
          candidate->capacity >= interval.capacity &&
          (best == blocks.end() || candidate->capacity < best->capacity)) {
        best = candidate;
      }
    }
    if (best == blocks.end()) {
      ET_CHECK_OR_RETURN_ERROR(
          arena_bytes_ <= SIZE_MAX - interval.capacity,
          MemoryAllocationFailed,
          "CPU activation arena overflow");
      offsets.at(interval.id) = arena_bytes_;
      blocks.push_back({arena_bytes_, interval.capacity, interval.last});
      arena_bytes_ += interval.capacity;
    } else {
      offsets.at(interval.id) = best->offset;
      best->last = interval.last;
    }
  }
  auto* arena = static_cast<uint8_t*>(
      allocator.allocate(std::max(arena_bytes_, alignment), alignment));
  ET_CHECK_OR_RETURN_ERROR(
      arena,
      MemoryAllocationFailed,
      "CPU arena allocation failed: %zu",
      arena_bytes_);
  std::memset(arena, 0, arena_bytes_);
  for (const auto& interval : intervals) {
    buffers_[interval.id] = {
        arena + offsets.at(interval.id),
        interval.capacity,
        requirements_[interval.id].writable_bytes,
        alignment,
        StorageOwner::Activation};
  }
  return Error::Ok;
}

std::string CPUPlan::describe(double preparation_ms) const {
  std::string report = "{\"regions\":[";
  for (size_t index = 0; index < steps_.size(); ++index) {
    const auto& step = steps_[index];
    if (index) {
      report += ',';
    }
    report += "{\"provider\":" + json_string(step.provider->name()) +
        ",\"implementation\":" + json_string(step.implementation->name()) +
        ",\"nodes\":[";
    for (size_t node = 0; node < step.region.nodes.size(); ++node) {
      if (node) {
        report += ',';
      }
      report += json_string(graph_.node(step.region.nodes[node]).name);
    }
    report += "],\"inputs\":[";
    for (size_t input = 0; input < step.region.inputs.size(); ++input) {
      if (input) {
        report += ',';
      }
      report += std::to_string(step.region.inputs[input]);
    }
    report += "],\"outputs\":[";
    for (size_t output = 0; output < step.region.outputs.size(); ++output) {
      if (output) {
        report += ',';
      }
      report += std::to_string(step.region.outputs[output]);
    }
    report += "],\"scratch_bytes\":" +
        std::to_string(step.storage.scratch.readable_bytes) + "}";
  }
  report += "],\"preferences\":[";
  for (size_t index = 0; index < configuration_.preferences.size(); ++index) {
    if (index) {
      report += ',';
    }
    const auto& preference = configuration_.preferences[index];
    report += "{\"provider\":" + json_string(preference.provider) +
        ",\"implementation\":" + json_string(preference.implementation) + "}";
  }
  report +=
      "],\"policy\":" +
      json_string(
          configuration_.preferences.empty() ? "auto_baseline_unknown_cost"
                                             : "explicit_preference") +
      ",\"preparation_ms\":" + std::to_string(preparation_ms) +
      ",\"activation_arena_bytes\":" + std::to_string(arena_bytes_) +
      ",\"private_constant_bytes\":" + std::to_string(private_constant_bytes_) +
      ",\"shared_scratch_bytes\":" + std::to_string(scratch_.readable_bytes) +
      ",\"xnn_workspace_bytes\":null,\"xnn_packed_cache_bytes\":null,"
      "\"xnn_workspace_shared\":true,\"xnn_packed_cache\":\"per_plan\","
      "\"delegate_io\":\"copy_for_unknown_external_tail_capacity\","
      "\"selection_context\":{\"threads\":" +
      std::to_string(execution_.threads) +
      ",\"avx2_fma\":" + (execution_.avx2_fma ? "true" : "false") +
      "},\"candidates\":[" + decisions_ + "]}";
  return report;
}
} // namespace executorch::backends::cpu
