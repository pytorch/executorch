// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/MemoryPlanning.h>

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

namespace ptn {
namespace {

struct Lifetime {
  size_t first = 0;
  size_t last = 0;
};

struct AllocationGroup {
  ValueId root = kInvalid;
  size_t size_bytes = 0;
  size_t alignment = 1;
  MemoryKind memory_kind = kInvalidMemoryKind;
  Lifetime lifetime;
  std::vector<ValueId> requested_values;
  bool initialized = false;
};

ValueId
alias_root(const Graph& graph, ValueId value_id, std::vector<ValueId>& roots) {
  ValueId current = value_id;
  while (!valid(roots[static_cast<size_t>(current)]) &&
         valid(graph.value(current).alias_id)) {
    current = graph.value(current).alias_id;
  }
  const ValueId root = valid(roots[static_cast<size_t>(current)])
      ? roots[static_cast<size_t>(current)]
      : current;
  current = value_id;
  while (!valid(roots[static_cast<size_t>(current)])) {
    roots[static_cast<size_t>(current)] = root;
    if (current == root) {
      break;
    }
    current = graph.value(current).alias_id;
  }
  return root;
}

bool is_persistent(const Value& value) {
  return value.role == ValueRole::Parameter ||
      value.role == ValueRole::Buffer ||
      value.role == ValueRole::ConstantTensor;
}

bool is_power_of_two(size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool lifetimes_overlap(const Lifetime& left, const Lifetime& right) {
  return !(left.last < right.first || right.last < left.first);
}

} // namespace

AllocationId MemoryPlan::allocation_id(ValueId value_id) const {
  if (!in_bounds(value_id, allocation_ids_.size())) {
    throw std::runtime_error("MemoryPlan::allocation_id: invalid value id");
  }
  return allocation_ids_[static_cast<size_t>(value_id)];
}

const PlannedAllocation& MemoryPlan::allocation(
    AllocationId allocation_id) const {
  if (!in_bounds(allocation_id, allocations_.size())) {
    throw std::runtime_error("MemoryPlan::allocation: invalid allocation id");
  }
  return allocations_[static_cast<size_t>(allocation_id)];
}

// cppcheck-suppress unusedFunction
MemoryPlan plan_memory(
    const Graph& graph,
    const std::vector<AllocationRequest>& requests) {
  validate_graph(graph);
  MemoryPlan plan;
  plan.allocation_ids_.assign(graph.values.size(), kInvalidAllocationId);
  if (requests.empty()) {
    return plan;
  }
  if (graph.schedule.empty()) {
    throw std::runtime_error("plan_memory: cannot plan an empty schedule");
  }

  std::vector<size_t> positions(
      graph.nodes.size(), std::numeric_limits<size_t>::max());
  for (size_t i = 0; i < graph.schedule.size(); ++i) {
    positions[static_cast<size_t>(graph.schedule[i])] = i;
  }
  std::vector<uint8_t> is_input(graph.values.size(), 0);
  std::vector<uint8_t> is_output(graph.values.size(), 0);
  for (ValueId id : graph.input_ids) {
    is_input[static_cast<size_t>(id)] = 1;
  }
  for (ValueId id : graph.output_ids) {
    is_output[static_cast<size_t>(id)] = 1;
  }

  std::vector<ValueId> roots(graph.values.size(), kInvalid);
  std::map<ValueId, std::vector<ValueId>> members;
  for (size_t i = 0; i < graph.values.size(); ++i) {
    const ValueId id = static_cast<ValueId>(i);
    roots[i] = alias_root(graph, id, roots);
    members[roots[i]].push_back(id);
  }

  const size_t graph_end = graph.schedule.size() - 1;
  auto lifetime = [&](ValueId value_id) -> std::optional<Lifetime> {
    const Value& value = graph.value(value_id);
    size_t first = std::numeric_limits<size_t>::max();
    size_t last = 0;
    if (is_input[static_cast<size_t>(value_id)] != 0 || is_persistent(value)) {
      first = 0;
    } else if (valid(value.producer_id)) {
      first = positions[static_cast<size_t>(value.producer_id)];
    }
    std::ranges::for_each(value.consumer_ids, [&](NodeId consumer) {
      last = std::max(last, positions[static_cast<size_t>(consumer)]);
    });
    if (is_output[static_cast<size_t>(value_id)] != 0 || is_persistent(value)) {
      last = graph_end;
    }
    if (first == std::numeric_limits<size_t>::max()) {
      return std::nullopt;
    }
    last = std::max(last, first);
    return Lifetime{first, last};
  };

  std::vector<uint8_t> requested(graph.values.size(), 0);
  std::map<ValueId, AllocationGroup> groups;
  for (const AllocationRequest& request : requests) {
    if (!in_bounds(request.value_id, graph.values.size())) {
      throw std::runtime_error("plan_memory: invalid value id");
    }
    const Value& value = graph.value(request.value_id);
    if (!value.is_tensor()) {
      throw std::runtime_error("plan_memory: request is not a tensor");
    }
    if (!is_power_of_two(request.alignment)) {
      throw std::runtime_error(
          "plan_memory: alignment must be a nonzero power of two");
    }
    if (!valid(request.memory_kind)) {
      throw std::runtime_error("plan_memory: invalid memory kind");
    }
    if (requested[static_cast<size_t>(request.value_id)] != 0) {
      throw std::runtime_error("plan_memory: duplicate value request");
    }
    requested[static_cast<size_t>(request.value_id)] = 1;

    const ValueId root = roots[static_cast<size_t>(request.value_id)];
    AllocationGroup& group = groups[root];
    if (group.initialized && group.memory_kind != request.memory_kind) {
      throw std::runtime_error(
          "plan_memory: alias group has incompatible memory kinds");
    }
    group.root = root;
    group.memory_kind = request.memory_kind;
    group.size_bytes = std::max(group.size_bytes, request.size_bytes);
    group.alignment = std::max(group.alignment, request.alignment);
    group.requested_values.push_back(request.value_id);
    group.initialized = true;
  }

  for (auto& [root, group] : groups) {
    bool first_member = true;
    for (ValueId member : members.at(root)) {
      const Value& value = graph.value(member);
      if (!value.is_tensor()) {
        continue;
      }
      const std::optional<Lifetime> member_lifetime = lifetime(member);
      if (!member_lifetime.has_value()) {
        if (requested[static_cast<size_t>(member)] != 0) {
          throw std::runtime_error(
              "plan_memory: requested value is not active");
        }
        continue;
      }
      if (first_member) {
        group.lifetime = *member_lifetime;
        first_member = false;
      } else {
        group.lifetime.first =
            std::min(group.lifetime.first, member_lifetime->first);
        group.lifetime.last =
            std::max(group.lifetime.last, member_lifetime->last);
      }
    }
    if (first_member) {
      throw std::runtime_error("plan_memory: alias group is not active");
    }
  }

  std::vector<AllocationGroup*> ordered_groups;
  ordered_groups.reserve(groups.size());
  for (auto& [root, group] : groups) {
    ordered_groups.push_back(&group);
  }
  std::sort(
      ordered_groups.begin(),
      ordered_groups.end(),
      [](const AllocationGroup* left, const AllocationGroup* right) {
        if (left->size_bytes != right->size_bytes) {
          return left->size_bytes > right->size_bytes;
        }
        return left->root < right->root;
      });

  std::vector<std::vector<Lifetime>> allocation_lifetimes;
  for (const AllocationGroup* group : ordered_groups) {
    AllocationId allocation_id = kInvalidAllocationId;
    for (size_t i = 0; i < plan.allocations_.size(); ++i) {
      if (plan.allocations_[i].memory_kind != group->memory_kind) {
        continue;
      }
      const bool overlaps = std::any_of(
          allocation_lifetimes[i].begin(),
          allocation_lifetimes[i].end(),
          [&](const Lifetime& existing) {
            return lifetimes_overlap(existing, group->lifetime);
          });
      if (!overlaps) {
        allocation_id = static_cast<AllocationId>(i);
        break;
      }
    }
    if (!valid(allocation_id)) {
      if (plan.allocations_.size() >
          static_cast<size_t>(std::numeric_limits<AllocationId>::max())) {
        throw std::runtime_error("plan_memory: allocation id space exhausted");
      }
      allocation_id = static_cast<AllocationId>(plan.allocations_.size());
      plan.allocations_.push_back(PlannedAllocation{
          .size_bytes = group->size_bytes,
          .alignment = group->alignment,
          .memory_kind = group->memory_kind,
      });
      allocation_lifetimes.emplace_back();
    }
    PlannedAllocation& allocation =
        plan.allocations_[static_cast<size_t>(allocation_id)];
    allocation.size_bytes = std::max(allocation.size_bytes, group->size_bytes);
    allocation.alignment = std::max(allocation.alignment, group->alignment);
    allocation.value_ids.insert(
        allocation.value_ids.end(),
        group->requested_values.begin(),
        group->requested_values.end());
    allocation_lifetimes[static_cast<size_t>(allocation_id)].push_back(
        group->lifetime);
    for (ValueId value_id : group->requested_values) {
      plan.allocation_ids_[static_cast<size_t>(value_id)] = allocation_id;
    }
  }
  return plan;
}

} // namespace ptn
