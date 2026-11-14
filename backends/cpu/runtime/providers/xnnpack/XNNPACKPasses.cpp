// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/providers/xnnpack/XNNPACKPasses.h>

#include <algorithm>
#include <limits>
#include <string_view>
#include <unordered_set>

namespace executorch::backends::cpu::xnnpack {
namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();

const ptn::Argument& arg(const Kernel& node, size_t index) {
  return node.inputs.at(index).arg;
}

bool is_value(const ptn::Argument& argument, ptn::ValueId id) {
  return argument.kind() == ptn::ArgKind::Tensor &&
      argument.as_tensor().id == id;
}

class HardswishMatcher {
 public:
  HardswishMatcher(
      const KernelRegion& region,
      const ptn::Graph& graph,
      const std::vector<Buffer>& buffers)
      : graph_(graph),
        buffers_(buffers),
        nodes_(region.nodes.begin(), region.nodes.end()),
        outputs_(region.outputs.begin(), region.outputs.end()) {}

  // Matches `div(mul(x, clamp...(add(x, 3))), 6)` ending at `div_id`.
  bool match(KernelId div_id, HardswishChain& chain) const {
    const auto& div = graph_.node(div_id);
    if (div.target != "torch.ops.aten.div.Tensor" || div.inputs.size() != 2 ||
        arg(div, 0).kind() != ptn::ArgKind::Tensor ||
        !scalar_constant(arg(div, 1), 6)) {
      return false;
    }
    const auto mul_id = private_producer(
        arg(div, 0).as_tensor().id, "torch.ops.aten.mul.Tensor");
    if (mul_id == ptn::kInvalid) {
      return false;
    }
    const auto& mul = graph_.node(mul_id);
    if (mul.inputs.size() != 2 || arg(mul, 0).kind() != ptn::ArgKind::Tensor ||
        arg(mul, 1).kind() != ptn::ArgKind::Tensor) {
      return false;
    }
    for (size_t gate : {1, 0}) {
      chain.input = arg(mul, 1 - gate).as_tensor().id;
      chain.nodes.clear();
      if (match_gate(arg(mul, gate).as_tensor().id, chain)) {
        chain.output = div.outputs.at(0).value_id;
        chain.nodes.push_back(mul_id);
        chain.nodes.push_back(div_id);
        return graph_.value(chain.input).tensor_meta().sizes ==
            graph_.value(chain.output).tensor_meta().sizes;
      }
    }
    return false;
  }

 private:
  // Matches `clamp...(add(chain.input, 3))` with combined bounds [0, 6].
  bool match_gate(ptn::ValueId gate, HardswishChain& chain) const {
    float min = -kInfinity;
    float max = kInfinity;
    auto value = gate;
    while (true) {
      const KernelId clamp_id =
          private_producer(value, "torch.ops.aten.clamp.default");
      if (clamp_id == ptn::kInvalid) {
        break;
      }
      const auto& clamp = graph_.node(clamp_id);
      float lower = 0;
      float upper = 0;
      if (clamp.inputs.size() != 3 ||
          arg(clamp, 0).kind() != ptn::ArgKind::Tensor ||
          !clamp_bound(arg(clamp, 1), -kInfinity, lower) ||
          !clamp_bound(arg(clamp, 2), kInfinity, upper)) {
        return false;
      }
      min = std::max(min, lower);
      max = std::min(max, upper);
      chain.nodes.push_back(clamp_id);
      value = arg(clamp, 0).as_tensor().id;
    }
    if (chain.nodes.empty() || min != 0 || max != 6) {
      return false;
    }
    const auto add_id = private_producer(value, "torch.ops.aten.add.Tensor");
    if (add_id == ptn::kInvalid) {
      return false;
    }
    const auto& add = graph_.node(add_id);
    if (add.inputs.size() != 3 || arg(add, 2).kind() != ptn::ArgKind::Int ||
        arg(add, 2).as_int().value != 1 ||
        !((is_value(arg(add, 0), chain.input) &&
           scalar_constant(arg(add, 1), 3)) ||
          (is_value(arg(add, 1), chain.input) &&
           scalar_constant(arg(add, 0), 3)))) {
      return false;
    }
    chain.nodes.push_back(add_id);
    return true;
  }

  // The `target` node in this region producing `value` for a single consumer.
  KernelId private_producer(ptn::ValueId value, std::string_view target) const {
    const auto& produced = graph_.value(value);
    if (outputs_.count(value) || produced.consumer_ids.size() != 1 ||
        !nodes_.count(produced.producer_id) ||
        graph_.node(produced.producer_id).target != target) {
      return ptn::kInvalid;
    }
    return produced.producer_id;
  }

  bool scalar_constant(const ptn::Argument& argument, float expected) const {
    if (argument.kind() != ptn::ArgKind::Tensor) {
      return false;
    }
    const auto id = argument.as_tensor().id;
    const auto& value = graph_.value(id);
    if ((value.role != ptn::ValueRole::Parameter &&
         value.role != ptn::ValueRole::ConstantTensor &&
         value.role != ptn::ValueRole::Buffer) ||
        value.tensor_meta().dtype != ptn::ScalarType::Float ||
        value.tensor_meta().numel() != 1) {
      return false;
    }
    const auto& buffer = buffers_.at(id);
    return buffer.data && buffer.readable_bytes >= sizeof(float) &&
        *static_cast<const float*>(buffer.data) == expected;
  }

  const ptn::Graph& graph_;
  const std::vector<Buffer>& buffers_;
  std::unordered_set<KernelId> nodes_;
  std::unordered_set<ptn::ValueId> outputs_;
};

} // namespace

bool clamp_bound(const ptn::Argument& argument, float fallback, float& bound) {
  switch (argument.kind()) {
    case ptn::ArgKind::None:
      bound = fallback;
      return true;
    case ptn::ArgKind::Int:
      bound = static_cast<float>(argument.as_int().value);
      return argument.as_int().id == ptn::kInvalid;
    case ptn::ArgKind::Float:
      bound = static_cast<float>(argument.as_float().value);
      return argument.as_float().id == ptn::kInvalid;
    default:
      return false;
  }
}

std::vector<HardswishChain> find_hardswish_chains(
    const KernelRegion& region,
    const ptn::Graph& graph,
    const std::vector<Buffer>& buffers) {
  const HardswishMatcher matcher(region, graph, buffers);
  std::vector<HardswishChain> chains;
  for (auto id : region.nodes) {
    HardswishChain chain;
    if (matcher.match(id, chain)) {
      chains.push_back(std::move(chain));
    }
  }
  return chains;
}

} // namespace executorch::backends::cpu::xnnpack
