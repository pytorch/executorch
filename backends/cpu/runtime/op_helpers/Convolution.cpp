// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/op_helpers/Convolution.h>

#include <algorithm>

namespace executorch::backends::cpu {
namespace {
std::optional<std::vector<int64_t>> static_ints(const ptn::Argument& argument) {
  if (argument.kind() != ptn::ArgKind::IntList) {
    return std::nullopt;
  }
  const auto& list = argument.as_int_list();
  if (!std::all_of(list.ids.begin(), list.ids.end(), [](auto id) {
        return id == ptn::kInvalid;
      })) {
    return std::nullopt;
  }
  return list.values;
}
} // namespace

std::optional<Convolution> parse_convolution(const Kernel& node) {
  if (node.target != "torch.ops.aten.convolution.default" ||
      node.inputs.size() != 9 || node.outputs.size() != 1 ||
      node.outputs[0].kind != ptn::OutputValueKind::Tensor ||
      std::any_of(node.inputs.begin(), node.inputs.end(), [](const auto& in) {
        return in.mutated;
      })) {
    return std::nullopt;
  }
  const auto& input = node.inputs[0].arg;
  const auto& weight = node.inputs[1].arg;
  const auto& bias = node.inputs[2].arg;
  const auto stride = static_ints(node.inputs[3].arg);
  const auto padding = static_ints(node.inputs[4].arg);
  const auto dilation = static_ints(node.inputs[5].arg);
  const auto& transposed = node.inputs[6].arg;
  const auto output_padding = static_ints(node.inputs[7].arg);
  const auto& groups = node.inputs[8].arg;
  if (input.kind() != ptn::ArgKind::Tensor ||
      weight.kind() != ptn::ArgKind::Tensor ||
      (bias.kind() != ptn::ArgKind::None &&
       bias.kind() != ptn::ArgKind::Tensor) ||
      !stride || !padding || !dilation || !output_padding ||
      transposed.kind() != ptn::ArgKind::Bool ||
      transposed.as_bool().id != ptn::kInvalid ||
      groups.kind() != ptn::ArgKind::Int ||
      groups.as_int().id != ptn::kInvalid) {
    return std::nullopt;
  }
  return Convolution{
      input.as_tensor().id,
      weight.as_tensor().id,
      bias.kind() == ptn::ArgKind::None ? ptn::kInvalid : bias.as_tensor().id,
      node.outputs[0].value_id,
      *stride,
      *padding,
      *dilation,
      *output_padding,
      transposed.as_bool().value,
      groups.as_int().value};
}

} // namespace executorch::backends::cpu
