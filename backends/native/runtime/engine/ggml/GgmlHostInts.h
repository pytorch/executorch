// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/Node.h>

namespace ptn {

struct Method;

namespace ggml_detail {

struct InputSlot;

// A small integer tensor computed on the host, contiguous and row-major.
struct HostInts {
  std::vector<int64_t> sizes;
  std::vector<int64_t> data;
};

// Evaluates small integer subgraphs of graph inputs and constants on the
// host, e.g. cache positions and key indices, with the current input values.
// Returns nullopt for anything outside its op set or element budget.
class HostIntEvaluator {
 public:
  using ConstantReader = std::function<std::optional<HostInts>(ValueId)>;
  // Values that read no graph input are the same on every call.
  using StaticCache = std::unordered_map<ValueId, std::optional<HostInts>>;

  HostIntEvaluator(
      const Method& method,
      const std::vector<InputSlot>& inputs,
      ConstantReader constants,
      StaticCache& static_values);

  const HostInts* tensor(ValueId id);

 private:
  std::optional<int64_t> integer(const Argument& arg);
  std::optional<std::vector<int64_t>> runtime_sizes(ValueId id);
  const HostInts*
  arg_tensor(const Node& node, std::string_view name, size_t positional);
  std::optional<int64_t> arg_int(
      const Node& node,
      std::string_view name,
      size_t positional,
      std::optional<int64_t> fallback = std::nullopt);
  std::optional<HostInts> compute(ValueId id);

  const Method& method_;
  const std::vector<InputSlot>& inputs_;
  ConstantReader constants_;
  StaticCache& static_values_;
  std::unordered_map<ValueId, std::optional<HostInts>> memo_;
  bool reads_input_ = false;
};

} // namespace ggml_detail
} // namespace ptn
