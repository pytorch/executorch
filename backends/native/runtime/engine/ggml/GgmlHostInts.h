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
  // The largest magnitude among these values and every integer they were
  // computed from.
  int64_t peak = 0;
};

// The most elements a host-evaluated tensor, or a constant it reads, holds.
inline constexpr size_t kHostEvalMaxElements = size_t{1} << 16;

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
  // A bound on the magnitude of every integer `id` is computed from: its
  // peak where the host computes it, else from the arguments of an arange or
  // full, or the bounds of the operands of an add, sub, cast, view or repeat.
  // nullopt outside that op set.
  std::optional<int64_t> peak_bound(ValueId id);
  // peak_bound without the values of graph inputs: from constants, integer
  // literals, input shapes and `input_peaks`, so it holds for every run with
  // these shapes. `input_peaks` may name only inputs every run rejects outside
  // [-peak, peak] before computing, such as row indices of a table of at most
  // 2^24 rows (KV cache write positions); any other input has no bound here.
  std::optional<int64_t> shape_bound(
      ValueId id,
      const std::unordered_map<ValueId, int64_t>& input_peaks);
  // What peak_bound can know of `id` from the graph alone: nullopt if no run
  // can bound it, else a floor of every run's bound, the largest magnitude
  // among the constants, full values and added literals it is computed from;
  // inputs and aranges count as 0.
  std::optional<int64_t> static_peak_bound(ValueId id);

 private:
  std::optional<int64_t> integer(const Argument& arg);
  // With `input_peaks` null, reads the inputs' values.
  std::optional<int64_t> bound(
      ValueId id,
      const std::unordered_map<ValueId, int64_t>* input_peaks);
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
  int64_t peak_ = 0;
};

} // namespace ggml_detail
} // namespace ptn
