// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlHostInts.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <utility>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlGraphBuilder.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlTypes.h>
#include <executorch/backends/native/runtime/graph/Graph.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/Value.h>

namespace ptn::ggml_detail {
namespace {

constexpr size_t kHostEvalMaxElements = size_t{1} << 16;

// a op b for equal shapes, or with b a single value; nullopt if fn overflows.
std::optional<HostInts> broadcast_ints(
    const HostInts& a,
    const HostInts& b,
    bool (*fn)(int64_t, int64_t, int64_t*)) {
  if (a.sizes != b.sizes && b.data.size() != 1) {
    return std::nullopt;
  }
  HostInts out = a;
  for (size_t i = 0; i < out.data.size(); ++i) {
    if (fn(a.data[i], b.data[b.data.size() == 1 ? 0 : i], &out.data[i])) {
      return std::nullopt;
    }
  }
  return out;
}

} // namespace

HostIntEvaluator::HostIntEvaluator(
    const Method& method,
    const std::vector<InputSlot>& inputs,
    ConstantReader constants,
    StaticCache& static_values)
    : method_(method),
      inputs_(inputs),
      constants_(std::move(constants)),
      static_values_(static_values) {}

// nullptr when the value is outside the op set or element budget. The
// pointee lives as long as this evaluator (or the static cache).
const HostInts* HostIntEvaluator::tensor(ValueId id) {
  if (const auto found = static_values_.find(id);
      found != static_values_.end()) {
    return found->second ? &*found->second : nullptr;
  }
  if (const auto found = memo_.find(id); found != memo_.end()) {
    // Only input-dependent values are memoized per call.
    reads_input_ = true;
    return found->second ? &*found->second : nullptr;
  }
  const bool outer_reads_input = reads_input_;
  reads_input_ = false;
  std::optional<HostInts> result;
  try {
    result = compute(id);
  } catch (const std::exception&) {
    result.reset();
  }
  if (result && result->data.size() > kHostEvalMaxElements) {
    result.reset();
  }
  auto& cache = reads_input_ ? memo_ : static_values_;
  reads_input_ = reads_input_ || outer_reads_input;
  const auto it = cache.insert_or_assign(id, std::move(result)).first;
  return it->second ? &*it->second : nullptr;
}

std::optional<int64_t> HostIntEvaluator::integer(const Argument& arg) {
  if (arg.kind() != ArgKind::Int) {
    return std::nullopt;
  }
  const IntArg& value = arg.as_int();
  if (!valid(value.id)) {
    return value.value;
  }
  const NodeId producer = method_.graph.value(value.id).producer_id;
  if (!valid(producer)) {
    return std::nullopt;
  }
  const Node& node = method_.graph.node(producer);
  const std::string op = op_key(node.target);
  if (op == "aten.sym_size.int") {
    const NamedArgument* self = find_arg(node, "self", 0);
    const NamedArgument* dim = find_arg(node, "dim", 1);
    if (self == nullptr || self->arg.kind() != ArgKind::Tensor ||
        dim == nullptr) {
      return std::nullopt;
    }
    const std::optional<std::vector<int64_t>> sizes =
        runtime_sizes(self->arg.as_tensor().id);
    const std::optional<int64_t> d = integer(dim->arg);
    if (!sizes || !d) {
      return std::nullopt;
    }
    return (*sizes)[normalize_dim(*d, sizes->size())];
  }
  return std::nullopt;
}

// Shapes of graph inputs are known before evaluating any value.
std::optional<std::vector<int64_t>> HostIntEvaluator::runtime_sizes(
    ValueId id) {
  const auto input = std::ranges::find(inputs_, id, &InputSlot::id);
  if (input != inputs_.end()) {
    reads_input_ = true;
    return input->sizes;
  }
  const HostInts* value = tensor(id);
  return value ? std::optional(value->sizes) : std::nullopt;
}

const HostInts* HostIntEvaluator::arg_tensor(
    const Node& node,
    std::string_view name,
    size_t positional) {
  const NamedArgument* arg = find_arg(node, name, positional);
  if (arg == nullptr || arg->arg.kind() != ArgKind::Tensor ||
      !valid(arg->arg.as_tensor().id)) {
    return nullptr;
  }
  return tensor(arg->arg.as_tensor().id);
}

std::optional<int64_t> HostIntEvaluator::arg_int(
    const Node& node,
    std::string_view name,
    size_t positional,
    std::optional<int64_t> fallback) {
  const NamedArgument* arg = find_arg(node, name, positional);
  if (arg == nullptr || arg->arg.kind() == ArgKind::None) {
    return fallback;
  }
  return integer(arg->arg);
}

std::optional<HostInts> HostIntEvaluator::compute(ValueId id) {
  const auto input = std::ranges::find(inputs_, id, &InputSlot::id);
  if (input != inputs_.end()) {
    reads_input_ = true;
    HostInts out{input->sizes, {}};
    if (input->meta.dtype == ScalarType::Long) {
      out.data.resize(input->bytes.size() / sizeof(int64_t));
      std::memcpy(out.data.data(), input->bytes.data(), input->bytes.size());
    } else if (input->meta.dtype == ScalarType::Int) {
      std::vector<int32_t> narrow(input->bytes.size() / sizeof(int32_t));
      std::memcpy(narrow.data(), input->bytes.data(), input->bytes.size());
      out.data.assign(narrow.begin(), narrow.end());
    } else {
      return std::nullopt;
    }
    if (out.data.size() != checked_numel(out.sizes)) {
      return std::nullopt;
    }
    return out;
  }
  const NodeId producer = method_.graph.value(id).producer_id;
  if (!valid(producer) || !method_.graph.node(producer).is_call()) {
    return constants_(id);
  }
  const Node& node = method_.graph.node(producer);
  const std::string op = op_key(node.target);
  if (op == "aten.arange.start_step" || op == "aten.arange.start" ||
      op == "aten.arange.default") {
    const bool has_start = op != "aten.arange.default";
    const std::optional<int64_t> start =
        has_start ? arg_int(node, "start", 0) : std::optional<int64_t>(0);
    const std::optional<int64_t> end = arg_int(node, "end", has_start ? 1 : 0);
    const std::optional<int64_t> step = op == "aten.arange.start_step"
        ? arg_int(node, "step", 2, 1)
        : std::optional<int64_t>(1);
    if (!start || !end || !step || *step <= 0 || *end < *start ||
        static_cast<size_t>((*end - *start + *step - 1) / *step) >
            kHostEvalMaxElements) {
      return std::nullopt;
    }
    HostInts out;
    for (int64_t v = *start; v < *end; v += *step) {
      out.data.push_back(v);
    }
    out.sizes = {static_cast<int64_t>(out.data.size())};
    return out;
  }
  if (op == "aten.add.Tensor" || op == "aten.sub.Tensor") {
    const HostInts* self = arg_tensor(node, "self", 0);
    if (self == nullptr || arg_int(node, "alpha", 2, 1) != 1) {
      return std::nullopt;
    }
    const NamedArgument* other = find_arg(node, "other", 1);
    HostInts scalar;
    const HostInts* rhs = nullptr;
    if (other != nullptr && other->arg.kind() == ArgKind::Tensor) {
      rhs = tensor(other->arg.as_tensor().id);
    } else if (other != nullptr) {
      if (const std::optional<int64_t> value = integer(other->arg)) {
        scalar = HostInts{{}, {*value}};
        rhs = &scalar;
      }
    }
    if (rhs == nullptr) {
      return std::nullopt;
    }
    using Fn = bool (*)(int64_t, int64_t, int64_t*);
    const Fn fn = op == "aten.add.Tensor"
        ? Fn([](int64_t a, int64_t b, int64_t* out) {
            return __builtin_add_overflow(a, b, out);
          })
        : Fn([](int64_t a, int64_t b, int64_t* out) {
            return __builtin_sub_overflow(a, b, out);
          });
    return broadcast_ints(*self, *rhs, fn);
  }
  if (op == "aten.clone.default" || op == "aten.alias.default" ||
      op == "aten.alias_copy.default" || op == "aten.contiguous.default") {
    const HostInts* self = arg_tensor(node, "self", 0);
    return self ? std::optional(*self) : std::nullopt;
  }
  if (op == "aten.unsqueeze.default" || op == "aten.unsqueeze_copy.default") {
    const HostInts* self = arg_tensor(node, "self", 0);
    std::optional<int64_t> dim = arg_int(node, "dim", 1);
    if (self == nullptr || !dim) {
      return std::nullopt;
    }
    const int64_t rank = static_cast<int64_t>(self->sizes.size()) + 1;
    if (*dim < 0) {
      *dim += rank;
    }
    if (*dim < 0 || *dim >= rank) {
      return std::nullopt;
    }
    HostInts out = *self;
    out.sizes.insert(out.sizes.begin() + *dim, 1);
    return out;
  }
  if (op == "aten.view.default" || op == "aten.view_copy.default" ||
      op == "aten.reshape.default" || op == "aten._unsafe_view.default") {
    const HostInts* self = arg_tensor(node, "self", 0);
    const NamedArgument* shape = find_arg(node, "shape", 1);
    if (self == nullptr || shape == nullptr ||
        shape->arg.kind() != ArgKind::IntList) {
      return std::nullopt;
    }
    std::vector<int64_t> sizes;
    for (size_t i = 0; i < shape->arg.as_int_list().values.size(); ++i) {
      const IntListArg& list = shape->arg.as_int_list();
      const std::optional<int64_t> v = list.ids.empty() || !valid(list.ids[i])
          ? std::optional<int64_t>(list.values[i])
          : integer(IntArg{list.values[i], list.ids[i]});
      if (!v) {
        return std::nullopt;
      }
      sizes.push_back(*v);
    }
    const auto inferred = std::ranges::find(sizes, -1);
    if (inferred != sizes.end()) {
      *inferred = 1;
      const size_t known = checked_numel(sizes);
      if (known == 0 || self->data.size() % known != 0) {
        return std::nullopt;
      }
      *inferred = static_cast<int64_t>(self->data.size() / known);
    }
    if (checked_numel(sizes) != self->data.size()) {
      return std::nullopt;
    }
    return HostInts{std::move(sizes), self->data};
  }
  return std::nullopt;
}

} // namespace ptn::ggml_detail
