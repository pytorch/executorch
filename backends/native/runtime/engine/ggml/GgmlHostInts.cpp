// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlHostInts.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
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

int64_t magnitude(int64_t value) {
  return value < -std::numeric_limits<int64_t>::max()
      ? std::numeric_limits<int64_t>::max()
      : std::abs(value);
}

// The largest magnitude among an integer input's values, at any size.
std::optional<int64_t> input_peak(const InputSlot& input) {
  const auto peak_of = [&input]<typename T>(T) {
    int64_t peak = 0;
    for (size_t offset = 0; offset + sizeof(T) <= input.bytes.size();
         offset += sizeof(T)) {
      T value;
      std::memcpy(&value, input.bytes.data() + offset, sizeof(T));
      peak = std::max(peak, magnitude(static_cast<int64_t>(value)));
    }
    return std::optional<int64_t>(peak);
  };
  switch (input.meta.dtype) {
    case ScalarType::Byte:
      return peak_of(uint8_t{});
    case ScalarType::Char:
      return peak_of(int8_t{});
    case ScalarType::Short:
      return peak_of(int16_t{});
    case ScalarType::Int:
      return peak_of(int32_t{});
    case ScalarType::Long:
      return peak_of(int64_t{});
    default:
      return std::nullopt;
  }
}

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
  const auto cached = [this](const std::optional<HostInts>& value) {
    if (!value) {
      return static_cast<const HostInts*>(nullptr);
    }
    peak_ = std::max(peak_, value->peak);
    return &*value;
  };
  if (const auto found = static_values_.find(id);
      found != static_values_.end()) {
    return cached(found->second);
  }
  if (const auto found = memo_.find(id); found != memo_.end()) {
    // Only input-dependent values are memoized per call.
    reads_input_ = true;
    return cached(found->second);
  }
  const bool outer_reads_input = reads_input_;
  const int64_t outer_peak = peak_;
  reads_input_ = false;
  peak_ = 0;
  std::optional<HostInts> result;
  try {
    result = compute(id);
  } catch (const std::exception&) {
    result.reset();
  }
  if (result && result->data.size() > kHostEvalMaxElements) {
    result.reset();
  }
  if (result) {
    for (const int64_t value : result->data) {
      peak_ = std::max(peak_, magnitude(value));
    }
    result->peak = peak_;
  }
  auto& cache = reads_input_ ? memo_ : static_values_;
  reads_input_ = reads_input_ || outer_reads_input;
  peak_ = outer_peak;
  const auto it = cache.insert_or_assign(id, std::move(result)).first;
  return cached(it->second);
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
        peak_ = std::max(peak_, magnitude(*value));
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
  if (op == "aten.slice.Tensor" || op == "aten.slice_copy.Tensor") {
    const HostInts* self = arg_tensor(node, "self", 0);
    const std::optional<int64_t> dim = arg_int(node, "dim", 1, 0);
    if (self == nullptr || !dim || arg_int(node, "step", 4, 1) != 1) {
      return std::nullopt;
    }
    const auto rank = static_cast<int64_t>(self->sizes.size());
    const int64_t axis = *dim < 0 ? *dim + rank : *dim;
    if (axis < 0 || axis >= rank) {
      return std::nullopt;
    }
    const int64_t extent = self->sizes[axis];
    const std::optional<int64_t> start = arg_int(node, "start", 2, 0);
    const std::optional<int64_t> end = arg_int(node, "end", 3, extent);
    if (!start || !end) {
      return std::nullopt;
    }
    // Python bounds: negative ones count from the end, then clamp.
    const auto bound = [extent](int64_t v) {
      return std::clamp<int64_t>(v < 0 ? v + extent : v, 0, extent);
    };
    const int64_t begin = bound(*start);
    const int64_t count = std::max<int64_t>(bound(*end) - begin, 0);
    HostInts out{self->sizes, {}};
    out.sizes[axis] = count;
    if (checked_numel(self->sizes) != self->data.size()) {
      return std::nullopt;
    }
    // An empty input or slice copies nothing, however large its other extents.
    if (self->data.empty() || count == 0) {
      return out;
    }
    // Every extent is now positive, and their product is the budgeted size.
    int64_t inner = 1;
    for (int64_t d = axis + 1; d < rank; ++d) {
      inner *= self->sizes[d];
    }
    const int64_t outer =
        static_cast<int64_t>(self->data.size()) / (extent * inner);
    for (int64_t o = 0; o < outer; ++o) {
      const auto first = self->data.begin() + (o * extent + begin) * inner;
      out.data.insert(out.data.end(), first, first + count * inner);
    }
    return out;
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
  if (op == "aten.select.int" || op == "aten.select_copy.int") {
    const HostInts* self = arg_tensor(node, "self", 0);
    std::optional<int64_t> dim = arg_int(node, "dim", 1);
    std::optional<int64_t> index = arg_int(node, "index", 2);
    if (self == nullptr || !dim || !index || self->sizes.empty()) {
      return std::nullopt;
    }
    const auto rank = static_cast<int64_t>(self->sizes.size());
    if (*dim < 0) {
      *dim += rank;
    }
    if (*dim < 0 || *dim >= rank) {
      return std::nullopt;
    }
    const int64_t extent = self->sizes[static_cast<size_t>(*dim)];
    if (*index < 0) {
      *index += extent;
    }
    if (*index < 0 || *index >= extent) {
      return std::nullopt;
    }
    size_t inner = 1;
    for (int64_t d = *dim + 1; d < rank; ++d) {
      inner *= static_cast<size_t>(self->sizes[static_cast<size_t>(d)]);
    }
    const size_t outer =
        self->data.size() / (inner * static_cast<size_t>(extent));
    HostInts out;
    out.sizes = self->sizes;
    out.sizes.erase(out.sizes.begin() + *dim);
    for (size_t o = 0; o < outer; ++o) {
      const size_t offset =
          (o * static_cast<size_t>(extent) + static_cast<size_t>(*index)) *
          inner;
      const auto first =
          self->data.begin() + static_cast<std::ptrdiff_t>(offset);
      out.data.insert(
          out.data.end(), first, first + static_cast<std::ptrdiff_t>(inner));
    }
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

std::optional<int64_t> HostIntEvaluator::peak_bound(ValueId id) {
  return bound(id, nullptr);
}

std::optional<int64_t> HostIntEvaluator::shape_bound(
    ValueId id,
    const std::unordered_map<ValueId, int64_t>& input_peaks) {
  return bound(id, &input_peaks);
}

std::optional<int64_t> HostIntEvaluator::bound(
    ValueId id,
    const std::unordered_map<ValueId, int64_t>* input_peaks) {
  const bool read_values = input_peaks == nullptr;
  const auto input = std::ranges::find(inputs_, id, &InputSlot::id);
  if (input != inputs_.end()) {
    if (read_values) {
      return input_peak(*input);
    }
    const auto found = input_peaks->find(id);
    return found != input_peaks->end() ? std::optional(found->second)
                                       : std::nullopt;
  }
  const NodeId producer = method_.graph.value(id).producer_id;
  const bool constant =
      !valid(producer) || !method_.graph.node(producer).is_call();
  if (constant || read_values) {
    if (const HostInts* value = tensor(id)) {
      return value->peak;
    }
  }
  if (constant) {
    return std::nullopt;
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
    if (!start || !end || !step || *step <= 0) {
      return std::nullopt;
    }
    // Every value lies between start and end.
    return std::max(magnitude(*start), magnitude(*end));
  }
  if (op == "aten.full.default" || op == "aten.full_like.default") {
    const std::optional<int64_t> fill = arg_int(node, "fill_value", 1);
    return fill ? std::optional(magnitude(*fill)) : std::nullopt;
  }
  const NamedArgument* self = find_arg(node, "self", 0);
  if (self == nullptr || self->arg.kind() != ArgKind::Tensor) {
    return std::nullopt;
  }
  const std::optional<int64_t> self_bound =
      bound(self->arg.as_tensor().id, input_peaks);
  if (!self_bound) {
    return std::nullopt;
  }
  if (op == "aten.add.Tensor" || op == "aten.sub.Tensor") {
    const NamedArgument* other = find_arg(node, "other", 1);
    if (other == nullptr || arg_int(node, "alpha", 2, 1) != 1) {
      return std::nullopt;
    }
    std::optional<int64_t> other_bound;
    if (other->arg.kind() == ArgKind::Tensor) {
      other_bound = bound(other->arg.as_tensor().id, input_peaks);
    } else if (const std::optional<int64_t> value = integer(other->arg)) {
      other_bound = magnitude(*value);
    }
    if (!other_bound) {
      return std::nullopt;
    }
    int64_t sum = 0;
    return __builtin_add_overflow(*self_bound, *other_bound, &sum)
        ? std::numeric_limits<int64_t>::max()
        : sum;
  }
  return keeps_integer_values(op) ? self_bound : std::nullopt;
}

std::optional<int64_t> HostIntEvaluator::static_peak_bound(ValueId id) {
  if (const auto input = std::ranges::find(inputs_, id, &InputSlot::id);
      input != inputs_.end()) {
    return is_integer(input->meta.dtype) ? std::optional<int64_t>(0)
                                         : std::nullopt;
  }
  const NodeId producer = method_.graph.value(id).producer_id;
  if (!valid(producer) || !method_.graph.node(producer).is_call()) {
    const HostInts* constant = tensor(id);
    return constant ? std::optional(constant->peak) : std::nullopt;
  }
  const Node& node = method_.graph.node(producer);
  const std::string op = op_key(node.target);
  // An integer argument is a literal, or a size each run resolves.
  const auto is_int = [&node](std::string_view name, size_t positional) {
    const NamedArgument* arg = find_arg(node, name, positional);
    return arg != nullptr && arg->arg.kind() == ArgKind::Int;
  };
  const auto literal = [&node](
                           std::string_view name,
                           size_t positional) -> std::optional<int64_t> {
    const NamedArgument* arg = find_arg(node, name, positional);
    if (arg == nullptr || arg->arg.kind() != ArgKind::Int ||
        valid(arg->arg.as_int().id)) {
      return std::nullopt;
    }
    return arg->arg.as_int().value;
  };
  const auto absent = [&node](std::string_view name, size_t positional) {
    const NamedArgument* arg = find_arg(node, name, positional);
    return arg == nullptr || arg->arg.kind() == ArgKind::None;
  };
  if (op == "aten.arange.start_step" || op == "aten.arange.start" ||
      op == "aten.arange.default") {
    const bool has_start = op != "aten.arange.default";
    const bool has_step = op == "aten.arange.start_step" && !absent("step", 2);
    const bool bounded = (!has_start || is_int("start", 0)) &&
        is_int("end", has_start ? 1 : 0) &&
        (!has_step ||
         (is_int("step", 2) && literal("step", 2).value_or(1) > 0));
    return bounded ? std::optional<int64_t>(0) : std::nullopt;
  }
  if (op == "aten.full.default" || op == "aten.full_like.default") {
    return is_int("fill_value", 1)
        ? std::optional(magnitude(literal("fill_value", 1).value_or(0)))
        : std::nullopt;
  }
  const NamedArgument* self = find_arg(node, "self", 0);
  if (self == nullptr || self->arg.kind() != ArgKind::Tensor) {
    return std::nullopt;
  }
  const std::optional<int64_t> self_peak =
      static_peak_bound(self->arg.as_tensor().id);
  if (!self_peak) {
    return std::nullopt;
  }
  if (op == "aten.add.Tensor" || op == "aten.sub.Tensor") {
    const NamedArgument* other = find_arg(node, "other", 1);
    if (other == nullptr ||
        !(absent("alpha", 2) ||
          (is_int("alpha", 2) && literal("alpha", 2).value_or(1) == 1))) {
      return std::nullopt;
    }
    std::optional<int64_t> other_peak;
    if (other->arg.kind() == ArgKind::Tensor) {
      other_peak = static_peak_bound(other->arg.as_tensor().id);
    } else if (is_int("other", 1)) {
      other_peak = magnitude(literal("other", 1).value_or(0));
    }
    if (!other_peak) {
      return std::nullopt;
    }
    return std::max(*self_peak, *other_peak);
  }
  return keeps_integer_values(op) ? self_peak : std::nullopt;
}

} // namespace ptn::ggml_detail
