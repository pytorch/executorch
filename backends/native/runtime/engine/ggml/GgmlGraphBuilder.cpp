// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlGraphBuilder.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <ggml.h>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlTypes.h>
#include <executorch/backends/native/runtime/graph/Graph.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/Value.h>

namespace ptn::ggml_detail {
namespace {

constexpr std::string_view kTorchOpsPrefix = "torch.ops.";

} // namespace

std::string op_key(std::string_view target) {
  if (target.starts_with(kTorchOpsPrefix)) {
    target.remove_prefix(kTorchOpsPrefix.size());
  }
  return std::string(target);
}

const NamedArgument*
find_arg(const Node& node, std::string_view name, size_t positional) {
  const auto named = std::find_if(
      node.inputs.begin(), node.inputs.end(), [&](const NamedArgument& arg) {
        return arg.name == name;
      });
  if (named != node.inputs.end()) {
    return &*named;
  }
  return positional < node.inputs.size() ? &node.inputs[positional] : nullptr;
}

const NamedArgument&
require_arg(const Node& node, std::string_view name, size_t positional) {
  const NamedArgument* arg = find_arg(node, name, positional);
  if (arg == nullptr) {
    throw std::runtime_error(
        "ggml: op '" + node.target + "' is missing argument '" +
        std::string(name) + "'");
  }
  return *arg;
}

std::vector<ValueId> output_ids(const Node& node) {
  std::vector<ValueId> ids;
  for (const Output& output : node.outputs) {
    if (output.kind == OutputValueKind::TensorList) {
      std::ranges::copy_if(
          output.elem_ids, std::back_inserter(ids), [](ValueId id) {
            return valid(id);
          });
    } else if (valid(output.value_id)) {
      ids.push_back(output.value_id);
    }
  }
  return ids;
}

static std::vector<int64_t> broadcast_shape(
    const std::vector<int64_t>& a,
    const std::vector<int64_t>& b) {
  const size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t x = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
    const int64_t y = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
    if (x != y && x != 1 && y != 1) {
      throw std::runtime_error("ggml: incompatible broadcast shapes");
    }
    out[i] = x == 1 ? y : x;
  }
  return out;
}

int64_t normalize_dim(int64_t dim, size_t rank) {
  if (dim < 0) {
    dim += static_cast<int64_t>(rank);
  }
  if (dim < 0 || dim >= static_cast<int64_t>(rank)) {
    throw std::runtime_error("ggml: tensor dimension is out of range");
  }
  return dim;
}

const ggml_tensor* storage_root(const ggml_tensor* tensor) {
  while (tensor->view_src != nullptr) {
    tensor = tensor->view_src;
  }
  return tensor;
}

// ggml's row gathers and scatters do not bounds-check: a stray row aborts on
// the CPU and reads or writes out of bounds on GPUs. Input bytes are empty
// while compiling.
void check_index_bytes(const InputSlot& input, int64_t rows) {
  if (input.bytes.empty()) {
    return;
  }
  const auto in_range = [rows](int64_t row) { return row >= 0 && row < rows; };
  bool ok = true;
  if (input.meta.dtype == ScalarType::Long) {
    std::vector<int64_t> indices(input.bytes.size() / sizeof(int64_t));
    std::memcpy(indices.data(), input.bytes.data(), input.bytes.size());
    ok = std::ranges::all_of(indices, in_range);
  } else {
    std::vector<int32_t> indices(input.bytes.size() / sizeof(int32_t));
    std::memcpy(indices.data(), input.bytes.data(), input.bytes.size());
    ok = std::ranges::all_of(indices, in_range);
  }
  if (!ok) {
    throw std::out_of_range(
        "ggml: row index is outside [0, " + std::to_string(rows) + ")");
  }
}

// cppcheck-suppress unusedFunction
ggml_tensor* reshape(
    ggml_context* ctx,
    ggml_tensor* tensor,
    const std::vector<int64_t>& sizes) {
  if (!ggml_is_contiguous(tensor)) {
    tensor = ggml_cont(ctx, tensor);
  }
  const auto ne = reversed_extents(sizes);
  switch (sizes.size()) {
    case 0:
    case 1:
      return ggml_reshape_1d(ctx, tensor, ne[0]);
    case 2:
      return ggml_reshape_2d(ctx, tensor, ne[0], ne[1]);
    case 3:
      return ggml_reshape_3d(ctx, tensor, ne[0], ne[1], ne[2]);
    case 4:
      return ggml_reshape_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3]);
    default:
      throw std::runtime_error("ggml: reshape supports rank 0 through 4");
  }
}

// cppcheck-suppress unusedFunction
ggml_tensor* view_with_shape(
    ggml_context* ctx,
    ggml_tensor* tensor,
    const std::vector<int64_t>& sizes,
    size_t offset) {
  const auto ne = reversed_extents(sizes);
  switch (sizes.size()) {
    case 0:
    case 1:
      return ggml_view_1d(ctx, tensor, ne[0], offset);
    case 2:
      return ggml_view_2d(ctx, tensor, ne[0], ne[1], tensor->nb[1], offset);
    case 3:
      return ggml_view_3d(
          ctx,
          tensor,
          ne[0],
          ne[1],
          ne[2],
          tensor->nb[1],
          tensor->nb[2],
          offset);
    case 4:
      return ggml_view_4d(
          ctx,
          tensor,
          ne[0],
          ne[1],
          ne[2],
          ne[3],
          tensor->nb[1],
          tensor->nb[2],
          tensor->nb[3],
          offset);
    default:
      throw std::runtime_error("ggml: view supports rank 0 through 4");
  }
}

GraphBuilder::GraphBuilder(
    const Method& method,
    const std::vector<InputSlot>& inputs,
    const std::vector<OutputSlot>& outputs,
    const std::unordered_map<ValueId, ggml_tensor*>& persistent,
    const std::unordered_map<ValueId, std::vector<uint8_t>>& rope_factor_bytes,
    const FoldedBatchNorms& batch_norms,
    std::unordered_map<ValueId, int64_t> key_extents,
    size_t graph_size)
    : method_(method),
      inputs_(inputs),
      outputs_(outputs),
      persistent_(persistent),
      rope_factor_bytes_(rope_factor_bytes),
      batch_norms_(batch_norms),
      key_extents_(std::move(key_extents)),
      graph_size_(graph_size),
      values_(method.graph.values.size()) {
  for (const auto& [id, tensor] : persistent) {
    persistent_storage_.insert(tensor);
  }
}

GraphInstance GraphBuilder::build() {
  // Metadata room for the tensors lowering creates: a generous allowance per
  // value and per node for views, casts and shape helpers, plus a reserve for
  // per-graph tensors such as masks. ggml aborts if a graph outgrows it.
  constexpr size_t kTensorsPerValue = 12;
  constexpr size_t kTensorsPerNode = 8;
  constexpr size_t kReservedTensors = 256;
  const size_t tensor_budget = method_.graph.values.size() * kTensorsPerValue +
      method_.graph.nodes.size() * kTensorsPerNode + kReservedTensors;
  ggml_init_params params{
      .mem_size = ggml_tensor_overhead() * tensor_budget +
          ggml_graph_overhead_custom(graph_size_, false),
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  instance_.context.reset(ggml_init(params));
  if (!instance_.context) {
    throw std::runtime_error("ggml: failed to allocate graph metadata");
  }
  instance_.graph = ggml_new_graph_custom(
      instance_.context.get(), graph_size_, /*grads=*/false);
  if (instance_.graph == nullptr) {
    throw std::runtime_error("ggml: failed to create cgraph");
  }

  seed_values();
  index_written_values();
  for (const NodeId id : method_.graph.schedule) {
    const Node& node = method_.graph.node(id);
    if (node.is_call()) {
      copy_required_ = touches_written_value(node);
      lower(node);
    }
  }
  // In-place writes are already in the graph (expand_write); write-backs run
  // after every output.
  for (const OutputSlot& output : outputs_) {
    TensorRef value = tensor(output.id);
    value.tensor = as_output(value.tensor, output.meta);
    ggml_set_output(value.tensor);
    ggml_build_forward_expand(instance_.graph, value.tensor);
    instance_.outputs.push_back(value);
  }
  std::vector<ggml_tensor*> copies;
  for (size_t i = 0; i < method_.output_specs.size(); ++i) {
    if (method_.output_specs[i].kind == OutputKind::BufferMutation) {
      if (ggml_tensor* copy = write_back(
              method_.graph.output_ids[i], method_.output_specs[i].target_id)) {
        copies.push_back(copy);
      }
    }
  }
  for (ggml_tensor* copy : copies) {
    ggml_build_forward_expand(instance_.graph, copy);
  }
  return std::move(instance_);
}

// Outputs leave through a byte copy, so they must be contiguous and of the
// declared type. int64 is widened on the host.
ggml_tensor* GraphBuilder::as_output(
    ggml_tensor* result,
    const TensorMeta& meta) {
  const ggml_type type = compute_type(meta);
  if (result->type != type) {
    return ggml_cast(instance_.context.get(), result, type);
  }
  return ggml_is_contiguous(result)
      ? result
      : ggml_cont(instance_.context.get(), result);
}

// ggml runs nodes in the order they enter the graph. An in-place write enters
// as soon as it is lowered, after every earlier op that reads the storage it
// writes, so those reads see the old value as the schedule says.
void GraphBuilder::expand_write(ggml_tensor* write) {
  const ggml_tensor* root = storage_root(write);
  ggml_context* ctx = instance_.context.get();
  for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr;
       t = ggml_get_next_tensor(ctx, t)) {
    if (t == write || t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW ||
        t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
        t->op == GGML_OP_TRANSPOSE) {
      continue;
    }
    if (std::ranges::any_of(t->src, [root](const ggml_tensor* src) {
          return src != nullptr && storage_root(src) == root;
        })) {
      ggml_build_forward_expand(instance_.graph, t);
    }
  }
  ggml_build_forward_expand(instance_.graph, write);
}

// In-place lowerings already wrote their buffer; anything else is copied
// back after every output has read the old value. The returned copy must
// enter the graph after every write-back's value, so that no value reads a
// buffer another write-back has already overwritten.
ggml_tensor* GraphBuilder::write_back(ValueId value_id, ValueId target_id) {
  const auto target = persistent_.find(target_id);
  if (target == persistent_.end()) {
    throw std::runtime_error("ggml: mutated buffer is not a persistent tensor");
  }
  ggml_tensor* value = tensor(value_id).tensor;
  if (value == target->second || value->view_src == target->second) {
    return nullptr;
  }
  value = snapshot(value);
  ggml_build_forward_expand(instance_.graph, value);
  return ggml_cpy(instance_.context.get(), value, target->second);
}

// `tensor`, or a copy of it if it shares persistent storage, which
// write-backs and in-place writes may overwrite.
ggml_tensor* GraphBuilder::snapshot(ggml_tensor* tensor) {
  const ggml_tensor* root = storage_root(tensor);
  for (const auto& [id, persistent] : persistent_) {
    if (persistent == root) {
      return ggml_cont(instance_.context.get(), tensor);
    }
  }
  return tensor;
}

void GraphBuilder::seed_values() {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    const InputSlot& input = inputs_[i];
    const bool narrow = input.meta.dtype == ScalarType::Long;
    ggml_tensor* tensor = new_tensor(
        instance_.context.get(), compute_type(input.meta), input.sizes);
    ggml_set_name(tensor, method_.graph.value(input.id).name.c_str());
    ggml_set_input(tensor);
    set(input.id, TensorRef{tensor, input.sizes});
    instance_.uploads.push_back(Upload{
        .tensor = tensor,
        .bytes = narrow ? narrow_to_int32(input.bytes) : input.bytes,
        .input = i});
  }
  for (const auto& [id, tensor] : persistent_) {
    set(id, TensorRef{tensor, method_.graph.value(id).tensor_meta().sizes});
  }
  for (ValueId id = 0; id < static_cast<ValueId>(method_.graph.values.size());
       ++id) {
    const Value& value = method_.graph.value(id);
    if (!value.is_scalar()) {
      continue;
    }
    const Scalar& scalar = value.scalar();
    if (scalar.is_int()) {
      set(id, scalar.to_int());
    } else if (scalar.is_double()) {
      set(id, scalar.to_double());
    } else {
      set(id, scalar.to_bool());
    }
  }
}

void GraphBuilder::set(ValueId id, RuntimeValue value) {
  if (!in_bounds(id, values_.size())) {
    throw std::runtime_error("ggml: value id is out of range");
  }
  const auto* ref = std::get_if<TensorRef>(&value);
  if (ref != nullptr) {
    const auto ne = reversed_extents(ref->sizes);
    if (!std::equal(ne.begin(), ne.end(), ref->tensor->ne)) {
      throw std::logic_error(
          "ggml: lowered tensor shape disagrees with value '" +
          method_.graph.value(id).name + "'");
    }
  }
  values_.at(static_cast<size_t>(id)) = std::move(value);
}

const RuntimeValue& GraphBuilder::value(ValueId id) const {
  if (!in_bounds(id, values_.size())) {
    throw std::runtime_error("ggml: argument value id is out of range");
  }
  return values_.at(static_cast<size_t>(id));
}

// The value as lowered, which may be a strided view (see set_view).
TensorRef GraphBuilder::view(ValueId id) const {
  const TensorRef* ref = std::get_if<TensorRef>(&value(id));
  if (ref == nullptr || ref->tensor == nullptr) {
    throw std::runtime_error("ggml: argument does not resolve to a tensor");
  }
  return *ref;
}

// The value with any deferred permutation copied out, once per value.
TensorRef GraphBuilder::tensor(ValueId id) const {
  TensorRef ref = view(id);
  if (!deferred_.contains(id)) {
    return ref;
  }
  auto [it, inserted] = deferred_copies_.try_emplace(id, nullptr);
  if (inserted) {
    it->second = ggml_cont(instance_.context.get(), ref.tensor);
  }
  ref.tensor = it->second;
  return ref;
}

TensorRef GraphBuilder::tensor_arg(
    const Node& node,
    std::string_view name,
    size_t positional) const {
  return tensor(tensor_arg_id(node, name, positional));
}

// For lowerings whose ggml ops read any strides.
TensorRef GraphBuilder::view_arg(
    const Node& node,
    std::string_view name,
    size_t positional) const {
  return view(tensor_arg_id(node, name, positional));
}

// For lowerings whose ggml ops read any outer strides but need contiguous
// rows: a deferred permutation that moves the last dim is copied.
TensorRef GraphBuilder::rows_arg(
    const Node& node,
    std::string_view name,
    size_t positional) const {
  const ValueId id = tensor_arg_id(node, name, positional);
  const TensorRef ref = view(id);
  return ggml_is_contiguous_rows(ref.tensor) ? ref : tensor(id);
}

ValueId GraphBuilder::tensor_arg_id(
    const Node& node,
    std::string_view name,
    size_t positional) const {
  const Argument& arg = require_arg(node, name, positional).arg;
  if (arg.kind() != ArgKind::Tensor || !valid(arg.as_tensor().id)) {
    throw std::runtime_error(
        "ggml: argument '" + std::string(name) + "' must be a tensor");
  }
  return arg.as_tensor().id;
}

// The result of a permutation. A contiguous result goes through
// materialize(). A strided view of storage the graph never writes stays a
// view, shared like materialize()'s (so a later in-place write throws):
// generic consumers get one copy from tensor(), while view-aware lowerings
// (permute, cast, rope, index_put rows, attention, matmul's right operand)
// read the strides, so a transpose -> rope -> transpose back chain costs no
// copies at any length.
void GraphBuilder::set_view(ValueId id, TensorRef ref) {
  const ggml_tensor* root = storage_root(ref.tensor);
  if (ggml_is_contiguous(ref.tensor)) {
    ref.tensor = materialize(ref.tensor);
    set(id, std::move(ref));
  } else if (
      copy_required_ || persistent_storage_.contains(root) ||
      written_storage_.contains(root)) {
    ref.tensor = ggml_cont(instance_.context.get(), ref.tensor);
    set(id, std::move(ref));
  } else {
    aliased_storage_.insert(root);
    set(id, std::move(ref));
    deferred_.insert(id);
  }
}

void GraphBuilder::set_single_view(const Node& node, TensorRef result) {
  const std::vector<ValueId> ids = output_ids(node);
  if (ids.size() != 1) {
    throw std::runtime_error(
        "ggml: op '" + node.target + "' must produce one tensor");
  }
  set_view(ids.front(), std::move(result));
}

// The key extent a mask value is computed over, if truncated.
std::optional<int64_t> GraphBuilder::key_extent_of(const Node& node) const {
  const std::vector<ValueId> ids = output_ids(node);
  if (ids.size() != 1) {
    return std::nullopt;
  }
  const auto found = key_extents_.find(ids.front());
  return found == key_extents_.end() ? std::nullopt
                                     : std::optional(found->second);
}

// The first `extent` entries along the last axis.
TensorRef GraphBuilder::last_axis_prefix(TensorRef ref, int64_t extent) {
  if (ref.sizes.empty() || ref.sizes.back() <= extent) {
    return ref;
  }
  ggml_tensor* t = ref.tensor;
  ref.tensor = ggml_view_4d(
      instance_.context.get(),
      t,
      extent,
      t->ne[1],
      t->ne[2],
      t->ne[3],
      t->nb[1],
      t->nb[2],
      t->nb[3],
      0);
  ref.sizes.back() = extent;
  return ref;
}

int64_t GraphBuilder::int_value(const Argument& arg) const {
  if (arg.kind() != ArgKind::Int) {
    throw std::runtime_error("ggml: expected integer argument");
  }
  const IntArg& integer = arg.as_int();
  if (!valid(integer.id)) {
    return integer.value;
  }
  const RuntimeValue& resolved = value(integer.id);
  if (const auto* i = std::get_if<int64_t>(&resolved)) {
    return *i;
  }
  throw std::runtime_error("ggml: symbolic integer is unresolved");
}

int64_t GraphBuilder::int_arg(
    const Node& node,
    std::string_view name,
    size_t positional,
    int64_t fallback,
    bool optional) const {
  const NamedArgument* found = find_arg(node, name, positional);
  if (found == nullptr || found->arg.kind() == ArgKind::None) {
    if (optional) {
      return fallback;
    }
    throw std::runtime_error(
        "ggml: op '" + node.target + "' is missing integer argument '" +
        std::string(name) + "'");
  }
  return int_value(found->arg);
}

double GraphBuilder::float_arg(
    const Node& node,
    std::string_view name,
    size_t positional,
    double fallback,
    bool optional) const {
  const NamedArgument* found = find_arg(node, name, positional);
  if (found == nullptr || found->arg.kind() == ArgKind::None) {
    if (optional) {
      return fallback;
    }
    throw std::runtime_error("ggml: missing floating-point argument");
  }
  if (found->arg.kind() == ArgKind::Float) {
    const FloatArg& arg = found->arg.as_float();
    if (!valid(arg.id)) {
      return arg.value;
    }
    const RuntimeValue& resolved = value(arg.id);
    if (const auto* f = std::get_if<double>(&resolved)) {
      return *f;
    }
  }
  if (found->arg.kind() == ArgKind::Int) {
    return static_cast<double>(int_value(found->arg));
  }
  throw std::runtime_error("ggml: floating-point argument is unresolved");
}

bool GraphBuilder::bool_arg(
    const Node& node,
    std::string_view name,
    size_t positional,
    bool fallback) const {
  const NamedArgument* found = find_arg(node, name, positional);
  if (found == nullptr || found->arg.kind() == ArgKind::None) {
    return fallback;
  }
  if (found->arg.kind() != ArgKind::Bool) {
    throw std::runtime_error("ggml: expected boolean argument");
  }
  const BoolArg& arg = found->arg.as_bool();
  if (!valid(arg.id)) {
    return arg.value;
  }
  const RuntimeValue& resolved = value(arg.id);
  if (const auto* b = std::get_if<bool>(&resolved)) {
    return *b;
  }
  throw std::runtime_error("ggml: symbolic boolean is unresolved");
}

std::vector<int64_t> GraphBuilder::int_list(const Argument& arg) const {
  if (arg.kind() != ArgKind::IntList) {
    throw std::runtime_error("ggml: expected integer-list argument");
  }
  const IntListArg& list = arg.as_int_list();
  std::vector<int64_t> out = list.values;
  if (list.ids.empty()) {
    return out;
  }
  if (list.ids.size() != out.size()) {
    throw std::runtime_error("ggml: malformed symbolic integer list");
  }
  for (size_t i = 0; i < out.size(); ++i) {
    if (valid(list.ids[i])) {
      const RuntimeValue& resolved = value(list.ids[i]);
      const auto* integer = std::get_if<int64_t>(&resolved);
      if (integer == nullptr) {
        throw std::runtime_error("ggml: symbolic list element is unresolved");
      }
      out[i] = *integer;
    }
  }
  return out;
}

void GraphBuilder::set_single_output(const Node& node, TensorRef result) {
  const std::vector<ValueId> ids = output_ids(node);
  if (ids.size() != 1) {
    throw std::runtime_error(
        "ggml: op '" + node.target + "' must produce one tensor");
  }
  set(ids.front(), std::move(result));
}

void GraphBuilder::lower(const Node& node) {
  const std::string key = op_key(node.target);
  if (key == "aten.linear.default") {
    lower_linear(node);
  } else if (key == "aten.embedding.default" || key == "aten.index.Tensor") {
    lower_row_gather(node);
  } else if (key == "aten.rms_norm.default") {
    lower_rms_norm(node);
  } else if (
      key == "aten.view_copy.default" || key == "aten.reshape.default" ||
      key == "aten._unsafe_view.default" || key == "aten.view.default") {
    lower_reshape(node);
  } else if (
      key == "aten.permute_copy.default" || key == "aten.permute.default") {
    lower_permute(node);
  } else if (key == "aten.transpose_copy.int") {
    lower_transpose(node);
  } else if (
      key == "aten.clone.default" || key == "aten.contiguous.default" ||
      key == "aten.alias.default") {
    lower_copy_or_alias(node, key == "aten.alias.default");
  } else if (key == "aten._to_copy.default") {
    lower_cast(node);
  } else if (
      key == "aten.add.Tensor" || key == "aten.mul.Tensor" ||
      key == "aten.sub.Tensor" || key == "aten.div.Tensor") {
    lower_binary(node, key);
  } else if (key == "aten.silu.default") {
    lower_unary(node, ggml_silu);
  } else if (key == "aten.sigmoid.default" || key == "aten.sigmoid_.default") {
    lower_unary(node, ggml_sigmoid);
  } else if (key == "aten.gelu.default" || key == "aten.gelu_.default") {
    lower_gelu(node);
  } else if (key == "aten.native_layer_norm.default") {
    lower_layer_norm(node);
  } else if (key == "aten.constant_pad_nd.default") {
    lower_constant_pad(node);
  } else if (key == "aten.log.default") {
    lower_unary(node, ggml_log);
  } else if (key == "aten.neg.default") {
    lower_unary(node, ggml_neg);
  } else if (key == "aten.relu.default" || key == "aten.relu_.default") {
    lower_unary(node, ggml_relu);
  } else if (
      key == "aten.hardtanh.default" || key == "aten.hardtanh_.default") {
    lower_hardtanh(node);
  } else if (key == "aten.convolution.default") {
    lower_convolution(node);
  } else if (key == "aten._native_batch_norm_legit_no_training.default") {
    lower_batch_norm(node);
  } else if (
      key == "aten.max_pool2d_with_indices.default" ||
      key == "aten.max_pool2d.default") {
    lower_max_pool2d(node);
  } else if (key == "aten.mean.dim") {
    lower_mean(node);
  } else if (key == "aten.slice_copy.Tensor" || key == "aten.slice.Tensor") {
    lower_slice(node, key == "aten.slice_copy.Tensor");
  } else if (key == "aten.select_copy.int" || key == "aten.select.int") {
    lower_select(node, key == "aten.select_copy.int");
  } else if (
      key == "aten.unsqueeze_copy.default" || key == "aten.unsqueeze.default") {
    lower_unsqueeze(node);
  } else if (key == "aten.squeeze_copy.dims" || key == "aten.squeeze.dims") {
    lower_squeeze(node);
  } else if (
      key == "aten.expand_copy.default" || key == "aten.expand.default") {
    lower_expand(node);
  } else if (key == "aten.repeat.default") {
    lower_repeat(node);
  } else if (key == "aten.cat.default") {
    lower_cat(node);
  } else if (
      key == "aten.matmul.default" || key == "aten.mm.default" ||
      key == "aten.bmm.default") {
    lower_matmul(node);
  } else if (key == "llama.update_cache.default") {
    lower_update_cache(node);
  } else if (key == "llama.custom_sdpa.default") {
    lower_custom_sdpa(node);
  } else if (key == "aten.le.Tensor" || key == "aten.le.Scalar") {
    lower_compare(node, /*strict=*/false);
  } else if (key == "aten.lt.Tensor" || key == "aten.lt.Scalar") {
    lower_compare(node, /*strict=*/true);
  } else if (key == "aten.arange.start_step") {
    lower_arange(node);
  } else if (key == "torchao.choose_qparams_affine.default") {
    lower_choose_qparams(node);
  } else if (key == "torchao.quantize_affine.default") {
    lower_quantize_affine(node);
  } else if (key == "torchao.dequantize_affine.default") {
    lower_dequantize_affine(node);
  } else if (key == "native.rope.default") {
    lower_rope(node);
  } else if (key == "aten.index_put_.default") {
    lower_index_put(node);
  } else if (key == "aten.scaled_dot_product_attention.default") {
    lower_sdpa(node);
  } else if (key == "aten.sym_size.int") {
    lower_sym_size(node);
  } else if (
      key == "_operator.add" || key == "_operator.sub" ||
      key == "_operator.mul" || key == "_operator.floordiv") {
    lower_symbolic(node, key);
  } else {
    throw std::runtime_error("ggml: unsupported PTG op '" + node.target + "'");
  }
}

// Row indices are checked on the host, which needs them to come from a
// graph input through integer casts, views or repeats, or an integer arange
// whose bounds are resolved when building the graph.
void GraphBuilder::check_row_indices(ValueId id, int64_t rows) {
  static const std::unordered_set<std::string> kSameValues = {
      "aten._to_copy.default",
      "aten.alias.default",
      "aten.clone.default",
      "aten.contiguous.default",
      "aten.expand.default",
      "aten.expand_copy.default",
      "aten.permute.default",
      "aten.permute_copy.default",
      "aten.repeat.default",
      "aten.reshape.default",
      "aten._unsafe_view.default",
      "aten.select.int",
      "aten.select_copy.int",
      "aten.slice.Tensor",
      "aten.slice_copy.Tensor",
      "aten.squeeze.dims",
      "aten.squeeze_copy.dims",
      "aten.transpose_copy.int",
      "aten.unsqueeze.default",
      "aten.unsqueeze_copy.default",
      "aten.view.default",
      "aten.view_copy.default",
  };
  const auto is_index = [this](ValueId value) {
    const ScalarType dtype = method_.graph.value(value).tensor_meta().dtype;
    return dtype == ScalarType::Long || dtype == ScalarType::Int;
  };
  while (is_index(id)) {
    const auto input = std::ranges::find(inputs_, id, &InputSlot::id);
    if (input != inputs_.end()) {
      check_index_bytes(*input, rows);
      instance_.row_index_checks.push_back(
          RowIndexCheck{.input_id = id, .rows = rows});
      return;
    }
    const NodeId producer = method_.graph.value(id).producer_id;
    if (!valid(producer)) {
      break;
    }
    const Node& node = method_.graph.node(producer);
    if (node.is_call() && op_key(node.target) == "aten.arange.start_step") {
      const int64_t start = int_arg(node, "start", 0, 0);
      const int64_t step = int_arg(node, "step", 2, 1, true);
      const int64_t count = ggml_nelements(tensor(id).tensor);
      const int64_t last = start + (count - 1) * step;
      if (start < 0 || last >= rows) {
        throw std::out_of_range("ggml: arange row index is outside the table");
      }
      return;
    }
    const NamedArgument* source = find_arg(node, "self", 0);
    if (!node.is_call() || !kSameValues.contains(op_key(node.target)) ||
        source == nullptr || source->arg.kind() != ArgKind::Tensor) {
      break;
    }
    id = source->arg.as_tensor().id;
  }
  throw std::runtime_error(
      "ggml: row indices must come from a graph input, through integer "
      "casts, views or repeats, or an integer arange");
}

ggml_tensor* GraphBuilder::as_f32(ggml_tensor* tensor) {
  return tensor->type == GGML_TYPE_F32
      ? tensor
      : ggml_cast(instance_.context.get(), tensor, GGML_TYPE_F32);
}

ggml_tensor* GraphBuilder::as_i32(ggml_tensor* tensor) {
  if (tensor->type == GGML_TYPE_I32) {
    return tensor;
  }
  // Reuse conversions of an immutable arithmetic result (or one of its
  // views), e.g. an index consumed by several independent operations.
  // Never reuse a conversion of a graph input or a potentially writable
  // op; explicit in-place writes invalidate conversions of their storage.
  const ggml_tensor* root = storage_root(tensor);
  const bool reusable = tensor->type == GGML_TYPE_F32 &&
      (root->op == GGML_OP_ADD || root->op == GGML_OP_SUB ||
       root->op == GGML_OP_MUL || root->op == GGML_OP_DIV ||
       root->op == GGML_OP_ARANGE || root->op == GGML_OP_REPEAT);
  if (reusable) {
    if (const auto it = i32_casts_.find(tensor); it != i32_casts_.end()) {
      return it->second;
    }
  }
  ggml_tensor* result =
      ggml_cast(instance_.context.get(), tensor, GGML_TYPE_I32);
  if (reusable) {
    i32_casts_.emplace(tensor, result);
  }
  return result;
}

void GraphBuilder::invalidate_cached_transforms(const ggml_tensor* target) {
  const ggml_tensor* root = storage_root(target);
  for (auto it = i32_casts_.begin(); it != i32_casts_.end();) {
    if (storage_root(it->first) == root) {
      it = i32_casts_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = attention_masks_.begin(); it != attention_masks_.end();) {
    if (storage_root(tensor(it->first.first).tensor) == root) {
      it = attention_masks_.erase(it);
    } else {
      ++it;
    }
  }
}

ggml_tensor* GraphBuilder::contiguous(ggml_tensor* tensor) {
  return ggml_is_contiguous(tensor)
      ? tensor
      : ggml_cont(instance_.context.get(), tensor);
}

// The values the graph writes in place, and those that share their storage
// through view lowerings, so copies that touch them stay real copies.
void GraphBuilder::index_written_values() {
  static const std::unordered_set<std::string> kViews = {
      "aten._unsafe_view.default",
      "aten.alias.default",
      "aten.permute.default",
      "aten.permute_copy.default",
      "aten.reshape.default",
      "aten.select.int",
      "aten.slice.Tensor",
      "aten.squeeze.dims",
      "aten.squeeze_copy.dims",
      "aten.transpose_copy.int",
      "aten.unsqueeze.default",
      "aten.unsqueeze_copy.default",
      "aten.view.default",
      "aten.view_copy.default",
  };
  for (const NodeId id : method_.graph.schedule) {
    const Node& node = method_.graph.node(id);
    const std::string key = node.is_call() ? op_key(node.target) : "";
    const NamedArgument* target = key == "aten.index_put_.default"
        ? find_arg(node, "self", 0)
        : key == "llama.update_cache.default" ? find_arg(node, "cache", 1)
                                              : nullptr;
    if (target != nullptr && target->arg.kind() == ArgKind::Tensor) {
      written_values_.insert(target->arg.as_tensor().id);
    }
  }
  for (bool grew = !written_values_.empty(); grew;) {
    grew = false;
    for (const NodeId id : method_.graph.schedule) {
      const Node& node = method_.graph.node(id);
      const NamedArgument* source = find_arg(node, "self", 0);
      if (!node.is_call() || !kViews.contains(op_key(node.target)) ||
          source == nullptr || source->arg.kind() != ArgKind::Tensor) {
        continue;
      }
      std::vector<ValueId> shared = output_ids(node);
      shared.push_back(source->arg.as_tensor().id);
      if (std::ranges::any_of(shared, [this](ValueId value) {
            return written_values_.contains(value);
          })) {
        for (const ValueId shared_id : shared) {
          grew |= written_values_.insert(shared_id).second;
        }
      }
    }
  }
}

bool GraphBuilder::touches_written_value(const Node& node) const {
  if (std::ranges::any_of(node.inputs, [this](const NamedArgument& input) {
        return input.arg.kind() == ArgKind::Tensor &&
            written_values_.contains(input.arg.as_tensor().id);
      })) {
    return true;
  }
  return std::ranges::any_of(output_ids(node), [this](ValueId value) {
    return written_values_.contains(value);
  });
}

// The result of a copying op. Contiguous storage is shared through a view
// instead of copied unless it is persistent, already written, or the node
// touches a value the graph writes in place (index_written_values).
ggml_tensor* GraphBuilder::materialize(ggml_tensor* tensor) {
  const ggml_tensor* root = storage_root(tensor);
  if (copy_required_ || !ggml_is_contiguous(tensor) ||
      persistent_storage_.contains(root) || written_storage_.contains(root)) {
    return ggml_cont(instance_.context.get(), tensor);
  }
  aliased_storage_.insert(root);
  return ggml_reshape_4d(
      instance_.context.get(),
      tensor,
      tensor->ne[0],
      tensor->ne[1],
      tensor->ne[2],
      tensor->ne[3]);
}

void GraphBuilder::mark_written(const ggml_tensor* target) {
  const ggml_tensor* root = storage_root(target);
  if (aliased_storage_.contains(root)) {
    throw std::runtime_error(
        "ggml: in-place write to storage shared by a copy's view");
  }
  written_storage_.insert(root);
}

GraphBuilder::BinaryOperands GraphBuilder::tensor_operands(
    const TensorRef& left,
    const TensorRef& right,
    bool force_f32) {
  BinaryOperands out{
      .lhs = left.tensor,
      .rhs = right.tensor,
      .sizes = broadcast_shape(left.sizes, right.sizes),
  };
  // ggml has no integer arithmetic kernels; F32 holds the I32 values the
  // engine carries exactly up to 2^24.
  const auto is_integer = [](const ggml_tensor* t) {
    return t->type == GGML_TYPE_I8 || t->type == GGML_TYPE_I16 ||
        t->type == GGML_TYPE_I32 || t->type == GGML_TYPE_I64;
  };
  if (force_f32 || out.lhs->type != out.rhs->type || is_integer(out.lhs)) {
    out.lhs = as_f32(out.lhs);
    out.rhs = as_f32(out.rhs);
  }
  // ggml only broadcasts the right operand.
  if (reversed_extents(left.sizes) != reversed_extents(out.sizes)) {
    ggml_context* ctx = instance_.context.get();
    out.lhs =
        ggml_repeat(ctx, out.lhs, new_tensor(ctx, out.lhs->type, out.sizes));
  }
  return out;
}

std::array<int, 2> GraphBuilder::pair_arg(
    const Node& node,
    std::string_view name,
    size_t positional,
    std::array<int, 2> fallback,
    int64_t min_value) const {
  const NamedArgument* found = find_arg(node, name, positional);
  if (found == nullptr || found->arg.kind() == ArgKind::None) {
    return fallback;
  }
  const std::vector<int64_t> values = int_list(found->arg);
  if (values.empty()) {
    return fallback;
  }
  if (values.size() > 2) {
    throw std::runtime_error(
        "ggml: '" + std::string(name) + "' must have one or two entries");
  }
  if (std::ranges::any_of(values, [min_value](int64_t entry) {
        return entry < min_value || entry > std::numeric_limits<int>::max();
      })) {
    throw std::runtime_error(
        "ggml: '" + std::string(name) + "' entry is out of range");
  }
  // PyTorch lists are (height, width); ggml takes width first.
  return {static_cast<int>(values.back()), static_cast<int>(values.front())};
}

std::optional<TensorRef> GraphBuilder::optional_tensor(
    const Node& node,
    std::string_view name,
    size_t positional) const {
  const NamedArgument* found = find_arg(node, name, positional);
  if (found == nullptr || found->arg.kind() != ArgKind::Tensor ||
      !valid(found->arg.as_tensor().id)) {
    return std::nullopt;
  }
  return tensor(found->arg.as_tensor().id);
}

// These lowerings produce only their first output, so a consumer of any
// other output is rejected here instead of failing to resolve later.
void GraphBuilder::require_only_first_output_used(const Node& node) const {
  const std::vector<ValueId> ids = output_ids(node);
  for (size_t i = 1; i < ids.size(); ++i) {
    if (!method_.graph.value(ids[i]).consumer_ids.empty() ||
        std::ranges::find(method_.graph.output_ids, ids[i]) !=
            method_.graph.output_ids.end()) {
      throw std::runtime_error(
          "ggml: op '" + node.target + "' lowers only its first output");
    }
  }
}

void GraphBuilder::require_f32(const Node& node, const ggml_tensor* tensor) {
  if (tensor->type != GGML_TYPE_F32) {
    throw std::runtime_error(
        "ggml: op '" + node.target + "' supports F32 tensors only");
  }
}

void GraphBuilder::lower_sym_size(const Node& node) {
  const TensorRef input = tensor_arg(node, "self", 0);
  const int64_t dim =
      normalize_dim(int_arg(node, "dim", 1, 0), input.sizes.size());
  const std::vector<ValueId> ids = output_ids(node);
  if (ids.size() != 1) {
    throw std::runtime_error("ggml: sym_size must produce one value");
  }
  set(ids.front(), input.sizes[static_cast<size_t>(dim)]);
}

int64_t GraphBuilder::symbolic_operand(const Argument& arg) const {
  if (arg.kind() == ArgKind::Int) {
    return int_value(arg);
  }
  throw std::runtime_error("ggml: symbolic arithmetic expects integers");
}

void GraphBuilder::lower_symbolic(const Node& node, const std::string& key) {
  if (node.inputs.size() < 2) {
    throw std::runtime_error("ggml: symbolic arithmetic is missing operands");
  }
  const int64_t a = symbolic_operand(node.inputs[0].arg);
  const int64_t b = symbolic_operand(node.inputs[1].arg);
  int64_t result = 0;
  bool overflow = false;
  if (key == "_operator.add") {
    overflow = __builtin_add_overflow(a, b, &result);
  } else if (key == "_operator.sub") {
    overflow = __builtin_sub_overflow(a, b, &result);
  } else if (key == "_operator.mul") {
    overflow = __builtin_mul_overflow(a, b, &result);
  } else {
    if (b == 0) {
      throw std::runtime_error("ggml: symbolic floor division by zero");
    }
    overflow = a == std::numeric_limits<int64_t>::min() && b == -1;
    result = overflow ? 0 : a / b;
    if (!overflow && a % b != 0 && (a < 0) != (b < 0)) {
      --result;
    }
  }
  if (overflow) {
    throw std::runtime_error("ggml: symbolic integer arithmetic overflows");
  }
  const std::vector<ValueId> ids = output_ids(node);
  if (ids.size() != 1) {
    throw std::runtime_error("ggml: symbolic op must produce one value");
  }
  set(ids.front(), result);
}

} // namespace ptn::ggml_detail
