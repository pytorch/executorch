// cppcheck-suppress-file useStlAlgorithm

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/Validation.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/CheckedMath.h>
#include <executorch/backends/native/runtime/deserialize/DeserializeError.h>
#include <executorch/backends/native/runtime/deserialize/Limits.h>
#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

namespace ptn {
namespace {

void validate_tensor_meta(const TensorMeta& meta, const std::string& name) {
  try {
    static_cast<void>(element_size(meta.dtype));
  } catch (const std::runtime_error&) {
    throw std::runtime_error(
        "native program: tensor '" + name + "' has an invalid dtype");
  }
  if (!meta.dim_order_hint.empty()) {
    if (meta.dim_order_hint.size() != meta.sizes.size()) {
      throw std::runtime_error(
          "native program: tensor '" + name +
          "' has an invalid dimension order");
    }
    std::vector<bool> seen(meta.sizes.size());
    for (const int32_t dim : meta.dim_order_hint) {
      if (dim < 0 || static_cast<size_t>(dim) >= seen.size() || seen[dim]) {
        throw std::runtime_error(
            "native program: tensor '" + name +
            "' has an invalid dimension order");
      }
      seen[dim] = true;
    }
  }
  if (meta.quant.has_value() && meta.dtype != ScalarType::Byte) {
    throw std::runtime_error(
        "native program: quantized tensor '" + name +
        "' must use Byte storage");
  }
}

const Value& bound_value(const Method& method, const DataBinding& binding) {
  if (!valid(binding.value_id) ||
      binding.value_id >= method.graph.values.size()) {
    throw std::runtime_error(
        "native program: method '" + method.name +
        "' has a binding with an invalid value id");
  }
  const Value& value =
      method.graph.values.at(static_cast<size_t>(binding.value_id));
  if (!value.is_tensor()) {
    throw std::runtime_error(
        "native program: method '" + method.name + "' binding '" + binding.key +
        "' does not name a tensor");
  }
  return value;
}

size_t tensor_nbytes(const TensorMeta& meta, const std::string& name) {
  size_t numel = 1;
  for (const int64_t dim : meta.sizes) {
    if (dim < 0 || static_cast<uint64_t>(dim) > SIZE_MAX ||
        !detail::checked_mul(numel, static_cast<size_t>(dim), numel)) {
      throw std::runtime_error(
          "native program: tensor '" + name + "' has an invalid shape");
    }
  }
  size_t nbytes = 0;
  if (!detail::checked_mul(numel, element_size(meta.dtype), nbytes)) {
    throw std::runtime_error(
        "native program: tensor '" + name + "' byte size overflows");
  }
  return nbytes;
}

size_t tensor_numel(const TensorMeta& meta, const std::string& name) {
  return tensor_nbytes(meta, name) / element_size(meta.dtype);
}

size_t affine_group_nbytes(
    const TensorMeta& meta,
    const AffineGroupQuant& quant,
    const std::string& name) {
  const int64_t levels =
      static_cast<int64_t>(quant.quant_max) - quant.quant_min + 1;
  if (levels <= 1 || levels > 256 || (levels & (levels - 1)) != 0) {
    throw std::runtime_error(
        "native program: tensor '" + name +
        "' has a non-power-of-two affine quantization range");
  }
  size_t bits = 0;
  for (int64_t value = levels; value > 1; value >>= 1) {
    ++bits;
  }
  size_t bit_count = 0;
  if (!detail::checked_mul(tensor_numel(meta, name), bits, bit_count)) {
    throw std::runtime_error(
        "native program: tensor '" + name + "' packed byte size overflows");
  }
  return bit_count / 8 + static_cast<size_t>(bit_count % 8 != 0);
}

void validate_quant_parameter(
    const Package& package,
    const std::string& key,
    ScalarType dtype,
    size_t expected_numel,
    const std::string& tensor_key) {
  const std::optional<ConstantInfo> info = package.constant_info(key);
  size_t expected_nbytes = 0;
  if (!detail::checked_mul(
          expected_numel, element_size(dtype), expected_nbytes) ||
      !info || info->dtype != dtype || info->nbytes != expected_nbytes) {
    throw std::runtime_error(
        "native package: quantization parameter '" + key + "' for '" +
        tensor_key + "' has incompatible metadata");
  }
}

void validate_quantized_constant(
    const TensorMeta& meta,
    const ConstantInfo& constant,
    const Package& package,
    const std::string& key) {
  if (constant.dtype != ScalarType::Byte) {
    throw std::runtime_error(
        "native package: packed constant '" + key + "' is not Byte");
  }
  if (const auto* affine = std::get_if<AffineGroupQuant>(&*meta.quant)) {
    // group_size 0 is one group over the whole last axis.
    const int64_t group_size = affine->group_size == 0 && !meta.sizes.empty()
        ? meta.sizes.back()
        : affine->group_size;
    if (meta.sizes.empty() || affine->scale_data_key.empty() ||
        group_size <= 0 || meta.sizes.back() % group_size != 0 ||
        constant.nbytes != affine_group_nbytes(meta, *affine, key)) {
      throw std::runtime_error(
          "native package: affine-packed constant '" + key +
          "' has incompatible metadata");
    }
    const size_t groups =
        tensor_numel(meta, key) / static_cast<size_t>(group_size);
    validate_quant_parameter(
        package, affine->scale_data_key, affine->scale_dtype, groups, key);
    if (!affine->zero_point_data_key.empty()) {
      validate_quant_parameter(
          package,
          affine->zero_point_data_key,
          affine->zero_point_dtype,
          groups,
          key);
    }
    return;
  }
  const auto* packed = std::get_if<PackedQuant>(&*meta.quant);
  if (packed == nullptr || packed->codec.empty()) {
    throw std::runtime_error(
        "native package: packed constant '" + key + "' has an invalid codec");
  }
}

ValueId storage_root(const Graph& graph, ValueId id) {
  while (valid(graph.value(id).alias_id)) {
    id = graph.value(id).alias_id;
  }
  return id;
}

// Schema rules validate_graph does not cover: IntList refs are all-or-parallel,
// an output aliases an input of its own node, and read-only tensors are never
// written in place.
void validate_node_arguments(const Graph& graph) {
  for (NodeId node_id : graph.schedule) {
    const Node& node = graph.node(node_id);
    for (const NamedArgument& input : node.inputs) {
      if (input.arg.kind() == ArgKind::IntList) {
        const IntListArg& list = input.arg.as_int_list();
        if (!list.ids.empty() && list.ids.size() != list.values.size()) {
          throw std::runtime_error(
              "native program: int list '" + input.name +
              "' has refs that do not match its values");
        }
      }
      if (input.mutated && input.arg.kind() == ArgKind::Tensor &&
          valid(input.arg.as_tensor().id)) {
        const ValueRole role =
            graph.value(storage_root(graph, input.arg.as_tensor().id)).role;
        if (role == ValueRole::Parameter || role == ValueRole::ConstantTensor) {
          throw std::runtime_error(
              "native program: argument '" + input.name +
              "' writes a read-only tensor in place");
        }
      }
    }
    const std::vector<ValueId> inputs = node.input_value_ids();
    for (const Output& output : node.outputs) {
      if (!valid(output.value_id)) {
        continue;
      }
      const ValueId alias_id = graph.value(output.value_id).alias_id;
      if (valid(alias_id) &&
          std::ranges::find(inputs, alias_id) == inputs.end()) {
        throw std::runtime_error(
            "native program: output '" + graph.value(output.value_id).name +
            "' aliases a value that is not an input of its node");
      }
    }
  }
  for (const Graph& subgraph : graph.subgraphs) {
    validate_node_arguments(subgraph);
  }
}

void validate_method_structure(const Method& method) {
  if (method.name.empty()) {
    throw std::runtime_error("native program: method name is empty");
  }
  validate_graph(method.graph);
  validate_node_arguments(method.graph);
  if (method.output_specs.size() != method.graph.output_ids.size()) {
    throw std::runtime_error(
        "native program: method '" + method.name +
        "' output specification count disagrees with graph outputs");
  }
  for (const Value& value : method.graph.values) {
    if (value.is_tensor()) {
      validate_tensor_meta(value.tensor_meta(), value.name);
    }
  }
  const auto id_is_out_of_bounds = [&method](const ValueId id) {
    return !in_bounds(id, method.graph.values.size());
  };
  if (std::any_of(
          method.graph.input_ids.begin(),
          method.graph.input_ids.end(),
          id_is_out_of_bounds)) {
    throw std::runtime_error(
        "native program: method '" + method.name + "' has an invalid input id");
  }
  if (std::any_of(
          method.graph.output_ids.begin(),
          method.graph.output_ids.end(),
          id_is_out_of_bounds)) {
    throw std::runtime_error(
        "native program: method '" + method.name +
        "' has an invalid output id");
  }
  std::unordered_set<std::string> keys;
  for (const DataBinding& binding : method.data_bindings) {
    if (binding.key.empty()) {
      throw std::runtime_error(
          "native program: method '" + method.name +
          "' has a binding with an empty FQN");
    }
    if (!keys.insert(binding.key).second) {
      throw std::runtime_error(
          "native program: method '" + method.name + "' binds '" + binding.key +
          "' more than once");
    }
    static_cast<void>(bound_value(method, binding));
    if (!binding.has_data &&
        (binding.role != ValueRole::Buffer || !binding.mutated)) {
      throw std::runtime_error(
          "native program: zero-initialized state must be a mutable buffer");
    }
    if (binding.mutated && binding.role != ValueRole::Buffer) {
      throw std::runtime_error(
          "native program: only buffers may hold persistent mutations");
    }
  }
  for (size_t i = 0; i < method.output_specs.size(); ++i) {
    const OutputSpec& spec = method.output_specs[i];
    if (spec.kind == OutputKind::UserOutput) {
      if (valid(spec.target_id)) {
        throw std::runtime_error(
            "native program: a user output has a mutation target");
      }
      continue;
    }
    if (!valid(spec.target_id) ||
        spec.target_id >= method.graph.values.size()) {
      throw std::runtime_error(
          "native program: a mutation output has an invalid target");
    }
    const ValueRole role =
        method.graph.values.at(static_cast<size_t>(spec.target_id)).role;
    if ((spec.kind == OutputKind::BufferMutation &&
         role != ValueRole::Buffer) ||
        (spec.kind == OutputKind::UserInputMutation &&
         role != ValueRole::UserInput)) {
      throw std::runtime_error(
          "native program: mutation output target has the wrong role");
    }
  }
}

struct StateRecord {
  ValueRole role;
  bool has_data;
  ScalarType dtype;
  std::vector<int64_t> sizes;
  std::vector<int64_t> lower_bounds;
  std::vector<int32_t> dim_order;
  std::optional<QuantScheme> quant;

  bool operator==(const StateRecord&) const = default;
};

} // namespace

// cppcheck-suppress unusedFunction
void validate_method_constants(const Method& method, const Package& package) {
  validate_method_structure(method);
  for (const DataBinding& binding : method.data_bindings) {
    if (!binding.has_data) {
      continue;
    }
    const Value& value = bound_value(method, binding);
    const TensorMeta& meta = value.tensor_meta();
    if (!meta.is_contiguous()) {
      throw std::runtime_error(
          "native program: serialized tensor '" + binding.key +
          "' is not contiguous");
    }
    const std::optional<ConstantInfo> constant =
        package.constant_info(binding.key);
    if (!constant) {
      throw std::runtime_error(
          "native package: missing constant '" + binding.key + "'");
    }
    if (meta.quant.has_value()) {
      validate_quantized_constant(meta, *constant, package, binding.key);
    } else if (
        constant->dtype != meta.dtype || *constant->sizes != meta.sizes ||
        constant->nbytes != tensor_nbytes(meta, binding.key)) {
      throw std::runtime_error(
          "native package: constant '" + binding.key +
          "' disagrees with its PTG binding");
    }
  }
}

// cppcheck-suppress unusedFunction
void validate_program_state(const Program& program) {
  if (program.num_methods() > detail::kMaxProgramMethods) {
    throw ResourceLimitError("native program: method count exceeds limit");
  }
  std::unordered_map<std::string, StateRecord> state;
  for (const std::string& method_name : program.method_names()) {
    const Method& method = program.get_method(method_name);
    validate_method_structure(method);
    for (const DataBinding& binding : method.data_bindings) {
      const TensorMeta& meta = bound_value(method, binding).tensor_meta();
      const StateRecord record{
          binding.role,
          binding.has_data,
          meta.dtype,
          meta.sizes,
          meta.lower_bounds,
          meta.is_contiguous() ? std::vector<int32_t>{} : meta.dim_order_hint,
          meta.quant};
      const auto [it, inserted] = state.emplace(binding.key, record);
      if (!inserted && it->second != record) {
        throw std::runtime_error(
            "native program: state '" + binding.key +
            "' has inconsistent definitions across methods");
      }
    }
  }
}

} // namespace ptn
