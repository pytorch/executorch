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
#include <vector>

#include <executorch/backends/native/runtime/deserialize/CheckedMath.h>
#include <executorch/backends/native/runtime/deserialize/DeserializeError.h>
#include <executorch/backends/native/runtime/deserialize/Limits.h>
#include <executorch/backends/native/runtime/graph/Ids.h>

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

void validate_method_structure(const Method& method) {
  if (method.name.empty()) {
    throw std::runtime_error("native program: method name is empty");
  }
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
  for (const DataBinding& binding : method.data_bindings) {
    if (binding.key.empty()) {
      throw std::runtime_error(
          "native program: method '" + method.name +
          "' has a binding with an empty FQN");
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
  std::vector<int32_t> dim_order;

  bool operator==(const StateRecord&) const = default;
};

} // namespace

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
    if (constant->dtype != meta.dtype || *constant->sizes != meta.sizes ||
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
          meta.is_contiguous() ? std::vector<int32_t>{} : meta.dim_order_hint};
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
