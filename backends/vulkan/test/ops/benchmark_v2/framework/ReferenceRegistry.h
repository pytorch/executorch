// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Reference.h>

#include <unordered_map>

namespace vkcompute::shader_profiling {

using ReferenceFunction =
    reference::Tensor (*)(const Json&, const reference::TensorMap&, size_t);

using ReferenceOutputsFunction =
    ReferenceOutputs (*)(const Json&, const reference::TensorMap&, size_t);

class ReferenceRegistry final {
 private:
  std::unordered_map<std::string, ReferenceFunction> functions_;
  std::unordered_map<std::string, ReferenceOutputsFunction> output_functions_;
  std::unordered_map<std::string, std::string> operators_;
  std::unordered_map<std::string, std::string> shaders_;

 public:
  ReferenceFunction get_reference(const std::string& name) const;
  ReferenceOutputsFunction get_reference_outputs(const std::string& name) const;
  bool has_reference(const std::string& name) const;
  const std::string& operator_reference(const std::string& name) const;
  const std::string& shader_reference(const std::string& name) const;
  void register_reference(const std::string& name, ReferenceFunction function);
  void register_reference_outputs(
      const std::string& name,
      ReferenceOutputsFunction function);
  void register_operator_reference(
      const std::string& name,
      const std::string& reference);
  void register_shader_reference(
      const std::string& name,
      const std::string& reference);
};

class ReferenceRegisterInit final {
 public:
  explicit ReferenceRegisterInit(void (*initialize)()) {
    initialize();
  }
};

ReferenceRegistry& reference_registry();

} // namespace vkcompute::shader_profiling

#define REGISTER_REFERENCE_FN(name, function)                             \
  ::vkcompute::shader_profiling::reference_registry().register_reference( \
      #name, &function)
#define REGISTER_OPERATOR_REFERENCE(name, reference)  \
  ::vkcompute::shader_profiling::reference_registry() \
      .register_operator_reference(#name, #reference)
#define REGISTER_SHADER_REFERENCE(name, reference)    \
  ::vkcompute::shader_profiling::reference_registry() \
      .register_shader_reference(#name, #reference)
#define REGISTER_REFERENCE_OUTPUTS_FN(name, function) \
  ::vkcompute::shader_profiling::reference_registry() \
      .register_reference_outputs(#name, &function)
#define REGISTER_REFERENCE_FUNCTIONS                                \
  static void register_references();                                \
  static const ::vkcompute::shader_profiling::ReferenceRegisterInit \
      reference_reg(&register_references);                          \
  static void register_references()
