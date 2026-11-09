// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vkcompute::shader_profiling {

using WorkCountFunction = double (*)(const Json&);

class ThroughputRegistry final {
 private:
  std::unordered_map<std::string, WorkCountFunction> operator_memory_;
  std::unordered_map<std::string, WorkCountFunction> operator_compute_;
  std::unordered_map<std::string, WorkCountFunction> shader_memory_;
  std::unordered_map<std::string, WorkCountFunction> shader_compute_;
  std::unordered_map<std::string, WorkCountFunction> shader_memory_prefixes_;
  std::unordered_map<std::string, WorkCountFunction> shader_compute_prefixes_;

 public:
  WorkCountFunction operator_memory(const std::string& name) const;
  WorkCountFunction operator_compute(const std::string& name) const;
  WorkCountFunction shader_memory(const std::string& name) const;
  WorkCountFunction shader_compute(const std::string& name) const;
  void register_operator_memory(
      const std::string& name,
      WorkCountFunction function);
  void register_operator_compute(
      const std::string& name,
      WorkCountFunction function);
  void register_shader_memory(
      const std::string& name,
      WorkCountFunction function);
  void register_shader_compute(
      const std::string& name,
      WorkCountFunction function);
  void register_shader_memory_prefix(
      const std::string& prefix,
      WorkCountFunction function);
  void register_shader_compute_prefix(
      const std::string& prefix,
      WorkCountFunction function);
};

class ThroughputRegisterInit final {
 public:
  explicit ThroughputRegisterInit(void (*initialize)()) {
    initialize();
  }
};

struct WorkAmounts {
  std::optional<double> memory_bytes;
  std::optional<double> compute_flops;
  std::vector<std::string> diagnostics;
};

ThroughputRegistry& throughput_registry();
WorkAmounts estimate_work(const Json& resolved_case, bool memory, bool compute);

namespace throughput {

double tensor_elements(const Json& tensor);
double tensor_bytes(const Json& tensor);
double tensor_traffic(const Json& spec);
const Json& tensor_argument(const Json& spec, size_t index);
const Json& output_tensor(const Json& spec);

} // namespace throughput

} // namespace vkcompute::shader_profiling

#define REGISTER_OPERATOR_MEMORY_TRAFFIC(name, function) \
  ::vkcompute::shader_profiling::throughput_registry()   \
      .register_operator_memory(#name, &function)
#define REGISTER_OPERATOR_COMPUTE_FLOPS(name, function) \
  ::vkcompute::shader_profiling::throughput_registry()  \
      .register_operator_compute(#name, &function)
#define REGISTER_SHADER_MEMORY(name, function)                                 \
  ::vkcompute::shader_profiling::throughput_registry().register_shader_memory( \
      #name, &function)
#define REGISTER_SHADER_COMPUTE(name, function)        \
  ::vkcompute::shader_profiling::throughput_registry() \
      .register_shader_compute(#name, &function)
#define REGISTER_SHADER_MEMORY_TRAFFIC(prefix, function) \
  ::vkcompute::shader_profiling::throughput_registry()   \
      .register_shader_memory_prefix(#prefix, &function)
#define REGISTER_SHADER_COMPUTE_FLOPS(prefix, function) \
  ::vkcompute::shader_profiling::throughput_registry()  \
      .register_shader_compute_prefix(#prefix, &function)

#define REGISTER_THROUGHPUT_FUNCTIONS                                \
  static void register_throughput_functions();                       \
  static const ::vkcompute::shader_profiling::ThroughputRegisterInit \
      throughput_reg(&register_throughput_functions);                \
  static void register_throughput_functions()
