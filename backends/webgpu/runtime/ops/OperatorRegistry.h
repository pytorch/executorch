/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace webgpu {

class WebGPUGraph;

using OpFunction = std::function<void(WebGPUGraph&, const std::vector<int>&)>;

class OperatorRegistry final {
  using OpTable = std::unordered_map<std::string, OpFunction>;
  OpTable table_;

 public:
  bool has_op(const std::string& name);
  OpFunction& get_op_fn(const std::string& name);
  void register_op(const std::string& name, const OpFunction& fn);
};

class OperatorRegisterInit final {
  using InitFn = void();

 public:
  explicit OperatorRegisterInit(InitFn* init_fn) {
    init_fn();
  }
};

OperatorRegistry& webgpu_operator_registry();

#define WEBGPU_REGISTER_OP(name, function)                                \
  ::executorch::backends::webgpu::webgpu_operator_registry().register_op( \
      #name,                                                              \
      std::bind(&function, std::placeholders::_1, std::placeholders::_2))

#define WEBGPU_REGISTER_OPERATORS                                   \
  static void register_webgpu_ops();                                \
  static const ::executorch::backends::webgpu::OperatorRegisterInit \
      webgpu_reg(&register_webgpu_ops);                             \
  static void register_webgpu_ops()

} // namespace webgpu
} // namespace backends
} // namespace executorch
