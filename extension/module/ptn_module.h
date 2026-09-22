/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <variant>
#include <vector>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>

namespace executorch::extension::native_module::internal {

using ET_RUNTIME_NAMESPACE::MethodMeta;
using ET_RUNTIME_NAMESPACE::Program;

class PtnModule {
 public:
  virtual ~PtnModule();

  virtual runtime::Result<size_t> num_methods() const = 0;
  virtual runtime::Result<std::unordered_set<std::string>> method_names()
      const = 0;
  virtual runtime::Result<MethodMeta> method_meta(
      const std::string& method_name) = 0;
  virtual runtime::Error load_method(const std::string& method_name) = 0;
  virtual bool unload_method(const std::string& method_name) = 0;
  virtual bool is_method_loaded(const std::string& method_name) const = 0;
  virtual runtime::Result<std::vector<runtime::EValue>> execute(
      const std::string& method_name,
      const std::vector<runtime::EValue>& inputs) = 0;
  virtual runtime::Error set_input(
      const std::string& method_name,
      const runtime::EValue& input,
      size_t index) = 0;
  virtual runtime::Error set_inputs(
      const std::string& method_name,
      const std::vector<runtime::EValue>& inputs) = 0;
  virtual runtime::Error set_output(
      const std::string& method_name,
      runtime::EValue output,
      size_t index) = 0;
  virtual runtime::Error set_outputs(
      const std::string& method_name,
      const std::vector<runtime::EValue>& outputs) = 0;
  virtual runtime::Result<std::vector<runtime::EValue>> get_outputs(
      const std::string& method_name) = 0;
  virtual runtime::Result<runtime::EValue> get_output(
      const std::string& method_name,
      size_t index) = 0;
};

struct PtnFileSource {
  enum class Mode : uint8_t {
    Read,
    Mmap,
  };

  std::string_view path;
  Mode mode;
};

using PtnSource =
    std::variant<std::reference_wrapper<runtime::DataLoader>, PtnFileSource>;

struct PtnHooks {
  runtime::Result<std::unique_ptr<PtnModule>> (*load)(
      const PtnSource& source,
      Program::Verification verification) = nullptr;
};

// This registration keeps PTN linkage optional: Module depends only on this
// hook contract, while a linked PTN provider supplies the implementation.
runtime::Error register_ptn_hooks(const PtnHooks& hooks);
const PtnHooks* get_ptn_hooks();

} // namespace executorch::extension::native_module::internal

namespace executorch::extension::native_module {

runtime::Result<std::unique_ptr<internal::PtnModule>> load_ptn(
    const internal::PtnSource& source,
    internal::Program::Verification verification);

} // namespace executorch::extension::native_module
