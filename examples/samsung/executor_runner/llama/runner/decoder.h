/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder in LLM and return the output.

#pragma once

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/compiler.h>

#include <executorch/examples/samsung/executor_runner/llama/runner/io_manager.h>

namespace example {

namespace llm = ::executorch::extension::llm;

static constexpr auto default_method_name = "forward";

class ET_EXPERIMENTAL Decoder
    : public executorch::extension::llm::TextDecoderRunner {
 public:
  explicit Decoder(
      executorch::extension::Module* module,
      llm::IOManager* io_manager)
      : TextDecoderRunner(module, io_manager) {}

  virtual ::executorch::runtime::Result<executorch::aten::Tensor> step(
      executorch::extension::TensorPtr& input,
      int64_t start_pos) override;

  inline ::executorch::runtime::Error load() override {
    executorch::runtime::Error ret = executorch::runtime::Error::Ok;

    /**
     * Dealing with multiple method.
     */
    for (auto method : methods_.name) {
      if (module_->is_method_loaded(method)) {
        ET_LOG(Info, "Method \"%s\" is already loaded", method.c_str());
        continue;
      }

      ret = module_->load_method(method);
      if (ret != executorch::runtime::Error::Ok) {
        ET_LOG(Error, "Failed to load method");
        return ret;
      }

      ET_LOG(Info, "Loading Method (%s)", method.c_str());
    }

    ret = load_io_manager();
    if (ret != executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to load io manager (%d)", ret);
      return ret;
    }

    return ret;
  }

  inline bool is_method_loaded() override {
    bool ret = false;
    for (auto method : methods_.name) {
      ret = module_->is_method_loaded(method);
      if (ret == false) {
        ET_LOG(Info, "method name \"%s\" is not loaded", method.c_str());
        return ret;
      }
    }

    return ret;
  }

  inline void register_method(std::string& method) {
    methods_.name[methods_.num++] = method;
    for (auto method : methods_.name) {
      ET_LOG(Info, "Registered method : \"%s\"", method.c_str());
    }
  }

 private:
  inline ::executorch::runtime::Error load_io_manager() {
    executorch::runtime::Error ret = executorch::runtime::Error::Ok;

    return io_manager_->load(
        methods_.name[static_cast<std::size_t>(DecoderMode::PREFILL)],
        methods_.name[static_cast<std::size_t>(DecoderMode::DECODE)]);
  }

  enum class DecoderMode : int { PREFILL = 0, DECODE = 1 };

  struct RegisterdMethod {
    uint32_t num = 0;
    std::vector<std::string> name = {default_method_name, default_method_name};
    std::vector<executorch::ET_RUNTIME_NAMESPACE::Method*> objs = {
        nullptr,
        nullptr};
  };

 private:
  RegisterdMethod methods_;
};

} // namespace example
