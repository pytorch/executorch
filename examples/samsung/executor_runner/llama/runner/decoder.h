/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder in LLM and return the output.

#pragma once

#include <executorch/examples/samsung/executor_runner/llama/runner/io_manager.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/platform/compiler.h>

namespace example {

namespace llm = ::executorch::extension::llm;

class ET_EXPERIMENTAL Decoder
    : public executorch::extension::llm::TextDecoderRunner {
 public:
  struct SharedMethodMemory {
    std::vector<std::vector<uint8_t>> buffers;
    std::vector<executorch::runtime::Span<uint8_t>> spans;
    std::unique_ptr<executorch::runtime::HierarchicalAllocator> allocator;
  };

  explicit Decoder(
      executorch::extension::Module* module,
      llm::IOManager* io_manager)
      : TextDecoderRunner(module, io_manager), shared_method_memory_(nullptr) {}

  virtual ::executorch::runtime::Result<executorch::aten::Tensor> step(
      executorch::extension::TensorPtr& input,
      int64_t start_pos) override;

  inline ::executorch::runtime::Error load() override {
    executorch::runtime::Error ret = executorch::runtime::Error::Ok;

    if (methods_.num > 1) {
      ret = create_shared_planned_memory();
      if (ret != executorch::runtime::Error::Ok) {
        ET_LOG(Error, "Failed to create share planned memory");
      }
    }

    /**
     * Dealing with multiple method.
     */
    for (auto method : methods_.name) {
      if (module_->is_method_loaded(method)) {
        ET_LOG(Info, "Method \"%s\" is already loaded", method.c_str());
        continue;
      }

      if (shared_method_memory_) {
        ret = module_->load_method(
            method, shared_method_memory_->allocator.get());
        if (ret != executorch::runtime::Error::Ok) {
          ET_LOG(Error, "Failed to load method");
          return ret;
        }
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

    ET_LOG(Info, "method[%d] : %s", methods_.num - 1, method.c_str());

    for (int i = 0; i < methods_.num; i++) {
      ET_LOG(
          Info, "Registered method[%d] : \"%s\"", i, methods_.name[i].c_str());
    }
  }

 private:
  inline ::executorch::runtime::Error load_io_manager() {
    executorch::runtime::Error ret = executorch::runtime::Error::Ok;

    if (methods_.num > 1) {
      ret = io_manager_->load(
          methods_.name[static_cast<std::size_t>(DecoderMode::PREFILL)],
          methods_.name[static_cast<std::size_t>(DecoderMode::DECODE)]);
    } else {
      ret = io_manager_->load(
          methods_.name[static_cast<std::size_t>(DecoderMode::DECODE)],
          methods_.name[static_cast<std::size_t>(DecoderMode::DECODE)]);
    }
    return ret;
  }

  ::executorch::runtime::Error create_shared_planned_memory();

  enum class DecoderMode : int { PREFILL = 1, DECODE = 0 };

  struct RegisterdMethod {
    uint32_t num = 0;
    std::vector<std::string> name = {default_method_name, default_method_name};
    std::vector<executorch::ET_RUNTIME_NAMESPACE::Method*> objs = {
        nullptr,
        nullptr};
  };

 private:
  RegisterdMethod methods_;

  std::unique_ptr<SharedMethodMemory> shared_method_memory_;
};

} // namespace example
