/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder in LLM and return the output.

#pragma once

#include <unordered_set>

#include <executorch/examples/samsung/executor_runner/llm/runner/io_manager.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/platform/compiler.h>

namespace example {

namespace llm = ::executorch::extension::llm;
namespace runtime = ::executorch::runtime;
namespace aten = ::executorch::aten;
namespace extension = ::executorch::extension;

class ET_EXPERIMENTAL Decoder : public llm::TextDecoderRunner {
 public:
  struct SharedMethodMemory {
    std::vector<std::vector<uint8_t>> buffers;
    std::vector<runtime::Span<uint8_t>> spans;
    std::unique_ptr<runtime::HierarchicalAllocator> allocator;
  };

  explicit Decoder(extension::Module* module, llm::IOManager* io_manager)
      : TextDecoderRunner(module, io_manager), shared_method_memory_(nullptr) {}

  runtime::Result<aten::Tensor> step(
      extension::TensorPtr& input,
      int64_t start_pos) override;

  runtime::Error load() override;

  inline void set_method(std::string& method) {
    method_names_.insert(method);
  }

 private:
  std::string prefill_method_;
  std::string decode_method_;
  std::unordered_set<std::string> method_names_;
  std::unique_ptr<SharedMethodMemory> shared_method_memory_;

  void extract_llm_methods();
  runtime::Error load_io_manager();
  runtime::Error create_shared_planned_memory();
};

} // namespace example
