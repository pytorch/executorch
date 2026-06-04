/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Example-local serving module for Qwen3.5 MoE. It does NOT define any
// Qwen-specific Python class: it constructs a Qwen35MoEEngine and hands it to
// the generic PyLLMEngine wrapper (llm_pybind_wrappers.h), so the Python
// surface is the same generic LLMEngine / LLMSession the text model exposes.
// The generic extension/llm/runner pybind is untouched.

#include <pybind11/pybind11.h>

#include <executorch/examples/models/qwen3_5_moe/qwen35_moe_engine.h>
#include <executorch/extension/llm/runner/llm_pybind_wrappers.h>
#include <executorch/runtime/platform/runtime.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace executorch::extension::llm;

PYBIND11_MODULE(_qwen35_moe, m) {
  m.doc() =
      "Example-local Qwen3.5 MoE serving module: create_engine() builds a "
      "Qwen35MoEEngine and returns the generic LLMEngine / LLMSession surface.";

  ::executorch::runtime::runtime_init();

  // Same generic Engine/Session surface as the _llm_runner module.
  pybind_wrappers::bind_engine_session_api(m);

  m.def(
      "create_engine",
      [](const std::string& model_path,
         const std::string& tokenizer_path,
         std::optional<std::string> data_path,
         bool cuda_graph) {
        Qwen35MoEConfig config;
        config.model_path = model_path;
        config.tokenizer_path = tokenizer_path;
        config.data_path = data_path.value_or("");
        config.cuda_graph = cuda_graph;
        auto res = Qwen35MoEEngine::create(config);
        if (!res.ok()) {
          throw std::runtime_error("Failed to create Qwen35MoEEngine");
        }
        return std::make_unique<pybind_wrappers::PyLLMEngine>(
            std::unique_ptr<LLMEngine>(std::move(res.get())));
      },
      py::arg("model_path"),
      py::arg("tokenizer_path"),
      py::arg("data_path") = py::none(),
      py::arg("cuda_graph") = false,
      "Load the Qwen3.5 MoE program once and return an LLMEngine.");
}
