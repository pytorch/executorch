/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/decoder_llm_runner.h>
#include <executorch/runtime/core/error.h>

// Include the Llama runner implementation
#include <examples/models/llama/runner/runner.h>

namespace py = pybind11;
using namespace executorch::extension::llm;
using namespace executorch::runtime;

namespace {

// Helper function to convert Error to a Python exception
void handle_error(const Error& error) {
  if (!error.ok()) {
    throw std::runtime_error(error.what());
  }
}

// Wrapper class for the IRunner interface to handle Python callbacks properly
class PyRunner : public IRunner {
 public:
  // Use trampoline pattern for virtual methods
  using IRunner::IRunner;

  bool is_loaded() const override {
    PYBIND11_OVERRIDE_PURE(bool, IRunner, is_loaded);
  }

  Error load() override {
    PYBIND11_OVERRIDE_PURE(Error, IRunner, load);
  }

  Error generate(
      const std::string& prompt,
      const GenerationConfig& config,
      std::function<void(const std::string&)> token_callback,
      std::function<void(const Stats&)> stats_callback) override {
    PYBIND11_OVERRIDE_PURE(
        Error,
        IRunner,
        generate,
        prompt,
        config,
        token_callback,
        stats_callback);
  }

  void stop() override {
    PYBIND11_OVERRIDE_PURE(void, IRunner, stop);
  }
};

} // namespace

PYBIND11_MODULE(llm_runner, m) {
  m.doc() = "Python bindings for ExecuTorch LLM Runner API";

  // Bind Stats class
  py::class_<Stats>(m, "Stats")
      .def(py::init<>())
      .def_readwrite("prefill_time_ms", &Stats::prefill_time_ms)
      .def_readwrite("decode_time_ms", &Stats::decode_time_ms)
      .def_readwrite("prompt_tokens", &Stats::prompt_tokens)
      .def_readwrite("generation_tokens", &Stats::generation_tokens)
      .def("__repr__", [](const Stats& self) {
        std::stringstream ss;
        ss << "Stats(prefill_time_ms=" << self.prefill_time_ms
           << ", decode_time_ms=" << self.decode_time_ms
           << ", prompt_tokens=" << self.prompt_tokens
           << ", generation_tokens=" << self.generation_tokens << ")";
        return ss.str();
      });

  // Bind RunnerConfig struct
  py::class_<RunnerConfig>(m, "RunnerConfig")
      .def(py::init<>())
      .def_readwrite("max_seq_len", &RunnerConfig::max_seq_len)
      .def_readwrite("max_context_len", &RunnerConfig::max_context_len)
      .def_readwrite("use_kv_cache", &RunnerConfig::use_kv_cache)
      .def_readwrite("enable_dynamic_shape", &RunnerConfig::enable_dynamic_shape)
      .def_readwrite("use_sdpa_with_kv_cache", &RunnerConfig::use_sdpa_with_kv_cache)
      .def("__repr__", [](const RunnerConfig& self) {
        std::stringstream ss;
        ss << "RunnerConfig(max_seq_len=" << self.max_seq_len
           << ", max_context_len=" << self.max_context_len
           << ", use_kv_cache=" << (self.use_kv_cache ? "True" : "False")
           << ", enable_dynamic_shape=" << (self.enable_dynamic_shape ? "True" : "False")
           << ", use_sdpa_with_kv_cache=" << (self.use_sdpa_with_kv_cache ? "True" : "False")
           << ")";
        return ss.str();
      });

  // Bind GenerationConfig struct
  py::class_<GenerationConfig>(m, "GenerationConfig")
      .def(py::init<>())
      .def_readwrite("temperature", &GenerationConfig::temperature)
      .def_readwrite("echo", &GenerationConfig::echo)
      .def_readwrite("max_new_tokens", &GenerationConfig::max_new_tokens)
      .def_readwrite("warming", &GenerationConfig::warming)
      .def("__repr__", [](const GenerationConfig& self) {
        std::stringstream ss;
        ss << "GenerationConfig(temperature=" << self.temperature
           << ", echo=" << (self.echo ? "True" : "False")
           << ", max_new_tokens=" << self.max_new_tokens
           << ", warming=" << (self.warming ? "True" : "False")
           << ")";
        return ss.str();
      });

  // Bind IRunner interface
  py::class_<IRunner, PyRunner>(m, "IRunner")
      .def(py::init<>())
      .def("is_loaded", &IRunner::is_loaded)
      .def("load", [](IRunner& self) {
        Error err = self.load();
        handle_error(err);
      })
      .def(
          "generate",
          [](IRunner& self,
             const std::string& prompt,
             const GenerationConfig& config,
             std::optional<std::function<void(const std::string&)>> token_callback,
             std::optional<std::function<void(const Stats&)>> stats_callback) {
            Error err = self.generate(
                prompt,
                config,
                token_callback.value_or([](const std::string&) {}),
                stats_callback.value_or([](const Stats&) {}));
            handle_error(err);
          },
          py::arg("prompt"),
          py::arg("config") = GenerationConfig(),
          py::arg("token_callback") = std::nullopt,
          py::arg("stats_callback") = std::nullopt)
      .def("stop", &IRunner::stop);

  // Bind DecoderLLMRunner class
  py::class_<DecoderLLMRunner, IRunner>(m, "DecoderLLMRunner")
      .def(py::init<std::unique_ptr<executorch::extension::Module>,
                    std::unique_ptr<::tokenizers::Tokenizer>,
                    std::unique_ptr<std::unordered_set<uint64_t>>,
                    RunnerConfig>(),
           py::arg("module"),
           py::arg("tokenizer"),
           py::arg("eos_ids") = nullptr,
           py::arg("runner_config") = RunnerConfig())
      .def_property_readonly("runner_config", &DecoderLLMRunner::get_runner_config)
      .def("is_loaded", &DecoderLLMRunner::is_loaded)
      .def("load", [](DecoderLLMRunner& self) {
        Error err = self.load();
        handle_error(err);
      })
      .def(
          "generate",
          [](DecoderLLMRunner& self,
             const std::string& prompt,
             const GenerationConfig& config,
             std::optional<std::function<void(const std::string&)>> token_callback,
             std::optional<std::function<void(const Stats&)>> stats_callback) {
            Error err = self.generate(
                prompt,
                config,
                token_callback.value_or([](const std::string&) {}),
                stats_callback.value_or([](const Stats&) {}));
            handle_error(err);
          },
          py::arg("prompt"),
          py::arg("config") = GenerationConfig(),
          py::arg("token_callback") = std::nullopt,
          py::arg("stats_callback") = std::nullopt)
      .def("stop", &DecoderLLMRunner::stop)
      .def("warmup", [](DecoderLLMRunner& self, const std::string& prompt, int32_t max_new_tokens) {
        Error err = self.warmup(prompt, max_new_tokens);
        handle_error(err);
      },
      py::arg("prompt"),
      py::arg("max_new_tokens") = 20);

  // Bind LLaMA Runner implementation
  py::class_<example::Runner, IRunner>(m, "LlamaRunner")
      .def(py::init<const std::string&, const std::string&, float, std::optional<const std::string>>(),
           py::arg("model_path"),
           py::arg("tokenizer_path"),
           py::arg("temperature") = 0.8f,
           py::arg("data_path") = std::nullopt)
      .def("is_loaded", &example::Runner::is_loaded)
      .def("load", [](example::Runner& self) {
        Error err = self.load();
        handle_error(err);
      })
      .def(
          "generate",
          [](example::Runner& self,
             const std::string& prompt,
             int32_t max_new_tokens,
             std::optional<std::function<void(const std::string&)>> token_callback,
             std::optional<std::function<void(const Stats&)>> stats_callback,
             bool echo) {
            // Convert to new API by creating a GenerationConfig
            GenerationConfig config;
            config.max_new_tokens = max_new_tokens; 
            config.echo = echo;
            config.warming = false;
            
            Error err = self.generate(
                prompt,
                config,
                token_callback.value_or([](const std::string&) {}),
                stats_callback.value_or([](const Stats&) {}));
            handle_error(err);
          },
          py::arg("prompt"),
          py::arg("max_new_tokens") = 128,
          py::arg("token_callback") = std::nullopt,
          py::arg("stats_callback") = std::nullopt,
          py::arg("echo") = true)
      .def("stop", &example::Runner::stop)
      .def("warmup", [](example::Runner& self, const std::string& prompt, int32_t max_new_tokens) {
        Error err = self.warmup(prompt, max_new_tokens);
        handle_error(err);
      },
      py::arg("prompt"),
      py::arg("max_new_tokens") = 128);
}