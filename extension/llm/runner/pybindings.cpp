/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace executorch::extension::llm;
using namespace executorch::extension;
using namespace executorch::runtime;

// Helper macro for error handling
#define THROW_IF_ERROR(error, message, ...)                       \
  ({                                                              \
    if ((error) != Error::Ok) {                                   \
      char msg_buf[256];                                          \
      snprintf(msg_buf, sizeof(msg_buf), message, ##__VA_ARGS__); \
      throw std::runtime_error(msg_buf);                          \
    }                                                             \
  })

// Python wrapper class for MultimodalRunner
class PyMultimodalRunner {
 public:
  // Constructor that takes a tokenizer path
  PyMultimodalRunner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      std::optional<const std::string> data_path = std::nullopt) {
    // Load tokenizer using the helper function
    auto tokenizer =
        load_tokenizer(tokenizer_path, nullptr, std::nullopt, 0, 0);
    if (!tokenizer) {
      throw std::runtime_error(
          "Failed to load tokenizer from: " + tokenizer_path);
    }

    // Create multimodal runner using the helper function
    runner_ =
        create_multimodal_runner(model_path, std::move(tokenizer), data_path);
    if (!runner_) {
      throw std::runtime_error(
          "Failed to create multimodal runner with model: " + model_path);
    }
  }

  void generate(
      const std::vector<MultimodalInput>& inputs,
      const GenerationConfig& config,
      py::object token_callback = py::none(),
      py::object stats_callback = py::none()) {
    if (!runner_) {
      throw std::runtime_error("Runner not initialized");
    }

    // Convert Python callbacks to C++ std::function
    std::function<void(const std::string&)> cpp_token_callback = nullptr;
    if (!token_callback.is_none()) {
      cpp_token_callback = [token_callback](const std::string& token) {
        py::gil_scoped_acquire acquire;
        token_callback(token);
      };
    }

    std::function<void(const Stats&)> cpp_stats_callback = nullptr;
    if (!stats_callback.is_none()) {
      cpp_stats_callback = [stats_callback](const Stats& stats) {
        py::gil_scoped_acquire acquire;
        stats_callback(stats);
      };
    }

    // Release GIL during generation
    {
      py::gil_scoped_release release;
      Error error = runner_->generate(
          inputs, config, cpp_token_callback, cpp_stats_callback);
      THROW_IF_ERROR(error, "Generation failed");
    }
  }

  std::string generate_text(
      const std::vector<MultimodalInput>& inputs,
      const GenerationConfig& config) {
    if (!runner_) {
      throw std::runtime_error("Runner not initialized");
    }

    std::string generated_text;
    auto cpp_token_callback = [&generated_text](const std::string& token) {
      generated_text += token;
    };
    Error error =
        runner_->generate(inputs, config, cpp_token_callback, nullptr);
    THROW_IF_ERROR(error, "Generation failed");

    return generated_text;
  }

  void stop() {
    if (runner_) {
      runner_->stop();
    }
  }

  void reset() {
    if (runner_) {
      runner_->reset();
    }
  }

  // Note: Since the runner owns the tokenizer and metadata after creation,
  // we cannot directly access them. This is a limitation of the current design.
  // For now, we'll return a placeholder value.
  int32_t get_vocab_size() const {
    // TODO: Consider exposing metadata through the MultimodalRunner interface
    return -1; // Indicate that vocab size is not available
  }

 private:
  std::unique_ptr<MultimodalRunner> runner_;
};

PYBIND11_MODULE(_llm_runner, m) {
  m.doc() = "Python bindings for ExecuTorch LLM Runners";

  // Initialize ExecuTorch runtime
  runtime_init();

  // Bind GenerationConfig
  py::class_<GenerationConfig>(m, "GenerationConfig")
      .def(py::init<>())
      .def_readwrite("echo", &GenerationConfig::echo)
      .def_readwrite("max_new_tokens", &GenerationConfig::max_new_tokens)
      .def_readwrite("warming", &GenerationConfig::warming)
      .def_readwrite("seq_len", &GenerationConfig::seq_len)
      .def_readwrite("temperature", &GenerationConfig::temperature)
      .def_readwrite("num_bos", &GenerationConfig::num_bos)
      .def_readwrite("num_eos", &GenerationConfig::num_eos)
      .def(
          "resolve_max_new_tokens",
          &GenerationConfig::resolve_max_new_tokens,
          py::arg("max_context_len"),
          py::arg("num_prompt_tokens"),
          "Resolve the maximum number of new tokens to generate based on constraints")
      .def("__repr__", [](const GenerationConfig& config) {
        return "<GenerationConfig max_new_tokens=" +
            std::to_string(config.max_new_tokens) +
            " seq_len=" + std::to_string(config.seq_len) +
            " temperature=" + std::to_string(config.temperature) +
            " echo=" + (config.echo ? "True" : "False") +
            " warming=" + (config.warming ? "True" : "False") + ">";
      });

  // Bind Stats
  py::class_<Stats>(m, "Stats")
      .def_readonly(
          "SCALING_FACTOR_UNITS_PER_SECOND",
          &Stats::SCALING_FACTOR_UNITS_PER_SECOND)
      .def_readonly("model_load_start_ms", &Stats::model_load_start_ms)
      .def_readonly("model_load_end_ms", &Stats::model_load_end_ms)
      .def_readonly("inference_start_ms", &Stats::inference_start_ms)
      .def_readonly("token_encode_end_ms", &Stats::token_encode_end_ms)
      .def_readonly(
          "model_execution_start_ms", &Stats::model_execution_start_ms)
      .def_readonly("model_execution_end_ms", &Stats::model_execution_end_ms)
      .def_readonly("prompt_eval_end_ms", &Stats::prompt_eval_end_ms)
      .def_readonly("first_token_ms", &Stats::first_token_ms)
      .def_readonly("inference_end_ms", &Stats::inference_end_ms)
      .def_readonly(
          "aggregate_sampling_time_ms", &Stats::aggregate_sampling_time_ms)
      .def_readonly("num_prompt_tokens", &Stats::num_prompt_tokens)
      .def_readonly("num_generated_tokens", &Stats::num_generated_tokens)
      .def("on_sampling_begin", &Stats::on_sampling_begin)
      .def("on_sampling_end", &Stats::on_sampling_end)
      .def(
          "reset",
          &Stats::reset,
          py::arg("all_stats") = false,
          "Reset stats, optionally including model load times")
      .def(
          "to_json_string",
          [](const Stats& stats) { return stats_to_json_string(stats); },
          "Convert stats to JSON string representation")
      .def("__repr__", [](const Stats& stats) {
        double tokens_per_second = 0.0;
        if (stats.inference_end_ms > stats.inference_start_ms) {
          tokens_per_second = static_cast<double>(stats.num_generated_tokens) *
              stats.SCALING_FACTOR_UNITS_PER_SECOND /
              (stats.inference_end_ms - stats.inference_start_ms);
        }
        return "<Stats num_prompt_tokens=" +
            std::to_string(stats.num_prompt_tokens) + " num_generated_tokens=" +
            std::to_string(stats.num_generated_tokens) +
            " tokens_per_second=" + std::to_string(tokens_per_second) + ">";
      });

  // Bind Image class
  py::class_<Image>(m, "Image")
      .def(py::init<>())
      .def_readwrite("data", &Image::data)
      .def_readwrite("width", &Image::width)
      .def_readwrite("height", &Image::height)
      .def_readwrite("channels", &Image::channels)
      .def("__repr__", [](const Image& img) {
        return "<Image height=" + std::to_string(img.height) +
            " width=" + std::to_string(img.width) +
            " channels=" + std::to_string(img.channels) + ">";
      });

  // Bind MultimodalInput
  py::class_<MultimodalInput>(m, "MultimodalInput")
      .def(
          py::init<const std::string&>(),
          py::arg("text"),
          "Create a MultimodalInput with text")
      .def(
          py::init<const Image&>(),
          py::arg("image"),
          "Create a MultimodalInput with an image")
      .def("is_text", &MultimodalInput::is_text)
      .def("is_image", &MultimodalInput::is_image)
      .def(
          "get_text",
          [](const MultimodalInput& input) -> py::object {
            if (input.is_text()) {
              return py::cast(input.get_text());
            }
            return py::none();
          })
      .def("__repr__", [](const MultimodalInput& input) -> std::string {
        if (input.is_text()) {
          return "<MultimodalInput type=text content=\"" +
              input.get_text().substr(0, 50) +
              (input.get_text().length() > 50 ? "..." : "") + "\">";
        } else if (input.is_image()) {
          return "<MultimodalInput type=image>";
        }
        return "<MultimodalInput type=unknown>";
      });

  // Bind helper functions using lambdas
  m.def(
      "make_text_input",
      [](const std::string& text) -> MultimodalInput {
        return MultimodalInput(text);
      },
      "Create a text input for multimodal processing",
      py::arg("text"));

  m.def(
      "make_image_input",
      [](torch::Tensor image_tensor) -> MultimodalInput {
        if (image_tensor.dim() == 4) {
          if (image_tensor.size(0) != 1) {
            throw std::runtime_error(
                "Batch size for 4D image tensor must be 1");
          }
          image_tensor = image_tensor.squeeze(0);
        }

        
        if (image_tensor.dim() != 3) {
          throw std::runtime_error(
              "Image tensor must be 3-dimensional (H, W, C) or 4-dimensional (1, H, W, C)");
        }

        int64_t height, width, channels;
        // Check for memory format and permute to CHW if necessary
        if (image_tensor.is_contiguous(at::MemoryFormat::ChannelsLast)) {
          // Input is HWC, permute to CHW
          height = image_tensor.size(0);
          width = image_tensor.size(1);
          channels = image_tensor.size(2);
          image_tensor = image_tensor.permute({2, 0, 1});
        } else if (image_tensor.is_contiguous(at::MemoryFormat::Contiguous)) {
          // Input is CHW
          channels = image_tensor.size(0);
          height = image_tensor.size(1);
          width = image_tensor.size(2);
        } else {
          throw std::runtime_error(
              "Image tensor must be contiguous in either channels last (H, W, C) or contiguous (C, H, W) format.");
        }

        if (channels != 3 && channels != 4) {
          throw std::runtime_error(
              "Image must have 3 (RGB) or 4 (RGBA) channels");
        }

        if (image_tensor.scalar_type() != torch::kUInt8) {
          if (image_tensor.max().item<double>() <= 1.0) {
            image_tensor = (image_tensor * 255).to(torch::kUInt8);
          } else {
            image_tensor = image_tensor.to(torch::kUInt8);
          }
        }

        image_tensor = image_tensor.contiguous();
        uint8_t* data = image_tensor.data_ptr<uint8_t>();
        std::vector<uint8_t> image_data(data, data + image_tensor.numel());

        Image image;
        image.data = std::move(image_data);
        image.width = static_cast<int32_t>(width);
        image.height = static_cast<int32_t>(height);
        image.channels = static_cast<int32_t>(channels);
        return MultimodalInput(std::move(image));
      },
      "Create an image input from a torch tensor (H, W, C), (1, H, W, C), (C, H, W), or (1, C, H, W)",
      py::arg("image_tensor"));

  // Bind PyMultimodalRunner
  py::class_<PyMultimodalRunner>(m, "MultimodalRunner")
      // Constructor with tokenizer path
      .def(
          py::init<
              const std::string&,
              const std::string&,
              std::optional<const std::string>>(),
          py::arg("model_path"),
          py::arg("tokenizer_path"),
          py::arg("data_path") = py::none(),
          "Initialize a MultimodalRunner with model and tokenizer paths")
      .def(
          "generate",
          &PyMultimodalRunner::generate,
          py::arg("inputs"),
          py::arg("config"),
          py::arg("token_callback") = py::none(),
          py::arg("stats_callback") = py::none(),
          "Generate text from multimodal inputs with optional callbacks")
      .def("stop", &PyMultimodalRunner::stop, "Stop the current generation")
      .def(
          "generate_text",
          &PyMultimodalRunner::generate_text,
          py::arg("inputs"),
          py::arg("config"),
          "Generate text from multimodal inputs and return the complete "
          "result")
      .def(
          "reset",
          &PyMultimodalRunner::reset,
          "Reset the runner state and KV cache")
      .def(
          "get_vocab_size",
          &PyMultimodalRunner::get_vocab_size,
          "Get the vocabulary size of the model")
      .def("__repr__", [](const PyMultimodalRunner& runner) {
        return "<MultimodalRunner>";
      });
}