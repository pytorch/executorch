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

#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
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
  PyMultimodalRunner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      float temperature = 0.8f) {
    // Load tokenizer
    tokenizer_ = get_tokenizer(tokenizer_path.c_str());
    if (!tokenizer_) {
      throw std::runtime_error("Failed to load tokenizer from: " + tokenizer_path);
    }

    // Load module
    module_ = std::make_unique<Module>(model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
    Error error = module_->load_method("forward");
    THROW_IF_ERROR(error, "Failed to load model from: %s", model_path.c_str());

    // Get model type from metadata
    const auto method_names = module_->method_names();
    ET_CHECK_MSG(!method_names.empty(), "No methods found in model");

    // Get metadata
    auto method_meta = module_->method_meta("forward");
    if (method_meta.ok()) {
      for (const auto& [key, value] : method_meta.get()) {
        metadata_[key] = std::stoi(value);
      }
    }

    // Set up sampler
    int32_t vocab_size = get_vocab_size();
    sampler_ = std::make_unique<Sampler>(
        vocab_size,
        temperature,
        0.9f,  // top_p
        0LL    // seed
    );

    // Create components
    stats_ = std::make_unique<Stats>(metadata_);
    
    // Create text decoder runner
    text_decoder_runner_ = std::make_unique<MultimodalDecoderRunner>(
        module_.get(),
        metadata_
    );

    // Create multimodal prefiller
    multimodal_prefiller_ = std::make_unique<MultimodalPrefiller>(
        module_.get(),
        metadata_
    );

    // Create IO manager
    io_manager_ = std::make_unique<IOManager>(
        module_.get(),
        tokenizer_.get(),
        text_decoder_runner_.get(),
        multimodal_prefiller_.get(),
        sampler_.get(),
        stats_.get(),
        metadata_
    );

    // Create text token generator  
    text_token_generator_ = std::make_unique<TextTokenGenerator>(
        tokenizer_.get(),
        sampler_.get(),
        text_decoder_runner_.get(),
        false,  // echo
        stats_.get(),
        false   // warming
    );

    // Finally create the runner
    runner_ = std::make_unique<MultimodalRunner>(
        metadata_,
        std::move(tokenizer_),
        std::move(module_),
        std::move(text_decoder_runner_),
        std::move(multimodal_prefiller_),
        std::move(io_manager_),
        std::move(text_token_generator_),
        std::move(stats_)
    );
  }

  void generate(
      const std::vector<MultimodalInput>& inputs,
      const GenerationConfig& config,
      py::object token_callback = py::none(),
      py::object stats_callback = py::none()) {
    
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
    std::string result;
    
    std::function<void(const std::string&)> token_callback = 
        [&result](const std::string& token) {
          result += token;
        };
    
    std::function<void(const Stats&)> stats_callback = nullptr;
    
    {
      py::gil_scoped_release release;
      Error error = runner_->generate(
          inputs, config, token_callback, stats_callback);
      THROW_IF_ERROR(error, "Generation failed");
    }
    
    return result;
  }

  void stop() {
    runner_->stop();
  }

  int32_t get_vocab_size() const {
    auto it = metadata_.find("vocab_size");
    if (it != metadata_.end()) {
      return static_cast<int32_t>(it->second);
    }
    // Default vocab size if not in metadata
    return tokenizer_->vocab_size();
  }

 private:
  std::unique_ptr<MultimodalRunner> runner_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<Module> module_;
  std::unique_ptr<MultimodalDecoderRunner> text_decoder_runner_;
  std::unique_ptr<MultimodalPrefiller> multimodal_prefiller_;
  std::unique_ptr<IOManager> io_manager_;
  std::unique_ptr<TextTokenGenerator> text_token_generator_;
  std::unique_ptr<Stats> stats_;
  std::unique_ptr<Sampler> sampler_;
  std::unordered_map<std::string, int64_t> metadata_;
};

// Helper functions for creating MultimodalInput
MultimodalInput make_text_input(const std::string& text) {
  return MultimodalInput::text(text);
}

MultimodalInput make_image_input(py::array_t<uint8_t> image_array) {
  // Get image dimensions
  py::buffer_info buf = image_array.request();
  
  if (buf.ndim != 3) {
    throw std::runtime_error("Image array must be 3-dimensional (H, W, C)");
  }
  
  size_t height = buf.shape[0];
  size_t width = buf.shape[1];
  size_t channels = buf.shape[2];
  
  if (channels != 3 && channels != 4) {
    throw std::runtime_error("Image must have 3 (RGB) or 4 (RGBA) channels");
  }
  
  // Create Image object from numpy array
  uint8_t* data = static_cast<uint8_t*>(buf.ptr);
  std::vector<uint8_t> image_data(data, data + height * width * channels);
  
  Image image(std::move(image_data), height, width, channels);
  return MultimodalInput::image(std::move(image));
}

PYBIND11_MODULE(_llm_runner, m) {
  m.doc() = "Python bindings for ExecuTorch LLM Runners";

  // Initialize ExecuTorch runtime
  runtime_init();

  // Bind GenerationConfig
  py::class_<GenerationConfig>(m, "GenerationConfig")
      .def(py::init<>())
      .def_readwrite("max_new_tokens", &GenerationConfig::max_new_tokens)
      .def_readwrite("temperature", &GenerationConfig::temperature)
      .def_readwrite("top_p", &GenerationConfig::top_p)
      .def_readwrite("top_k", &GenerationConfig::top_k)
      .def_readwrite("repetition_penalty", &GenerationConfig::repetition_penalty)
      .def_readwrite("presence_penalty", &GenerationConfig::presence_penalty)
      .def_readwrite("frequency_penalty", &GenerationConfig::frequency_penalty)
      .def_readwrite("warming", &GenerationConfig::warming)
      .def_readwrite("echo", &GenerationConfig::echo)
      .def_readwrite("seed", &GenerationConfig::seed)
      .def("__repr__", [](const GenerationConfig& config) {
        return "<GenerationConfig max_new_tokens=" + 
               std::to_string(config.max_new_tokens) + 
               " temperature=" + std::to_string(config.temperature) + 
               " top_p=" + std::to_string(config.top_p) + ">";
      });

  // Bind Stats
  py::class_<Stats>(m, "Stats")
      .def_readonly("model_load_start_ms", &Stats::model_load_start_ms)
      .def_readonly("model_load_end_ms", &Stats::model_load_end_ms)
      .def_readonly("inference_start_ms", &Stats::inference_start_ms)
      .def_readonly("inference_end_ms", &Stats::inference_end_ms)
      .def_readonly("prompt_eval_start_ms", &Stats::prompt_eval_start_ms)
      .def_readonly("prompt_eval_end_ms", &Stats::prompt_eval_end_ms)
      .def_readonly("first_token_ms", &Stats::first_token_ms)
      .def_readonly("aggregate_sampling_time_ms", &Stats::aggregate_sampling_time_ms)
      .def_readonly("num_prompt_tokens", &Stats::num_prompt_tokens)
      .def_readonly("num_generated_tokens", &Stats::num_generated_tokens)
      .def("get_model_load_time_ms", &Stats::get_model_load_time_ms)
      .def("get_inference_time_ms", &Stats::get_inference_time_ms)
      .def("get_prompt_eval_time_ms", &Stats::get_prompt_eval_time_ms)
      .def("get_eval_time_ms", &Stats::get_eval_time_ms)
      .def("get_sampling_time_ms", &Stats::get_sampling_time_ms)
      .def("get_tokens_per_second", &Stats::get_tokens_per_second)
      .def("__repr__", [](const Stats& stats) {
        return "<Stats tokens_per_second=" + 
               std::to_string(stats.get_tokens_per_second()) + 
               " num_generated=" + std::to_string(stats.num_generated_tokens) + ">";
      });

  // Bind Image class
  py::class_<Image>(m, "Image")
      .def(py::init<std::vector<uint8_t>, size_t, size_t, size_t>(),
           py::arg("data"), py::arg("height"), py::arg("width"), py::arg("channels"))
      .def_property_readonly("height", [](const Image& img) { return img.height_; })
      .def_property_readonly("width", [](const Image& img) { return img.width_; })
      .def_property_readonly("channels", [](const Image& img) { return img.channels_; })
      .def("__repr__", [](const Image& img) {
        return "<Image height=" + std::to_string(img.height_) + 
               " width=" + std::to_string(img.width_) + 
               " channels=" + std::to_string(img.channels_) + ">";
      });

  // Bind MultimodalInput
  py::class_<MultimodalInput>(m, "MultimodalInput")
      .def_static("text", &MultimodalInput::text, 
                  "Create a text input", py::arg("text"))
      .def_static("image", &MultimodalInput::image,
                  "Create an image input", py::arg("image"))
      .def("is_text", &MultimodalInput::is_text)
      .def("is_image", &MultimodalInput::is_image)
      .def("get_text", [](const MultimodalInput& input) -> py::object {
        if (input.is_text()) {
          return py::cast(input.get_text());
        }
        return py::none();
      })
      .def("__repr__", [](const MultimodalInput& input) {
        if (input.is_text()) {
          return "<MultimodalInput type=text content=\"" + 
                 input.get_text().substr(0, 50) + 
                 (input.get_text().length() > 50 ? "..." : "") + "\">";
        } else if (input.is_image()) {
          return "<MultimodalInput type=image>";
        }
        return "<MultimodalInput type=unknown>";
      });

  // Bind helper functions
  m.def("make_text_input", &make_text_input, 
        "Create a text input for multimodal processing",
        py::arg("text"));
  
  m.def("make_image_input", &make_image_input,
        "Create an image input from a numpy array (H, W, C)",
        py::arg("image_array"));

  // Bind PyMultimodalRunner
  py::class_<PyMultimodalRunner>(m, "MultimodalRunner")
      .def(py::init<const std::string&, const std::string&, float>(),
           py::arg("model_path"),
           py::arg("tokenizer_path"),
           py::arg("temperature") = 0.8f,
           "Initialize a MultimodalRunner with model and tokenizer paths")
      .def("generate", &PyMultimodalRunner::generate,
           py::arg("inputs"),
           py::arg("config"),
           py::arg("token_callback") = py::none(),
           py::arg("stats_callback") = py::none(),
           "Generate text from multimodal inputs with optional callbacks")
      .def("generate_text", &PyMultimodalRunner::generate_text,
           py::arg("inputs"),
           py::arg("config"),
           "Generate text and return the complete result as a string")
      .def("stop", &PyMultimodalRunner::stop,
           "Stop the current generation")
      .def("get_vocab_size", &PyMultimodalRunner::get_vocab_size,
           "Get the vocabulary size of the model")
      .def("__repr__", [](const PyMultimodalRunner& runner) {
        return "<MultimodalRunner vocab_size=" + 
               std::to_string(runner.get_vocab_size()) + ">";
      });
}