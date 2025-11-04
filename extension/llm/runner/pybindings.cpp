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

#include <executorch/extension/llm/runner/audio.h>
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

  void prefill(std::vector<MultimodalInput> inputs) {
    if (!runner_) {
      throw std::runtime_error("Runner not initialized");
    }
    {
      py::gil_scoped_release release;
      Error error = runner_->prefill(inputs);
      THROW_IF_ERROR(error, "Prefill failed");
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
      // Constructor with keyword arguments for all fields (all optional via
      // defaults)
      .def(
          py::init([](bool echo,
                      int32_t max_new_tokens,
                      bool warming,
                      int32_t seq_len,
                      float temperature,
                      int32_t num_bos,
                      int32_t num_eos) {
            GenerationConfig cfg;
            cfg.echo = echo;
            cfg.max_new_tokens = max_new_tokens;
            cfg.warming = warming;
            cfg.seq_len = seq_len;
            cfg.temperature = temperature;
            cfg.num_bos = num_bos;
            cfg.num_eos = num_eos;
            return cfg;
          }),
          py::arg("echo") = true,
          py::arg("max_new_tokens") = -1,
          py::arg("warming") = false,
          py::arg("seq_len") = -1,
          py::arg("temperature") = 0.8f,
          py::arg("num_bos") = 0,
          py::arg("num_eos") = 0)
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
      .def(
          py::init<std::vector<uint8_t>&&, int32_t, int32_t, int32_t>(),
          py::arg("data"),
          py::arg("width"),
          py::arg("height"),
          py::arg("channels"))
      .def(
          py::init<std::vector<float>&&, int32_t, int32_t, int32_t>(),
          py::arg("data"),
          py::arg("width"),
          py::arg("height"),
          py::arg("channels"))
      .def("is_uint8", &Image::is_uint8)
      .def("is_float", &Image::is_float)
      .def_property_readonly("width", &Image::width)
      .def_property_readonly("height", &Image::height)
      .def_property_readonly("channels", &Image::channels)
      .def_property_readonly(
          "uint8_data",
          static_cast<const std::vector<uint8_t>& (Image::*)() const&>(
              &Image::get_uint8_data))
      .def_property_readonly(
          "float_data",
          static_cast<const std::vector<float>& (Image::*)() const&>(
              &Image::get_float_data))
      .def("__repr__", [](const Image& img) {
        std::string dtype = "unknown";
        if (img.is_uint8()) {
          dtype = "uint8";
        } else if (img.is_float()) {
          dtype = "float32";
        }
        return "<Image height=" + std::to_string(img.height()) +
            " width=" + std::to_string(img.width()) +
            " channels=" + std::to_string(img.channels()) + " dtype=" + dtype +
            ">";
      });

  // Bind Audio class
  py::class_<Audio>(m, "Audio")
      .def(py::init<>())
      .def(
          py::init<std::vector<uint8_t>&&, int32_t, int32_t, int32_t>(),
          py::arg("data"),
          py::arg("batch_size"),
          py::arg("n_bins"),
          py::arg("n_frames"),
          "Create preprocessed audio data (uint8)")
      .def(
          py::init<std::vector<float>&&, int32_t, int32_t, int32_t>(),
          py::arg("data"),
          py::arg("batch_size"),
          py::arg("n_bins"),
          py::arg("n_frames"),
          "Create preprocessed audio data (float32)")
      .def("is_uint8", &Audio::is_uint8)
      .def("is_float", &Audio::is_float)
      .def_property_readonly(
          "uint8_data",
          static_cast<const std::vector<uint8_t>& (Audio::*)() const&>(
              &Audio::get_uint8_data))
      .def_property_readonly(
          "float_data",
          static_cast<const std::vector<float>& (Audio::*)() const&>(
              &Audio::get_float_data))
      .def_property_readonly("batch_size", &Audio::get_batch_size)
      .def_property_readonly("n_bins", &Audio::get_n_bins)
      .def_property_readonly("n_frames", &Audio::get_n_frames)
      .def("toTensor", &Audio::toTensor)
      .def("__repr__", [](const Audio& audio) {
        std::string dtype = "unknown";
        if (audio.is_uint8()) {
          dtype = "uint8";
        } else if (audio.is_float()) {
          dtype = "float32";
        }
        return "<Audio batch_size=" + std::to_string(audio.get_batch_size()) +
            " n_bins=" + std::to_string(audio.get_n_bins()) +
            " n_frames=" + std::to_string(audio.get_n_frames()) +
            " dtype=" + dtype + ">";
      });

  // Bind RawAudio class
  py::class_<RawAudio>(m, "RawAudio")
      .def(py::init<>())
      .def(
          py::init<std::vector<uint8_t>&&, int32_t, int32_t, int32_t>(),
          py::arg("data"),
          py::arg("batch_size"),
          py::arg("n_channels"),
          py::arg("n_samples"),
          "Create raw audio data")
      .def_readwrite("data", &RawAudio::data)
      .def_readwrite("batch_size", &RawAudio::batch_size)
      .def_readwrite("n_channels", &RawAudio::n_channels)
      .def_readwrite("n_samples", &RawAudio::n_samples)
      .def("__repr__", [](const RawAudio& audio) {
        return "<RawAudio batch_size=" + std::to_string(audio.batch_size) +
            " n_channels=" + std::to_string(audio.n_channels) +
            " n_samples=" + std::to_string(audio.n_samples) + ">";
      });

  // Bind MultimodalInput
  py::class_<MultimodalInput>(m, "MultimodalInput")
      .def(
          py::init<const std::string&>(),
          py::arg("text"),
          "Create a MultimodalInput with text")
      .def(
          py::init<const std::vector<uint64_t>&>(),
          py::arg("tokens"),
          "Create a MultimodalInput with pre-tokenized tokens (List[int])")
      .def(
          py::init<const std::vector<uint64_t>&>(),
          py::arg("tokens"),
          "Create a MultimodalInput with pre-tokenized tokens (List[int])")
      .def(
          py::init<const Image&>(),
          py::arg("image"),
          "Create a MultimodalInput with an image")
      .def(
          py::init<const Audio&>(),
          py::arg("audio"),
          "Create a MultimodalInput with preprocessed audio")
      .def(
          py::init<const RawAudio&>(),
          py::arg("raw_audio"),
          "Create a MultimodalInput with raw audio")
      .def("is_text", &MultimodalInput::is_text)
      .def("is_tokens", &MultimodalInput::is_tokens)
      .def("is_image", &MultimodalInput::is_image)
      .def("is_audio", &MultimodalInput::is_audio)
      .def("is_raw_audio", &MultimodalInput::is_raw_audio)
      .def(
          "get_text",
          [](const MultimodalInput& input) -> py::object {
            if (input.is_text()) {
              return py::cast(input.get_text());
            }
            return py::none();
          })
      .def(
          "get_tokens",
          [](const MultimodalInput& input) -> py::object {
            if (input.is_tokens()) {
              return py::cast(input.get_tokens());
            }
            return py::none();
          })
      .def(
          "get_image",
          [](const MultimodalInput& input) -> py::object {
            if (input.is_image()) {
              return py::cast(input.get_image());
            }
            return py::none();
          })
      .def(
          "get_audio",
          [](const MultimodalInput& input) -> py::object {
            if (input.is_audio()) {
              return py::cast(input.get_audio());
            }
            return py::none();
          })
      .def(
          "get_raw_audio",
          [](const MultimodalInput& input) -> py::object {
            if (input.is_raw_audio()) {
              return py::cast(input.get_raw_audio());
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
        } else if (input.is_tokens()) {
          return "<MultimodalInput type=tokens>";
        } else if (input.is_audio()) {
          return "<MultimodalInput type=audio>";
        } else if (input.is_raw_audio()) {
          return "<MultimodalInput type=raw_audio>";
        }
        return "<MultimodalInput type=unknown>";
      });

  // Bind helper functions using lambdas
  m.def(
      "make_token_input",
      [](py::sequence tokens) -> MultimodalInput {
        std::vector<uint64_t> vec;
        vec.reserve(py::len(tokens));
        for (auto item : tokens) {
          uint64_t v = py::cast<uint64_t>(item);
          vec.push_back(v);
        }
        return MultimodalInput(std::move(vec));
      },
      "Create a token input from a Python sequence of ints",
      py::arg("tokens"));

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

        image_tensor = image_tensor.contiguous();
        if (image_tensor.scalar_type() == torch::kUInt8) {
          uint8_t* data = image_tensor.data_ptr<uint8_t>();
          std::vector<uint8_t> image_data(data, data + image_tensor.numel());
          return MultimodalInput(Image(
              std::move(image_data),
              static_cast<int32_t>(width),
              static_cast<int32_t>(height),
              static_cast<int32_t>(channels)));
        } else if (image_tensor.scalar_type() == torch::kFloat) {
          float* data = image_tensor.data_ptr<float>();
          std::vector<float> image_data(data, data + image_tensor.numel());
          return MultimodalInput(Image(
              std::move(image_data),
              static_cast<int32_t>(width),
              static_cast<int32_t>(height),
              static_cast<int32_t>(channels)));
        } else {
          throw std::runtime_error(
              "Unsupported image tensor dtype. Only uint8 and float32 are supported.");
        }
      },
      "Create an image input from a torch tensor (H, W, C), (1, H, W, C), (C, H, W), or (1, C, H, W)",
      py::arg("image_tensor"));

  m.def(
      "make_audio_input",
      [](torch::Tensor audio_tensor) -> MultimodalInput {
        if (audio_tensor.dim() != 3) {
          throw std::runtime_error(
              "Audio tensor must be 3-dimensional (batch_size, n_bins, n_frames)");
        }

        int64_t batch_size = audio_tensor.size(0);
        int64_t n_bins = audio_tensor.size(1);
        int64_t n_frames = audio_tensor.size(2);

        audio_tensor = audio_tensor.contiguous();
        if (audio_tensor.scalar_type() == torch::kUInt8) {
          uint8_t* data = audio_tensor.data_ptr<uint8_t>();
          std::vector<uint8_t> audio_data(data, data + audio_tensor.numel());
          return MultimodalInput(Audio(
              std::move(audio_data),
              static_cast<int32_t>(batch_size),
              static_cast<int32_t>(n_bins),
              static_cast<int32_t>(n_frames)));
        } else if (audio_tensor.scalar_type() == torch::kFloat) {
          float* data = audio_tensor.data_ptr<float>();
          std::vector<float> audio_data(data, data + audio_tensor.numel());
          return MultimodalInput(Audio(
              std::move(audio_data),
              static_cast<int32_t>(batch_size),
              static_cast<int32_t>(n_bins),
              static_cast<int32_t>(n_frames)));
        } else {
          throw std::runtime_error(
              "Unsupported audio tensor dtype. Only uint8 and float32 are supported for preprocessed audio.");
        }
      },
      "Create a preprocessed audio input from a torch tensor (batch_size, n_bins, n_frames)",
      py::arg("audio_tensor"));

  m.def(
      "make_raw_audio_input",
      [](torch::Tensor audio_tensor) -> MultimodalInput {
        if (audio_tensor.dim() != 3) {
          throw std::runtime_error(
              "Raw audio tensor must be 3-dimensional (batch_size, n_channels, n_samples)");
        }

        int64_t batch_size = audio_tensor.size(0);
        int64_t n_channels = audio_tensor.size(1);
        int64_t n_samples = audio_tensor.size(2);

        audio_tensor = audio_tensor.contiguous();
        if (audio_tensor.scalar_type() == torch::kUInt8) {
          uint8_t* data = audio_tensor.data_ptr<uint8_t>();
          std::vector<uint8_t> audio_data(data, data + audio_tensor.numel());
          return MultimodalInput(RawAudio{
              std::move(audio_data),
              static_cast<int32_t>(batch_size),
              static_cast<int32_t>(n_channels),
              static_cast<int32_t>(n_samples)});
        } else {
          throw std::runtime_error(
              "Unsupported raw audio tensor dtype. Only uint8 is supported for raw audio.");
        }
      },
      "Create a raw audio input from a torch tensor (batch_size, n_channels, n_samples)",
      py::arg("audio_tensor"));

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
      .def(
          "prefill",
          &PyMultimodalRunner::prefill,
          py::arg("inputs"),
          "Prefill multimodal inputs (e.g., chat history) without generating tokens")
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