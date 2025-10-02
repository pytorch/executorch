/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#if defined(EXECUTORCH_BUILD_QNN)
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#endif

#if defined(EXECUTORCH_BUILD_MEDIATEK)
#include <executorch/examples/mediatek/executor_runner/mtk_llama_runner.h>
#endif

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;
using executorch::extension::Module;

namespace {
bool utf8_check_validity(const char* str, size_t length) {
  for (size_t i = 0; i < length; ++i) {
    uint8_t byte = static_cast<uint8_t>(str[i]);
    if (byte >= 0x80) { // Non-ASCII byte
      if (i + 1 >= length) { // Incomplete sequence
        return false;
      }
      uint8_t next_byte = static_cast<uint8_t>(str[i + 1]);
      if ((byte & 0xE0) == 0xC0 &&
          (next_byte & 0xC0) == 0x80) { // 2-byte sequence
        i += 1;
      } else if (
          (byte & 0xF0) == 0xE0 && (next_byte & 0xC0) == 0x80 &&
          (i + 2 < length) &&
          (static_cast<uint8_t>(str[i + 2]) & 0xC0) ==
              0x80) { // 3-byte sequence
        i += 2;
      } else if (
          (byte & 0xF8) == 0xF0 && (next_byte & 0xC0) == 0x80 &&
          (i + 2 < length) &&
          (static_cast<uint8_t>(str[i + 2]) & 0xC0) == 0x80 &&
          (i + 3 < length) &&
          (static_cast<uint8_t>(str[i + 3]) & 0xC0) ==
              0x80) { // 4-byte sequence
        i += 3;
      } else {
        return false; // Invalid sequence
      }
    }
  }
  return true; // All bytes were valid
}

std::string token_buffer;
} // namespace

namespace executorch_jni {

class ExecuTorchLlmCallbackJni
    : public facebook::jni::JavaClass<ExecuTorchLlmCallbackJni> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/llm/LlmCallback;";

  void onResult(std::string result) const {
    static auto cls = ExecuTorchLlmCallbackJni::javaClassStatic();
    static const auto method =
        cls->getMethod<void(facebook::jni::local_ref<jstring>)>("onResult");

    token_buffer += result;
    if (!utf8_check_validity(token_buffer.c_str(), token_buffer.size())) {
      ET_LOG(
          Info, "Current token buffer is not valid UTF-8. Waiting for more.");
      return;
    }
    result = token_buffer;
    token_buffer = "";
    facebook::jni::local_ref<jstring> s = facebook::jni::make_jstring(result);
    method(self(), s);
  }

  void onStats(const llm::Stats& result) const {
    static auto cls = ExecuTorchLlmCallbackJni::javaClassStatic();
    static const auto on_stats_method =
        cls->getMethod<void(facebook::jni::local_ref<jstring>)>("onStats");
    on_stats_method(
        self(),
        facebook::jni::make_jstring(
            executorch::extension::llm::stats_to_json_string(result)));
  }
};

class ExecuTorchLlmJni : public facebook::jni::HybridClass<ExecuTorchLlmJni> {
 private:
  friend HybridBase;
  float temperature_ = 0.0f;
  int model_type_category_;
  std::unique_ptr<llm::IRunner> runner_;
  std::unique_ptr<executorch::extension::llm::MultimodalRunner>
      multi_modal_runner_;

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/llm/LlmModule;";

  constexpr static int MODEL_TYPE_CATEGORY_LLM = 1;
  constexpr static int MODEL_TYPE_CATEGORY_MULTIMODAL = 2;
  constexpr static int MODEL_TYPE_MEDIATEK_LLAMA = 3;
  constexpr static int MODEL_TYPE_QNN_LLAMA = 4;

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature,
      facebook::jni::alias_ref<jstring> data_path) {
    return makeCxxInstance(
        model_type_category,
        model_path,
        tokenizer_path,
        temperature,
        data_path);
  }

  ExecuTorchLlmJni(
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature,
      facebook::jni::alias_ref<jstring> data_path = nullptr) {
    temperature_ = temperature;
#if defined(ET_USE_THREADPOOL)
    // Reserve 1 thread for the main thread.
    int32_t num_performant_cores =
        ::executorch::extension::cpuinfo::get_num_performant_cores() - 1;
    if (num_performant_cores > 0) {
      ET_LOG(Info, "Resetting threadpool to %d threads", num_performant_cores);
      ::executorch::extension::threadpool::get_threadpool()
          ->_unsafe_reset_threadpool(num_performant_cores);
    }
#endif

    model_type_category_ = model_type_category;
    if (model_type_category == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      multi_modal_runner_ = llm::create_multimodal_runner(
          model_path->toStdString().c_str(),
          llm::load_tokenizer(tokenizer_path->toStdString()));
    } else if (model_type_category == MODEL_TYPE_CATEGORY_LLM) {
      std::optional<const std::string> data_path_str = data_path
          ? std::optional<const std::string>{data_path->toStdString()}
          : std::nullopt;
      runner_ = executorch::extension::llm::create_text_llm_runner(
          model_path->toStdString(),
          llm::load_tokenizer(tokenizer_path->toStdString()),
          data_path_str);
#if defined(EXECUTORCH_BUILD_QNN)
    } else if (model_type_category == MODEL_TYPE_QNN_LLAMA) {
      std::unique_ptr<executorch::extension::Module> module = std::make_unique<
          executorch::extension::Module>(
          model_path->toStdString().c_str(),
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
      std::string decoder_model = "llama3"; // use llama3 for now
      runner_ = std::make_unique<example::Runner<uint16_t>>( // QNN runner
          std::move(module),
          decoder_model.c_str(),
          model_path->toStdString().c_str(),
          tokenizer_path->toStdString().c_str(),
          data_path->toStdString().c_str(),
          "");
      model_type_category_ = MODEL_TYPE_CATEGORY_LLM;
#endif
#if defined(EXECUTORCH_BUILD_MEDIATEK)
    } else if (model_type_category == MODEL_TYPE_MEDIATEK_LLAMA) {
      runner_ = std::make_unique<MTKLlamaRunner>(
          model_path->toStdString().c_str(),
          tokenizer_path->toStdString().c_str());
      // Interpret the model type as LLM
      model_type_category_ = MODEL_TYPE_CATEGORY_LLM;
#endif
    }
  }

  jint generate(
      facebook::jni::alias_ref<jstring> prompt,
      jint seq_len,
      facebook::jni::alias_ref<ExecuTorchLlmCallbackJni> callback,
      jboolean echo) {
    if (model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      std::vector<llm::MultimodalInput> inputs;
      if (!prompt->toStdString().empty()) {
        inputs.emplace_back(llm::MultimodalInput{prompt->toStdString()});
      }
      executorch::extension::llm::GenerationConfig config{
          .echo = static_cast<bool>(echo),
          .seq_len = seq_len,
          .temperature = temperature_,
      };
      multi_modal_runner_->generate(
          std::move(inputs),
          config,
          [callback](const std::string& result) { callback->onResult(result); },
          [callback](const llm::Stats& result) { callback->onStats(result); });
    } else if (model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      executorch::extension::llm::GenerationConfig config{
          .echo = static_cast<bool>(echo),
          .seq_len = seq_len,
          .temperature = temperature_,
      };
      runner_->generate(
          prompt->toStdString(),
          config,
          [callback](std::string result) { callback->onResult(result); },
          [callback](const llm::Stats& result) { callback->onStats(result); });
    }
    return 0;
  }

  // Returns status_code
  // Contract is valid within an AAR (JNI + corresponding Java code)
  jint prefill_text_input(facebook::jni::alias_ref<jstring> prompt) {
    if (model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      runner_->prefill(prompt->toStdString(), {});
      return 0;
    } else if (model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      multi_modal_runner_->prefill(
          {llm::MultimodalInput{prompt->toStdString()}});
      return 0;
    }
  }

  jint prefill_images_input(
      facebook::jni::alias_ref<jintArray> image,
      jint width,
      jint height,
      jint channels) {
    if (model_type_category_ != MODEL_TYPE_CATEGORY_MULTIMODAL) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    if (image == nullptr) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    std::vector<llm::Image> images;
    if (image == nullptr) {
      return static_cast<jint>(Error::EndOfMethod);
    }
    auto image_size = image->size();
    if (image_size != 0) {
      std::vector<jint> image_data_jint(image_size);
      std::vector<uint8_t> image_data(image_size);
      image->getRegion(0, image_size, image_data_jint.data());
      for (int i = 0; i < image_size; i++) {
        image_data[i] = image_data_jint[i];
      }
      llm::Image image_runner{std::move(image_data), width, height, channels};
      multi_modal_runner_->prefill(
          {llm::MultimodalInput{std::move(image_runner)}});
    }

    return 0;
  }

  llm::MultimodalInput processRawAudioFile(
    const std::string& audio_path,
    const std::string& processor_path) {
  if (processor_path.empty()) {
    ET_LOG(Error, "Processor path is required for raw audio processing");
    throw std::runtime_error(
        "Processor path is required for raw audio processing");
  }

  // Load the audio processor .pte.
  std::unique_ptr<Module> processor_module;
  try {
    processor_module =
        std::make_unique<Module>(processor_path, Module::LoadMode::File);
    auto load_error = processor_module->load();
    if (load_error != ::executorch::runtime::Error::Ok) {
      ET_LOG(
          Error,
          "Failed to load processor module from: %s",
          processor_path.c_str());
      throw std::runtime_error("Failed to load processor module");
    }
  } catch (const std::exception& e) {
    ET_LOG(Error, "Exception while loading processor module: %s", e.what());
    throw std::runtime_error("Exception while loading processor module");
  }

  // Load the audio data from file.
  std::ifstream f(audio_path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    ET_LOG(Error, "Failed to open audio file: %s", audio_path.c_str());
    throw std::runtime_error("Failed to open audio file");
  }

  std::size_t n_floats = f.tellg() / sizeof(float);
  f.seekg(0, std::ios::beg);

  std::vector<float> audio_data(n_floats);
  f.read(
      reinterpret_cast<char*>(audio_data.data()),
      audio_data.size() * sizeof(float));
  f.close();

  ET_LOG(
      Info, "Loaded .bin file: %s, %zu floats", audio_path.c_str(), n_floats);

  // Execute the processor
  std::vector<executorch::aten::SizesType> tensor_shape = {
      static_cast<executorch::aten::SizesType>(audio_data.size())};
  auto input_tensor = executorch::extension::from_blob(
      audio_data.data(), tensor_shape, ::executorch::aten::ScalarType::Float);

  ET_LOG(Info, "Processing audio through processor module...");
  auto result = processor_module->execute("forward", input_tensor);
  if (!result.ok()) {
    ET_LOG(Error, "Failed to execute processor's forward method");
    throw std::runtime_error("Failed to execute processor forward method");
  }

  auto outputs = result.get();
  if (outputs.empty()) {
    ET_LOG(Error, "Processor returned no outputs");
    throw std::runtime_error("Processor returned no outputs");
  }

  // Extract processed audio features
  const auto& processed_tensor = outputs[0].toTensor();
  const float* processed_data = processed_tensor.const_data_ptr<float>();
  const auto& sizes = processed_tensor.sizes();

  ET_LOG(
      Info,
      "Processed audio tensor shape: [%d, %d, %d]",
      static_cast<int>(sizes[0]),
      static_cast<int>(sizes[1]),
      static_cast<int>(sizes[2]));

  // Create Audio multimodal input from processed features
  int32_t batch_size = static_cast<int32_t>(sizes[0]);
  int32_t n_bins = static_cast<int32_t>(sizes[1]);
  int32_t n_frames = static_cast<int32_t>(sizes[2]);
  size_t total_elements = batch_size * n_bins * n_frames;
  std::vector<float> audio_vec(processed_data, processed_data + total_elements);
  auto processed_audio = ::executorch::extension::llm::Audio(
      std::move(audio_vec), batch_size, n_bins, n_frames);
  ET_LOG(
      Info,
      "Created processed Audio: batch_size=%d, n_bins=%d, n_frames=%d",
      batch_size,
      n_bins,
      n_frames);
  return ::executorch::extension::llm::make_audio_input(
      std::move(processed_audio));
}

  jint prefill_audio_input(
      facebook::jni::alias_ref<jbyteArray> audio,
      jint batch_size,
      jint n_bins,
      jint n_frames) {
    if (model_type_category_ != MODEL_TYPE_CATEGORY_MULTIMODAL) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    if (audio == nullptr) {
      return static_cast<jint>(Error::InvalidArgument);
    }
    auto audio_size = audio->size();
    std::vector<uint8_t> audio_data(audio_size);
    if (audio_size != 0) {
      std::vector<jbyte> audio_data_jbyte(audio_size);
      audio->getRegion(0, audio_size, audio_data_jbyte.data());
      for (int i = 0; i < audio_size; i++) {
        audio_data[i] = audio_data_jbyte[i];
      }
      llm::Audio audio_input{std::move(audio_data), batch_size, n_bins, n_frames};
      multi_modal_runner_->prefill(
          {processRawAudioFile("/data/local/tmp/llama/audio.bin", "/data/local/tmp/llama/voxtral_preprocessor.pte")});
    }
    return 0;
  }

  void stop() {
    if (model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      multi_modal_runner_->stop();
    } else if (model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      runner_->stop();
    }
  }

  void reset_context() {
    if (runner_ != nullptr) {
      runner_->reset();
    }
    if (multi_modal_runner_ != nullptr) {
      multi_modal_runner_->reset();
    }
  }

  jint load() {
    if (model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      return static_cast<jint>(multi_modal_runner_->load());
    } else if (model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      return static_cast<jint>(runner_->load());
    }
    return static_cast<jint>(Error::InvalidArgument);
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchLlmJni::initHybrid),
        makeNativeMethod("generate", ExecuTorchLlmJni::generate),
        makeNativeMethod("stop", ExecuTorchLlmJni::stop),
        makeNativeMethod("load", ExecuTorchLlmJni::load),
        makeNativeMethod(
            "appendImagesInput", ExecuTorchLlmJni::prefill_images_input),
        makeNativeMethod(
            "appendTextInput", ExecuTorchLlmJni::prefill_text_input),
        makeNativeMethod(
            "appendAudioInput", ExecuTorchLlmJni::prefill_audio_input),
        makeNativeMethod("resetContext", ExecuTorchLlmJni::reset_context),
    });
  }
};

} // namespace executorch_jni

void register_natives_for_llm() {
  executorch_jni::ExecuTorchLlmJni::registerNatives();
}
