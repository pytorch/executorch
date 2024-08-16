/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/examples/models/llama2/runner/runner.h>
#include <executorch/examples/models/llava/runner/llava_runner.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/backends/xnnpack/threadpool/cpuinfo_utils.h>
#include <executorch/backends/xnnpack/threadpool/threadpool.h>
#endif

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#ifdef __ANDROID__
#include <android/log.h>

// For Android, write to logcat
void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
  int android_log_level = ANDROID_LOG_UNKNOWN;
  if (level == 'D') {
    android_log_level = ANDROID_LOG_DEBUG;
  } else if (level == 'I') {
    android_log_level = ANDROID_LOG_INFO;
  } else if (level == 'E') {
    android_log_level = ANDROID_LOG_ERROR;
  } else if (level == 'F') {
    android_log_level = ANDROID_LOG_FATAL;
  }

  __android_log_print(android_log_level, "LLAMA", "%s", message);
}
#endif

using namespace torch::executor;

namespace executorch_jni {

class ExecuTorchLlamaCallbackJni
    : public facebook::jni::JavaClass<ExecuTorchLlamaCallbackJni> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/LlamaCallback;";

  void onResult(std::string result) const {
    static auto cls = ExecuTorchLlamaCallbackJni::javaClassStatic();
    static const auto method =
        cls->getMethod<void(facebook::jni::local_ref<jstring>)>("onResult");
    facebook::jni::local_ref<jstring> s = facebook::jni::make_jstring(result);
    method(self(), s);
  }

  void onStats(const Stats& result) const {
    static auto cls = ExecuTorchLlamaCallbackJni::javaClassStatic();
    static const auto method = cls->getMethod<void(jfloat)>("onStats");
    double eval_time =
        (double)(result.inference_end_ms - result.prompt_eval_end_ms);

    float tps = result.num_generated_tokens / eval_time *
        result.SCALING_FACTOR_UNITS_PER_SECOND;

    method(self(), tps);
  }
};

class ExecuTorchLlamaJni
    : public facebook::jni::HybridClass<ExecuTorchLlamaJni> {
 private:
  friend HybridBase;
  int model_type_category_;
  std::unique_ptr<Runner> runner_;
  std::unique_ptr<MultimodalRunner> multi_modal_runner_;

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/LlamaModule;";

  constexpr static int MODEL_TYPE_CATEGORY_LLM = 1;
  constexpr static int MODEL_TYPE_CATEGORY_MULTIMODAL = 2;

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature) {
    return makeCxxInstance(model_type_category, model_path, tokenizer_path, temperature);
  }

  ExecuTorchLlamaJni(
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature) {
#if defined(ET_USE_THREADPOOL)
    // Reserve 1 thread for the main thread.
    uint32_t num_performant_cores =
        torch::executorch::cpuinfo::get_num_performant_cores() - 1;
    if (num_performant_cores > 0) {
      ET_LOG(Info, "Resetting threadpool to %d threads", num_performant_cores);
      torch::executorch::threadpool::get_threadpool()->_unsafe_reset_threadpool(
          num_performant_cores);
    }
#endif

    model_type_category_ = model_type_category;
    if (model_type_category == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      multi_modal_runner_ = std::make_unique<LlavaRunner>(
          model_path->toStdString().c_str(),
          tokenizer_path->toStdString().c_str(),
          temperature);
    } else if (model_type_category == MODEL_TYPE_CATEGORY_LLM) {
      runner_ = std::make_unique<Runner>(
          model_path->toStdString().c_str(),
          tokenizer_path->toStdString().c_str(),
          temperature);
    }
  }

  jint generate(
      facebook::jni::alias_ref<jintArray> image,
      jint width,
      jint height,
      jint channels,
      facebook::jni::alias_ref<jstring> prompt,
      jint seq_len,
      facebook::jni::alias_ref<ExecuTorchLlamaCallbackJni> callback) {
    if (model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      auto image_size = image->size();
      std::vector<Image> images;
      if (image_size != 0) {
        std::vector<jint> image_data_jint(image_size);
        std::vector<uint8_t> image_data(image_size);
        image->getRegion(0, image_size, image_data_jint.data());
        for (int i = 0; i < image_size; i++) {
          image_data[i] = image_data_jint[i];
        }
        Image image_runner{image_data, width, height, channels};
        images.push_back(image_runner);
      }
      multi_modal_runner_->generate(
          images,
          prompt->toStdString(),
          seq_len,
          [callback](std::string result) { callback->onResult(result); },
          [callback](const Stats& result) { callback->onStats(result); });
    } else if (model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      runner_->generate(
          prompt->toStdString(),
          seq_len,
          [callback](std::string result) { callback->onResult(result); },
          [callback](const Stats& result) { callback->onStats(result); });
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
        makeNativeMethod("initHybrid", ExecuTorchLlamaJni::initHybrid),
        makeNativeMethod("generate", ExecuTorchLlamaJni::generate),
        makeNativeMethod("stop", ExecuTorchLlamaJni::stop),
        makeNativeMethod("load", ExecuTorchLlamaJni::load),
    });
  }
};

} // namespace executorch_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { executorch_jni::ExecuTorchLlamaJni::registerNatives(); });
}
