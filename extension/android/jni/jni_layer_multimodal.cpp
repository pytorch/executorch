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

#include <executorch/examples/models/llava/runner/multimodal_runner.h>
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

  __android_log_print(android_log_level, "MULTIMODAL", "%s", message);
}
#endif

using namespace torch::executor;

namespace executorch_jni {

class ExecuTorchMultiModalCallbackJni
    : public facebook::jni::JavaClass<ExecuTorchMultiModalCallbackJni> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/MultiModalCallback;";

  void onResult(std::string result) const {
    static auto cls = ExecuTorchMultiModalCallbackJni::javaClassStatic();
    static const auto method =
        cls->getMethod<void(facebook::jni::local_ref<jstring>)>("onResult");
    facebook::jni::local_ref<jstring> s = facebook::jni::make_jstring(result);
    method(self(), s);
  }

  void onStats(const MultiModalRunner::Stats& result) const {
    static auto cls = ExecuTorchMultiModalCallbackJni::javaClassStatic();
    static const auto method = cls->getMethod<void(jfloat)>("onStats");
    double eval_time =
        (double)(result.inference_end_ms - result.prompt_eval_end_ms);

    float tps = result.num_generated_tokens / eval_time *
        result.SCALING_FACTOR_UNITS_PER_SECOND;

    method(self(), tps);
  }
};

class ExecuTorchMultiModalJni
    : public facebook::jni::HybridClass<ExecuTorchMultiModalJni> {
 private:
  friend HybridBase;
  std::unique_ptr<MultiModalRunner> runner_;

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/MultiModalModule;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature) {
    return makeCxxInstance(model_path, tokenizer_path, temperature);
  }

  ExecuTorchMultiModalJni(
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

    runner_ = std::make_unique<MultiModalRunner>(
        model_path->toStdString().c_str(),
        tokenizer_path->toStdString().c_str(),
        temperature);
  }

  jint generate(
      facebook::jni::alias_ref<jintArray> image,
      jint width,
      jint height,
      jint channels,
      facebook::jni::alias_ref<jstring> prompt,
      jint startPos,
      facebook::jni::alias_ref<ExecuTorchMultiModalCallbackJni> callback) {
    auto image_size = image->size();
    std::vector<jint> image_data_jint(image_size);
    std::vector<uint8_t> image_data(image_size);
    image->getRegion(0, image_size, image_data_jint.data());
    for (int i = 0; i < image_size; i++) {
      image_data[i] = image_data_jint[i];
    }
    Image image_runner{image_data, width, height, channels};
    runner_->generate(
        image_runner,
        prompt->toStdString(),
        1024,
        [callback](std::string result) { callback->onResult(result); },
        [callback](const MultiModalRunner::Stats& result) {
          callback->onStats(result);
        });
    return 0;
  }

  void stop() {
    runner_->stop();
  }

  jint load() {
    return static_cast<jint>(runner_->load());
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchMultiModalJni::initHybrid),
        makeNativeMethod("generate", ExecuTorchMultiModalJni::generate),
        makeNativeMethod("stop", ExecuTorchMultiModalJni::stop),
        makeNativeMethod("load", ExecuTorchMultiModalJni::load),
    });
  }
};

} // namespace executorch_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { executorch_jni::ExecuTorchMultiModalJni::registerNatives(); });
}
