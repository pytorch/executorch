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

  void onStats(const Runner::Stats& result) const {
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
  std::unique_ptr<Runner> runner_;

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/LlamaModule;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature) {
    return makeCxxInstance(model_path, tokenizer_path, temperature);
  }

  ExecuTorchLlamaJni(
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

    runner_ = std::make_unique<Runner>(
        model_path->toStdString().c_str(),
        tokenizer_path->toStdString().c_str(),
        temperature);
  }

  jint generate(
      facebook::jni::alias_ref<jstring> prompt,
      facebook::jni::alias_ref<ExecuTorchLlamaCallbackJni> callback) {
    runner_->generate(
        prompt->toStdString(),
        128,
        [callback](std::string result) { callback->onResult(result); },
        [callback](const Runner::Stats& result) { callback->onStats(result); });
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
