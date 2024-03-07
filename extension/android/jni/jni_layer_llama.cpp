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

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

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
    runner_ = std::make_unique<Runner>(
        model_path->toStdString().c_str(),
        tokenizer_path->toStdString().c_str(),
        temperature);
  }

  jint generate(
      facebook::jni::alias_ref<jstring> prompt,
      facebook::jni::alias_ref<ExecuTorchLlamaCallbackJni> callback) {
    runner_->generate(
        prompt->toStdString(), 128, [callback](std::string result) {
          callback->onResult(result);
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
