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

#include <executorch/extension/llm/runner/asr_runner.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#if defined(EXECUTORCH_BUILD_QNN)
#include <executorch/examples/qualcomm/oss_scripts/whisper/runner/runner.h>
#endif

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

using ::executorch::runtime::Error;

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

class ExecuTorchASRCallbackJni
    : public facebook::jni::JavaClass<ExecuTorchASRCallbackJni> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/audio/ASRCallback;";

  void onResult(std::string result) const {
    static auto cls = ExecuTorchASRCallbackJni::javaClassStatic();
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
};

class ExecuTorchASRJni : public facebook::jni::HybridClass<ExecuTorchASRJni> {
 private:
  friend HybridBase;
  std::unique_ptr<::executorch::extension::llm::ASRRunner> runner_;

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/extension/audio/ASRModule;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path) {
    return makeCxxInstance(model_path, tokenizer_path);
  }

  ExecuTorchASRJni(
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path) {
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
#if defined(EXECUTORCH_BUILD_QNN)
    // create runner
    runner_ = std::make_unique<example::WhisperRunner>(
        model_path->toStdString(), tokenizer_path->toStdString());
#endif
  }

  jint transcribe(
      jint seq_len,
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<jbyteArray>::javaobject> inputs,
      facebook::jni::alias_ref<ExecuTorchASRCallbackJni> callback) {
    // Convert Java byte[][] to C++ vector<vector<char>>
    std::vector<std::vector<char>> cppData;
    auto input_size = inputs->size();

    for (jsize i = 0; i < input_size; i++) {
      auto byte_array = inputs->getElement(i);
      if (byte_array) {
        auto array_length = byte_array->size();
        auto bytes = byte_array->getRegion(0, array_length);
        std::vector<char> charVector;
        charVector.reserve(array_length);
        for (jsize j = 0; j < array_length; j++) {
          charVector.push_back(static_cast<char>(bytes[j]));
        }
        cppData.push_back(std::move(charVector));
      }
    }

    runner_->transcribe(seq_len, cppData, [callback](std::string result) {
      callback->onResult(result);
    });
    return 0;
  }

  jint load() {
    return static_cast<jint>(runner_->load());
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchASRJni::initHybrid),
        makeNativeMethod("transcribe", ExecuTorchASRJni::transcribe),
        makeNativeMethod("load", ExecuTorchASRJni::load),
    });
  }
};

} // namespace executorch_jni

void register_natives_for_asr() {
  executorch_jni::ExecuTorchASRJni::registerNatives();
}
