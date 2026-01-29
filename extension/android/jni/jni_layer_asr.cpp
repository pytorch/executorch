/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <jni.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <executorch/extension/asr/runner/runner.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/platform/log.h>

namespace asr = ::executorch::extension::asr;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;

namespace {

// Handle struct that holds both the ASR runner and optional preprocessor
struct AsrModuleHandle {
  std::unique_ptr<asr::AsrRunner> runner;
  std::unique_ptr<Module> preprocessor;
};

// Helper to get a string from jstring
std::string jstringToString(JNIEnv* env, jstring jstr) {
  if (jstr == nullptr) {
    return "";
  }
  const char* chars = env->GetStringUTFChars(jstr, nullptr);
  std::string result(chars);
  env->ReleaseStringUTFChars(jstr, chars);
  return result;
}

// Helper for UTF-8 validity checking (for streaming tokens)
bool utf8_check_validity(const char* str, size_t length) {
  for (size_t i = 0; i < length; ++i) {
    uint8_t byte = static_cast<uint8_t>(str[i]);
    if (byte >= 0x80) {
      if (i + 1 >= length) {
        return false;
      }
      uint8_t next_byte = static_cast<uint8_t>(str[i + 1]);
      if ((byte & 0xE0) == 0xC0 && (next_byte & 0xC0) == 0x80) {
        i += 1;
      } else if (
          (byte & 0xF0) == 0xE0 && (next_byte & 0xC0) == 0x80 &&
          (i + 2 < length) &&
          (static_cast<uint8_t>(str[i + 2]) & 0xC0) == 0x80) {
        i += 2;
      } else if (
          (byte & 0xF8) == 0xF0 && (next_byte & 0xC0) == 0x80 &&
          (i + 2 < length) &&
          (static_cast<uint8_t>(str[i + 2]) & 0xC0) == 0x80 &&
          (i + 3 < length) &&
          (static_cast<uint8_t>(str[i + 3]) & 0xC0) == 0x80) {
        i += 3;
      } else {
        return false;
      }
    }
  }
  return true;
}

// Global cached JNI references for callback (shared across threads)
struct AsrCallbackCache {
  jclass callbackClass = nullptr;
  jmethodID onTokenMethod = nullptr;
  jmethodID onCompleteMethod = nullptr;
};

AsrCallbackCache callbackCache;
std::once_flag callbackCacheInitFlag;

void initCallbackCache(JNIEnv* env) {
  std::call_once(callbackCacheInitFlag, [env]() {
    jclass localClass =
        env->FindClass("org/pytorch/executorch/extension/asr/AsrCallback");
    if (localClass != nullptr) {
      callbackCache.callbackClass = (jclass)env->NewGlobalRef(localClass);
      callbackCache.onTokenMethod = env->GetMethodID(
          callbackCache.callbackClass, "onToken", "(Ljava/lang/String;)V");
      callbackCache.onCompleteMethod = env->GetMethodID(
          callbackCache.callbackClass, "onComplete", "(Ljava/lang/String;)V");
      env->DeleteLocalRef(localClass);
    }
  });
}

// Helper to create a unique_ptr for JNI global references
auto make_scoped_global_ref(JNIEnv* env, jobject obj) {
  auto deleter = [env](jobject ref) {
    if (ref != nullptr) {
      env->DeleteGlobalRef(ref);
    }
  };
  jobject globalRef = obj ? env->NewGlobalRef(obj) : nullptr;
  return std::unique_ptr<std::remove_pointer_t<jobject>, decltype(deleter)>(
      globalRef, deleter);
}

} // namespace

extern "C" {

/*
 * Class:     org_pytorch_executorch_extension_asr_AsrModule
 * Method:    nativeCreate
 * Signature:
 * (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_extension_asr_AsrModule_nativeCreate(
    JNIEnv* env,
    jobject /* this */,
    jstring modelPath,
    jstring tokenizerPath,
    jstring dataPath,
    jstring preprocessorPath) {
  std::string modelPathStr = jstringToString(env, modelPath);
  std::string tokenizerPathStr = jstringToString(env, tokenizerPath);
  std::string dataPathStr = jstringToString(env, dataPath);
  std::string preprocessorPathStr = jstringToString(env, preprocessorPath);

  std::optional<std::string> dataPathOpt;
  if (!dataPathStr.empty()) {
    dataPathOpt = dataPathStr;
  }

  try {
    auto handle = std::make_unique<AsrModuleHandle>();

    // Create the ASR runner
    handle->runner = std::make_unique<asr::AsrRunner>(
        modelPathStr, dataPathOpt, tokenizerPathStr);

    // Create the preprocessor module if path is provided
    if (!preprocessorPathStr.empty()) {
      handle->preprocessor =
          std::make_unique<Module>(preprocessorPathStr, Module::LoadMode::Mmap);
      auto load_error = handle->preprocessor->load();
      if (load_error != Error::Ok) {
        ET_LOG(Error, "Failed to load preprocessor module");
        env->ThrowNew(
            env->FindClass("java/lang/RuntimeException"),
            "Failed to load preprocessor module");
        return 0;
      }
    }

    return reinterpret_cast<jlong>(handle.release());
  } catch (const std::exception& e) {
    ET_LOG(Error, "Failed to create AsrModule: %s", e.what());
    env->ThrowNew(
        env->FindClass("java/lang/RuntimeException"),
        ("Failed to create AsrModule: " + std::string(e.what())).c_str());
    return 0;
  }
}

/*
 * Class:     org_pytorch_executorch_extension_asr_AsrModule
 * Method:    nativeDestroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_asr_AsrModule_nativeDestroy(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong nativeHandle) {
  if (nativeHandle != 0) {
    auto* handle = reinterpret_cast<AsrModuleHandle*>(nativeHandle);
    delete handle;
  }
}

/*
 * Class:     org_pytorch_executorch_extension_asr_AsrModule
 * Method:    nativeLoad
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_asr_AsrModule_nativeLoad(
    JNIEnv* env,
    jobject /* this */,
    jlong nativeHandle) {
  if (nativeHandle == 0) {
    env->ThrowNew(
        env->FindClass("java/lang/IllegalStateException"),
        "Module has been destroyed");
    return -1;
  }

  auto* handle = reinterpret_cast<AsrModuleHandle*>(nativeHandle);
  Error error = handle->runner->load();
  return static_cast<jint>(error);
}

/*
 * Class:     org_pytorch_executorch_extension_asr_AsrModule
 * Method:    nativeIsLoaded
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL
Java_org_pytorch_executorch_extension_asr_AsrModule_nativeIsLoaded(
    JNIEnv* env,
    jobject /* this */,
    jlong nativeHandle) {
  if (nativeHandle == 0) {
    return JNI_FALSE;
  }

  auto* handle = reinterpret_cast<AsrModuleHandle*>(nativeHandle);
  return handle->runner->is_loaded() ? JNI_TRUE : JNI_FALSE;
}

/*
 * Class:     org_pytorch_executorch_extension_asr_AsrModule
 * Method:    nativeTranscribe
 * Signature:
 * (JLjava/lang/String;JFJLorg/pytorch/executorch/extension/asr/AsrCallback;)I
 */
JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_asr_AsrModule_nativeTranscribe(
    JNIEnv* env,
    jobject /* this */,
    jlong nativeHandle,
    jstring wavPath,
    jlong maxNewTokens,
    jfloat temperature,
    jlong decoderStartTokenId,
    jobject callback) {
  if (nativeHandle == 0) {
    env->ThrowNew(
        env->FindClass("java/lang/IllegalStateException"),
        "Module has been destroyed");
    return -1;
  }

  if (wavPath == nullptr) {
    env->ThrowNew(
        env->FindClass("java/lang/IllegalArgumentException"),
        "WAV path cannot be null");
    return -1;
  }

  auto* handle = reinterpret_cast<AsrModuleHandle*>(nativeHandle);
  std::string wavPathStr = jstringToString(env, wavPath);

  // Load audio data from WAV file
  std::vector<float> audioData;
  try {
    audioData = ::executorch::extension::llm::load_wav_audio_data(wavPathStr);
  } catch (const std::exception& e) {
    env->ThrowNew(
        env->FindClass("java/lang/RuntimeException"),
        ("Failed to load WAV file: " + std::string(e.what())).c_str());
    return -1;
  }

  if (audioData.empty()) {
    env->ThrowNew(
        env->FindClass("java/lang/IllegalArgumentException"),
        "WAV file contains no audio data");
    return -1;
  }

  ET_LOG(Info, "Loaded %zu audio samples from WAV file", audioData.size());

  // Create tensor from audio data
  TensorPtr featuresTensor;

  if (handle->preprocessor) {
    // Run preprocessor to convert raw audio to features
    auto audioTensor = from_blob(
        audioData.data(),
        {static_cast<::executorch::aten::SizesType>(audioData.size())},
        ::executorch::aten::ScalarType::Float);

    auto processedResult = handle->preprocessor->execute("forward", audioTensor);
    if (processedResult.error() != Error::Ok) {
      env->ThrowNew(
          env->FindClass("java/lang/RuntimeException"),
          "Audio preprocessing failed");
      return -1;
    }

    auto outputs = std::move(processedResult.get());
    if (outputs.empty() || !outputs[0].isTensor()) {
      env->ThrowNew(
          env->FindClass("java/lang/RuntimeException"),
          "Preprocessor returned unexpected output");
      return -1;
    }

    auto tensor = outputs[0].toTensor();
    featuresTensor =
        std::make_shared<::executorch::aten::Tensor>(std::move(tensor));

    ET_LOG(
        Info,
        "Preprocessor output shape: %d dims",
        static_cast<int>(featuresTensor->dim()));
  } else {
    // No preprocessor - use raw audio as features (1D tensor)
    // This is for models that expect raw waveform input
    featuresTensor = from_blob(
        audioData.data(),
        {1,
         static_cast<::executorch::aten::SizesType>(audioData.size()),
         1},
        ::executorch::aten::ScalarType::Float);
  }

  // Build config
  asr::AsrTranscribeConfig config;
  config.max_new_tokens = static_cast<int64_t>(maxNewTokens);
  config.temperature = temperature;
  config.decoder_start_token_id = static_cast<int64_t>(decoderStartTokenId);

  // Set up callback
  std::function<void(const std::string&)> tokenCallback = nullptr;

  // Use unique_ptr with custom deleter to ensure global ref is released
  auto scopedCallback = make_scoped_global_ref(env, callback);

  // Local token buffer for UTF-8 accumulation (per-call, not shared)
  std::string tokenBuffer;

  if (scopedCallback) {
    initCallbackCache(env);

    jobject callbackRef = scopedCallback.get();
    tokenCallback = [env, callbackRef, &tokenBuffer](const std::string& token) {
      tokenBuffer += token;
      if (!utf8_check_validity(tokenBuffer.c_str(), tokenBuffer.size())) {
        ET_LOG(
            Info, "Current token buffer is not valid UTF-8. Waiting for more.");
        return;
      }

      std::string completeToken = tokenBuffer;
      tokenBuffer.clear();

      jstring jToken = env->NewStringUTF(completeToken.c_str());
      env->CallVoidMethod(callbackRef, callbackCache.onTokenMethod, jToken);
      if (env->ExceptionCheck()) {
        ET_LOG(Error, "Exception occurred in AsrCallback.onToken");
        env->ExceptionClear();
      }
      env->DeleteLocalRef(jToken);
    };
  }

  // Run transcription
  auto result = handle->runner->transcribe(featuresTensor, config, tokenCallback);

  // Call onComplete if callback provided
  if (scopedCallback) {
    jstring emptyStr = env->NewStringUTF("");
    env->CallVoidMethod(
        scopedCallback.get(), callbackCache.onCompleteMethod, emptyStr);
    if (env->ExceptionCheck()) {
      ET_LOG(Error, "Exception occurred in AsrCallback.onComplete");
      env->ExceptionClear();
    }
    env->DeleteLocalRef(emptyStr);
  }

  if (!result.ok()) {
    return static_cast<jint>(result.error());
  }

  return 0;
}

} // extern "C"
