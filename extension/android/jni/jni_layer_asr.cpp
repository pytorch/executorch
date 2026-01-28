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
#include <string>
#include <unordered_set>
#include <vector>

#include <executorch/extension/asr/runner/runner.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/platform/log.h>

namespace asr = ::executorch::extension::asr;
using ::executorch::runtime::Error;

namespace {

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

// Thread-local token buffer for UTF-8 accumulation
thread_local std::string asr_token_buffer;

// Cached JNI references for callback
struct AsrCallbackCache {
  jclass callbackClass = nullptr;
  jmethodID onTokenMethod = nullptr;
  jmethodID onCompleteMethod = nullptr;
  bool initialized = false;

  void init(JNIEnv* env) {
    if (initialized) {
      return;
    }
    jclass localClass =
        env->FindClass("org/pytorch/executorch/extension/asr/AsrCallback");
    if (localClass != nullptr) {
      callbackClass = (jclass)env->NewGlobalRef(localClass);
      onTokenMethod =
          env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");
      onCompleteMethod = env->GetMethodID(
          callbackClass, "onComplete", "(Ljava/lang/String;)V");
      env->DeleteLocalRef(localClass);
      initialized = true;
    }
  }
};

thread_local AsrCallbackCache callbackCache;

} // namespace

extern "C" {

/*
 * Class:     org_pytorch_executorch_extension_asr_AsrModule
 * Method:    nativeCreate
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_extension_asr_AsrModule_nativeCreate(
    JNIEnv* env,
    jobject /* this */,
    jstring modelPath,
    jstring tokenizerPath,
    jstring dataPath) {
  std::string modelPathStr = jstringToString(env, modelPath);
  std::string tokenizerPathStr = jstringToString(env, tokenizerPath);
  std::string dataPathStr = jstringToString(env, dataPath);

  std::optional<std::string> dataPathOpt;
  if (!dataPathStr.empty()) {
    dataPathOpt = dataPathStr;
  }

  try {
    auto* runner =
        new asr::AsrRunner(modelPathStr, dataPathOpt, tokenizerPathStr);
    return reinterpret_cast<jlong>(runner);
  } catch (const std::exception& e) {
    ET_LOG(Error, "Failed to create AsrRunner: %s", e.what());
    env->ThrowNew(
        env->FindClass("java/lang/RuntimeException"),
        ("Failed to create AsrRunner: " + std::string(e.what())).c_str());
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
    auto* runner = reinterpret_cast<asr::AsrRunner*>(nativeHandle);
    delete runner;
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

  auto* runner = reinterpret_cast<asr::AsrRunner*>(nativeHandle);
  Error error = runner->load();
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

  auto* runner = reinterpret_cast<asr::AsrRunner*>(nativeHandle);
  return runner->is_loaded() ? JNI_TRUE : JNI_FALSE;
}

/*
 * Class:     org_pytorch_executorch_extension_asr_AsrModule
 * Method:    nativeTranscribe
 * Signature:
 * (J[FIIIJFJLorg/pytorch/executorch/extension/asr/AsrCallback;)I
 */
JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_asr_AsrModule_nativeTranscribe(
    JNIEnv* env,
    jobject /* this */,
    jlong nativeHandle,
    jfloatArray features,
    jint batchSize,
    jint timeSteps,
    jint featureDim,
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

  if (features == nullptr) {
    env->ThrowNew(
        env->FindClass("java/lang/IllegalArgumentException"),
        "Features array cannot be null");
    return -1;
  }

  auto* runner = reinterpret_cast<asr::AsrRunner*>(nativeHandle);

  // Get features from Java array
  jsize featuresLen = env->GetArrayLength(features);
  if (featuresLen == 0) {
    env->ThrowNew(
        env->FindClass("java/lang/IllegalArgumentException"),
        "Features array cannot be empty");
    return -1;
  }

  // Copy feature data
  std::vector<float> featuresData(featuresLen);
  env->GetFloatArrayRegion(features, 0, featuresLen, featuresData.data());

  // Create tensor from features
  auto featuresTensor = ::executorch::extension::from_blob(
      featuresData.data(),
      {static_cast<::executorch::aten::SizesType>(batchSize),
       static_cast<::executorch::aten::SizesType>(timeSteps),
       static_cast<::executorch::aten::SizesType>(featureDim)},
      ::executorch::aten::ScalarType::Float);

  // Build config
  asr::AsrTranscribeConfig config;
  config.max_new_tokens = static_cast<int64_t>(maxNewTokens);
  config.temperature = temperature;
  config.decoder_start_token_id = static_cast<int64_t>(decoderStartTokenId);

  // Set up callback
  std::function<void(const std::string&)> tokenCallback = nullptr;

  // We need to keep a global ref to the callback for the duration of
  // transcription
  jobject globalCallback = nullptr;
  if (callback != nullptr) {
    globalCallback = env->NewGlobalRef(callback);
    callbackCache.init(env);

    // Reset token buffer
    asr_token_buffer.clear();

    tokenCallback = [env, globalCallback](const std::string& token) {
      asr_token_buffer += token;
      if (!utf8_check_validity(
              asr_token_buffer.c_str(), asr_token_buffer.size())) {
        ET_LOG(
            Info, "Current token buffer is not valid UTF-8. Waiting for more.");
        return;
      }

      std::string completeToken = asr_token_buffer;
      asr_token_buffer.clear();

      jstring jToken = env->NewStringUTF(completeToken.c_str());
      env->CallVoidMethod(
          globalCallback, callbackCache.onTokenMethod, jToken);
      env->DeleteLocalRef(jToken);
    };
  }

  // Run transcription
  auto result = runner->transcribe(featuresTensor, config, tokenCallback);

  // Call onComplete if callback provided
  if (globalCallback != nullptr) {
    jstring emptyStr = env->NewStringUTF("");
    env->CallVoidMethod(
        globalCallback, callbackCache.onCompleteMethod, emptyStr);
    env->DeleteLocalRef(emptyStr);
    env->DeleteGlobalRef(globalCallback);
  }

  if (!result.ok()) {
    return static_cast<jint>(result.error());
  }

  return 0;
}

} // extern "C"
