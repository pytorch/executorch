/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <jni.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/extension/android/jni/jni_helper.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#if defined(EXECUTORCH_BUILD_QNN)
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#endif

#if defined(EXECUTORCH_BUILD_MEDIATEK)
#include <executorch/examples/mediatek/executor_runner/mtk_llama_runner.h>
#endif

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;

namespace {

// Global JavaVM pointer for obtaining JNIEnv in callbacks
JavaVM* g_jvm = nullptr;

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

// Helper to convert jstring to std::string
std::string jstring_to_string(JNIEnv* env, jstring jstr) {
  if (jstr == nullptr) {
    return "";
  }
  const char* chars = env->GetStringUTFChars(jstr, nullptr);
  if (chars == nullptr) {
    return "";
  }
  std::string result(chars);
  env->ReleaseStringUTFChars(jstr, chars);
  return result;
}

// Helper to convert Java List<String> to std::vector<std::string>
std::vector<std::string> jlist_to_string_vector(JNIEnv* env, jobject jlist) {
  std::vector<std::string> result;
  if (jlist == nullptr) {
    return result;
  }

  jclass list_class = env->FindClass("java/util/List");
  if (list_class == nullptr) {
    env->ExceptionClear();
    return result;
  }

  jmethodID size_method = env->GetMethodID(list_class, "size", "()I");
  jmethodID get_method =
      env->GetMethodID(list_class, "get", "(I)Ljava/lang/Object;");

  if (size_method == nullptr || get_method == nullptr) {
    env->ExceptionClear();
    env->DeleteLocalRef(list_class);
    return result;
  }

  jint size = env->CallIntMethod(jlist, size_method);
  for (jint i = 0; i < size; ++i) {
    jobject str_obj = env->CallObjectMethod(jlist, get_method, i);
    if (str_obj != nullptr) {
      result.push_back(jstring_to_string(env, static_cast<jstring>(str_obj)));
      env->DeleteLocalRef(str_obj);
    }
  }

  env->DeleteLocalRef(list_class);
  return result;
}

} // namespace

namespace executorch_jni {

// Model type category constants
constexpr int MODEL_TYPE_CATEGORY_LLM = 1;
constexpr int MODEL_TYPE_CATEGORY_MULTIMODAL = 2;
constexpr int MODEL_TYPE_MEDIATEK_LLAMA = 3;
constexpr int MODEL_TYPE_QNN_LLAMA = 4;

// Native handle class that holds the runner state
class ExecuTorchLlmNative {
 public:
  float temperature_ = 0.0f;
  int model_type_category_;
  std::unique_ptr<llm::IRunner> runner_;
  std::unique_ptr<executorch::extension::llm::MultimodalRunner>
      multi_modal_runner_;
  std::vector<llm::MultimodalInput> prefill_inputs_;

  ExecuTorchLlmNative(
      JNIEnv* env,
      jint model_type_category,
      jstring model_path,
      jstring tokenizer_path,
      jfloat temperature,
      jobject data_files) {
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
    std::string model_path_str = jstring_to_string(env, model_path);
    std::string tokenizer_path_str = jstring_to_string(env, tokenizer_path);
    std::vector<std::string> data_files_vector =
        jlist_to_string_vector(env, data_files);

    if (model_type_category == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      multi_modal_runner_ = llm::create_multimodal_runner(
          model_path_str.c_str(), llm::load_tokenizer(tokenizer_path_str));
    } else if (model_type_category == MODEL_TYPE_CATEGORY_LLM) {
      runner_ = executorch::extension::llm::create_text_llm_runner(
          model_path_str, llm::load_tokenizer(tokenizer_path_str), data_files_vector);
#if defined(EXECUTORCH_BUILD_QNN)
    } else if (model_type_category == MODEL_TYPE_QNN_LLAMA) {
      std::unique_ptr<executorch::extension::Module> module = std::make_unique<
          executorch::extension::Module>(
          model_path_str.c_str(),
          data_files_vector,
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
      std::string decoder_model = "llama3"; // use llama3 for now
      runner_ = std::make_unique<example::Runner<uint16_t>>( // QNN runner
          std::move(module),
          decoder_model.c_str(),
          model_path_str.c_str(),
          tokenizer_path_str.c_str(),
          "",
          "");
      model_type_category_ = MODEL_TYPE_CATEGORY_LLM;
#endif
#if defined(EXECUTORCH_BUILD_MEDIATEK)
    } else if (model_type_category == MODEL_TYPE_MEDIATEK_LLAMA) {
      runner_ = std::make_unique<MTKLlamaRunner>(
          model_path_str.c_str(), tokenizer_path_str.c_str());
      // Interpret the model type as LLM
      model_type_category_ = MODEL_TYPE_CATEGORY_LLM;
#endif
    }
  }
};

// Helper class for callback invocation
class CallbackHelper {
 public:
  CallbackHelper(JNIEnv* env, jobject callback)
      : env_(env), callback_(nullptr), callback_class_(nullptr) {
    if (callback != nullptr) {
      callback_ = env_->NewGlobalRef(callback);
      jclass local_class = env_->GetObjectClass(callback);
      callback_class_ = static_cast<jclass>(env_->NewGlobalRef(local_class));
      env_->DeleteLocalRef(local_class);
      on_result_method_ = env_->GetMethodID(
          callback_class_, "onResult", "(Ljava/lang/String;)V");
      on_stats_method_ =
          env_->GetMethodID(callback_class_, "onStats", "(Ljava/lang/String;)V");
    }
  }

  ~CallbackHelper() {
    if (g_jvm == nullptr) {
      return;
    }
    // Get the current JNIEnv (might be different thread)
    JNIEnv* env = nullptr;
    int status = g_jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
      g_jvm->AttachCurrentThread(&env, nullptr);
    }
    if (env != nullptr) {
      if (callback_ != nullptr) {
        env->DeleteGlobalRef(callback_);
      }
      if (callback_class_ != nullptr) {
        env->DeleteGlobalRef(callback_class_);
      }
    }
  }

  void onResult(const std::string& result) {
    JNIEnv* env = getEnv();
    if (env == nullptr || callback_ == nullptr || on_result_method_ == nullptr) {
      return;
    }

    std::string current_result = result;
    token_buffer += current_result;
    if (!utf8_check_validity(token_buffer.c_str(), token_buffer.size())) {
      ET_LOG(
          Info, "Current token buffer is not valid UTF-8. Waiting for more.");
      return;
    }
    current_result = token_buffer;
    token_buffer = "";

    jstring jstr = env->NewStringUTF(current_result.c_str());
    if (jstr != nullptr) {
      env->CallVoidMethod(callback_, on_result_method_, jstr);
      env->DeleteLocalRef(jstr);
    }
  }

  void onStats(const llm::Stats& stats) {
    JNIEnv* env = getEnv();
    if (env == nullptr || callback_ == nullptr || on_stats_method_ == nullptr) {
      return;
    }

    std::string stats_json =
        executorch::extension::llm::stats_to_json_string(stats);
    jstring jstr = env->NewStringUTF(stats_json.c_str());
    if (jstr != nullptr) {
      env->CallVoidMethod(callback_, on_stats_method_, jstr);
      env->DeleteLocalRef(jstr);
    }
  }

 private:
  JNIEnv* getEnv() {
    if (g_jvm == nullptr) {
      return nullptr;
    }
    JNIEnv* env = nullptr;
    int status = g_jvm->GetEnv((void**)&env, JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
      g_jvm->AttachCurrentThread(&env, nullptr);
    }
    return env;
  }

  JNIEnv* env_;
  jobject callback_;
  jclass callback_class_ = nullptr;
  jmethodID on_result_method_ = nullptr;
  jmethodID on_stats_method_ = nullptr;
};

} // namespace executorch_jni

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeCreate(
    JNIEnv* env,
    jobject /* this */,
    jint model_type_category,
    jstring model_path,
    jstring tokenizer_path,
    jfloat temperature,
    jobject data_files) {
  auto* native = new executorch_jni::ExecuTorchLlmNative(
      env, model_type_category, model_path, tokenizer_path, temperature, data_files);
  return reinterpret_cast<jlong>(native);
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeDestroy(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong native_handle) {
  auto* native =
      reinterpret_cast<executorch_jni::ExecuTorchLlmNative*>(native_handle);
  delete native;
}

JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeGenerate(
    JNIEnv* env,
    jobject /* this */,
    jlong native_handle,
    jstring prompt,
    jint seq_len,
    jobject callback,
    jboolean echo) {
  auto* native =
      reinterpret_cast<executorch_jni::ExecuTorchLlmNative*>(native_handle);
  if (native == nullptr) {
    return -1;
  }

  std::string prompt_str = jstring_to_string(env, prompt);

  // Create a shared callback helper for use in lambdas
  auto callback_helper =
      std::make_shared<executorch_jni::CallbackHelper>(env, callback);

  if (native->model_type_category_ ==
      executorch_jni::MODEL_TYPE_CATEGORY_MULTIMODAL) {
    std::vector<llm::MultimodalInput> inputs = native->prefill_inputs_;
    native->prefill_inputs_.clear();
    if (!prompt_str.empty()) {
      inputs.emplace_back(llm::MultimodalInput{prompt_str});
    }
    executorch::extension::llm::GenerationConfig config{
        .echo = static_cast<bool>(echo),
        .seq_len = seq_len,
        .temperature = native->temperature_,
    };
    native->multi_modal_runner_->generate(
        std::move(inputs),
        config,
        [callback_helper](const std::string& result) {
          callback_helper->onResult(result);
        },
        [callback_helper](const llm::Stats& result) {
          callback_helper->onStats(result);
        });
  } else if (
      native->model_type_category_ ==
      executorch_jni::MODEL_TYPE_CATEGORY_LLM) {
    executorch::extension::llm::GenerationConfig config{
        .echo = static_cast<bool>(echo),
        .seq_len = seq_len,
        .temperature = native->temperature_,
    };
    native->runner_->generate(
        prompt_str,
        config,
        [callback_helper](std::string result) {
          callback_helper->onResult(result);
        },
        [callback_helper](const llm::Stats& result) {
          callback_helper->onStats(result);
        });
  }
  return 0;
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeStop(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong native_handle) {
  auto* native =
      reinterpret_cast<executorch_jni::ExecuTorchLlmNative*>(native_handle);
  if (native == nullptr) {
    return;
  }

  if (native->model_type_category_ ==
      executorch_jni::MODEL_TYPE_CATEGORY_MULTIMODAL) {
    native->multi_modal_runner_->stop();
  } else if (
      native->model_type_category_ ==
      executorch_jni::MODEL_TYPE_CATEGORY_LLM) {
    native->runner_->stop();
  }
}

JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeLoad(
    JNIEnv* env,
    jobject /* this */,
    jlong native_handle) {
  auto* native =
      reinterpret_cast<executorch_jni::ExecuTorchLlmNative*>(native_handle);
  if (native == nullptr) {
    return -1;
  }

  int result = -1;
  std::stringstream ss;

  if (native->model_type_category_ ==
      executorch_jni::MODEL_TYPE_CATEGORY_MULTIMODAL) {
    result = static_cast<jint>(native->multi_modal_runner_->load());
    if (result != 0) {
      ss << "Failed to load multimodal runner: [" << result << "]";
    }
  } else if (
      native->model_type_category_ ==
      executorch_jni::MODEL_TYPE_CATEGORY_LLM) {
    result = static_cast<jint>(native->runner_->load());
    if (result != 0) {
      ss << "Failed to load llm runner: [" << result << "]";
    }
  } else {
    ss << "Invalid model type category: " << native->model_type_category_
       << ". Valid values are: "
       << executorch_jni::MODEL_TYPE_CATEGORY_LLM << " or "
       << executorch_jni::MODEL_TYPE_CATEGORY_MULTIMODAL;
  }
  if (result != 0) {
    executorch::jni_helper::throwExecutorchException(
        env, static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
  }
  return result; // 0 on success to keep backward compatibility
}

JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendTextInput(
    JNIEnv* env,
    jobject /* this */,
    jlong native_handle,
    jstring prompt) {
  auto* native =
      reinterpret_cast<executorch_jni::ExecuTorchLlmNative*>(native_handle);
  if (native == nullptr) {
    return -1;
  }

  std::string prompt_str = jstring_to_string(env, prompt);
  native->prefill_inputs_.emplace_back(llm::MultimodalInput{prompt_str});
  return 0;
}

JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendImagesInput(
    JNIEnv* env,
    jobject /* this */,
    jlong native_handle,
    jintArray image,
    jint width,
    jint height,
    jint channels) {
  auto* native =
      reinterpret_cast<executorch_jni::ExecuTorchLlmNative*>(native_handle);
  if (native == nullptr) {
    return -1;
  }

  if (image == nullptr) {
    return static_cast<jint>(Error::EndOfMethod);
  }

  jsize image_size = env->GetArrayLength(image);
  if (image_size != 0) {
    std::vector<jint> image_data_jint(image_size);
    std::vector<uint8_t> image_data(image_size);
    env->GetIntArrayRegion(image, 0, image_size, image_data_jint.data());
    for (int i = 0; i < image_size; i++) {
      image_data[i] = static_cast<uint8_t>(image_data_jint[i]);
    }
    llm::Image image_runner{std::move(image_data), width, height, channels};
    native->prefill_inputs_.emplace_back(
        llm::MultimodalInput{std::move(image_runner)});
  }

  return 0;
}

JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendNormalizedImagesInput(
    JNIEnv* env,
    jobject /* this */,
    jlong native_handle,
    jfloatArray image,
    jint width,
    jint height,
    jint channels) {
  auto* native =
      reinterpret_cast<executorch_jni::ExecuTorchLlmNative*>(native_handle);
  if (native == nullptr) {
    return -1;
  }

  if (image == nullptr) {
    return static_cast<jint>(Error::EndOfMethod);
  }

  jsize image_size = env->GetArrayLength(image);
  if (image_size != 0) {
    std::vector<jfloat> image_data_jfloat(image_size);
    std::vector<float> image_data(image_size);
    env->GetFloatArrayRegion(image, 0, image_size, image_data_jfloat.data());
    for (int i = 0; i < image_size; i++) {
      image_data[i] = image_data_jfloat[i];
    }
    llm::Image image_runner{std::move(image_data), width, height, channels};
    native->prefill_inputs_.emplace_back(
        llm::MultimodalInput{std::move(image_runner)});
  }

  return 0;
}

JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendAudioInput(
    JNIEnv* env,
    jobject /* this */,
    jlong native_handle,
    jbyteArray data,
    jint batch_size,
    jint n_bins,
    jint n_frames) {
  auto* native =
      reinterpret_cast<executorch_jni::ExecuTorchLlmNative*>(native_handle);
  if (native == nullptr) {
    return -1;
  }

  if (data == nullptr) {
    return static_cast<jint>(Error::EndOfMethod);
  }

  jsize data_size = env->GetArrayLength(data);
  if (data_size != 0) {
    std::vector<jbyte> data_jbyte(data_size);
    std::vector<uint8_t> data_u8(data_size);
    env->GetByteArrayRegion(data, 0, data_size, data_jbyte.data());
    for (int i = 0; i < data_size; i++) {
      data_u8[i] = static_cast<uint8_t>(data_jbyte[i]);
    }
    llm::Audio audio{std::move(data_u8), batch_size, n_bins, n_frames};
    native->prefill_inputs_.emplace_back(llm::MultimodalInput{std::move(audio)});
  }
  return 0;
}

JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendAudioInputFloat(
    JNIEnv* env,
    jobject /* this */,
    jlong native_handle,
    jfloatArray data,
    jint batch_size,
    jint n_bins,
    jint n_frames) {
  auto* native =
      reinterpret_cast<executorch_jni::ExecuTorchLlmNative*>(native_handle);
  if (native == nullptr) {
    return -1;
  }

  if (data == nullptr) {
    return static_cast<jint>(Error::EndOfMethod);
  }

  jsize data_size = env->GetArrayLength(data);
  if (data_size != 0) {
    std::vector<jfloat> data_jfloat(data_size);
    std::vector<float> data_f(data_size);
    env->GetFloatArrayRegion(data, 0, data_size, data_jfloat.data());
    for (int i = 0; i < data_size; i++) {
      data_f[i] = data_jfloat[i];
    }
    llm::Audio audio{std::move(data_f), batch_size, n_bins, n_frames};
    native->prefill_inputs_.emplace_back(llm::MultimodalInput{std::move(audio)});
  }
  return 0;
}

JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendRawAudioInput(
    JNIEnv* env,
    jobject /* this */,
    jlong native_handle,
    jbyteArray data,
    jint batch_size,
    jint n_channels,
    jint n_samples) {
  auto* native =
      reinterpret_cast<executorch_jni::ExecuTorchLlmNative*>(native_handle);
  if (native == nullptr) {
    return -1;
  }

  if (data == nullptr) {
    return static_cast<jint>(Error::EndOfMethod);
  }

  jsize data_size = env->GetArrayLength(data);
  if (data_size != 0) {
    std::vector<jbyte> data_jbyte(data_size);
    std::vector<uint8_t> data_u8(data_size);
    env->GetByteArrayRegion(data, 0, data_size, data_jbyte.data());
    for (int i = 0; i < data_size; i++) {
      data_u8[i] = static_cast<uint8_t>(data_jbyte[i]);
    }
    llm::RawAudio audio{std::move(data_u8), batch_size, n_channels, n_samples};
    native->prefill_inputs_.emplace_back(llm::MultimodalInput{std::move(audio)});
  }
  return 0;
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_llm_LlmModule_nativeResetContext(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong native_handle) {
  auto* native =
      reinterpret_cast<executorch_jni::ExecuTorchLlmNative*>(native_handle);
  if (native == nullptr) {
    return;
  }

  if (native->runner_ != nullptr) {
    native->runner_->reset();
  }
  if (native->multi_modal_runner_ != nullptr) {
    native->multi_modal_runner_->reset();
  }
}

} // extern "C"

void register_natives_for_llm(JNIEnv* env) {
  // Store the JavaVM for later use in callbacks
  env->GetJavaVM(&g_jvm);

  jclass llm_module_class =
      env->FindClass("org/pytorch/executorch/extension/llm/LlmModule");
  if (llm_module_class == nullptr) {
    ET_LOG(Error, "Failed to find LlmModule class");
    env->ExceptionClear();
    return;
  }

  // clang-format off
  static const JNINativeMethod methods[] = {
      {"nativeCreate",
       "(ILjava/lang/String;Ljava/lang/String;FLjava/util/List;)J",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeCreate)},
      {"nativeDestroy", "(J)V",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeDestroy)},
      {"nativeGenerate",
       "(JLjava/lang/String;ILorg/pytorch/executorch/extension/llm/LlmCallback;Z)I",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeGenerate)},
      {"nativeStop", "(J)V",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeStop)},
      {"nativeLoad", "(J)I",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeLoad)},
      {"nativeAppendTextInput", "(JLjava/lang/String;)I",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendTextInput)},
      {"nativeAppendImagesInput", "(J[IIII)I",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendImagesInput)},
      {"nativeAppendNormalizedImagesInput", "(J[FIII)I",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendNormalizedImagesInput)},
      {"nativeAppendAudioInput", "(J[BIII)I",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendAudioInput)},
      {"nativeAppendAudioInputFloat", "(J[FIII)I",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendAudioInputFloat)},
      {"nativeAppendRawAudioInput", "(J[BIII)I",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendRawAudioInput)},
      {"nativeResetContext", "(J)V",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_extension_llm_LlmModule_nativeResetContext)},
  };
  // clang-format on

  int num_methods = sizeof(methods) / sizeof(methods[0]);
  int result = env->RegisterNatives(llm_module_class, methods, num_methods);
  if (result != JNI_OK) {
    ET_LOG(Error, "Failed to register native methods for LlmModule");
  }

  env->DeleteLocalRef(llm_module_class);
}
