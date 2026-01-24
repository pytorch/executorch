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
#include <sstream>
#include <unordered_map>
#include <vector>
#include <jni.h>

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

struct LlmWrapper {
    int model_type_category_;
    float temperature_;
    std::unique_ptr<llm::IRunner> runner_;
    std::unique_ptr<executorch::extension::llm::MultimodalRunner> multi_modal_runner_;
    std::vector<llm::MultimodalInput> prefill_inputs_;
    std::string token_buffer; // Per-instance token buffer

    LlmWrapper(int model_type_category, float temperature)
        : model_type_category_(model_type_category), temperature_(temperature) {}
};

constexpr int MODEL_TYPE_CATEGORY_LLM = 1;
constexpr int MODEL_TYPE_CATEGORY_MULTIMODAL = 2;
constexpr int MODEL_TYPE_MEDIATEK_LLAMA = 3;
constexpr int MODEL_TYPE_QNN_LLAMA = 4;

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeInit(
    JNIEnv* env,
    jclass clazz,
    jint model_type_category,
    jstring model_path,
    jstring tokenizer_path,
    jfloat temperature,
    jobject data_files) {

    (void)clazz; // Unused

    const char* model_path_ptr = env->GetStringUTFChars(model_path, nullptr);
    const char* tokenizer_path_ptr = env->GetStringUTFChars(tokenizer_path, nullptr);
    
    std::string model_path_str(model_path_ptr);
    std::string tokenizer_path_str(tokenizer_path_ptr);

    env->ReleaseStringUTFChars(model_path, model_path_ptr);
    env->ReleaseStringUTFChars(tokenizer_path, tokenizer_path_ptr);

    auto wrapper = std::make_unique<LlmWrapper>(model_type_category, temperature);

#if defined(ET_USE_THREADPOOL)
    int32_t num_performant_cores =
        ::executorch::extension::cpuinfo::get_num_performant_cores() - 1;
    if (num_performant_cores > 0) {
      ET_LOG(Info, "Resetting threadpool to %d threads", num_performant_cores);
      ::executorch::extension::threadpool::get_threadpool()
          ->_unsafe_reset_threadpool(num_performant_cores);
    }
#endif

    std::vector<std::string> data_files_vector;
    if (data_files != nullptr) {
        // Handle List<String>
        jclass list_class = env->FindClass("java/util/List");
        jmethodID size_method = env->GetMethodID(list_class, "size", "()I");
        jmethodID get_method = env->GetMethodID(list_class, "get", "(I)Ljava/lang/Object;");
        
        jint size = env->CallIntMethod(data_files, size_method);
        for(jint i = 0; i < size; ++i) {
            jstring jstr = (jstring)env->CallObjectMethod(data_files, get_method, i);
            const char* cstr = env->GetStringUTFChars(jstr, nullptr);
            data_files_vector.emplace_back(cstr);
            env->ReleaseStringUTFChars(jstr, cstr);
            env->DeleteLocalRef(jstr);
        }
        env->DeleteLocalRef(list_class);
    }

    if (model_type_category == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      wrapper->multi_modal_runner_ = llm::create_multimodal_runner(
          model_path_str.c_str(),
          llm::load_tokenizer(tokenizer_path_str));
    } else if (model_type_category == MODEL_TYPE_CATEGORY_LLM) {
        wrapper->runner_ = executorch::extension::llm::create_text_llm_runner(
          model_path_str,
          llm::load_tokenizer(tokenizer_path_str),
          data_files_vector);
#if defined(EXECUTORCH_BUILD_QNN)
    } else if (model_type_category == MODEL_TYPE_QNN_LLAMA) {
      std::unique_ptr<executorch::extension::Module> module = std::make_unique<
          executorch::extension::Module>(
          model_path_str.c_str(),
          data_files_vector,
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
      std::string decoder_model = "llama3"; // use llama3 for now
      wrapper->runner_ = std::make_unique<example::Runner<uint16_t>>( // QNN runner
          std::move(module),
          decoder_model.c_str(),
          model_path_str.c_str(),
          tokenizer_path_str.c_str(),
          "",
          "");
      wrapper->model_type_category_ = MODEL_TYPE_CATEGORY_LLM;
#endif
#if defined(EXECUTORCH_BUILD_MEDIATEK)
    } else if (model_type_category == MODEL_TYPE_MEDIATEK_LLAMA) {
      wrapper->runner_ = std::make_unique<MTKLlamaRunner>(
          model_path_str.c_str(),
          tokenizer_path_str.c_str());
      // Interpret the model type as LLM
      wrapper->model_type_category_ = MODEL_TYPE_CATEGORY_LLM;
#endif
    }

    return reinterpret_cast<jlong>(wrapper.release());
}

JNIEXPORT void JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle) {
    if (handle != 0) {
        delete reinterpret_cast<LlmWrapper*>(handle);
    }
}

JNIEXPORT jint JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeLoad(
    JNIEnv* env, jobject thiz, jlong handle) {
    
    LlmWrapper* wrapper = reinterpret_cast<LlmWrapper*>(handle);
    int result = -1;
    std::stringstream ss;

    if (wrapper->model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      result = static_cast<jint>(wrapper->multi_modal_runner_->load());
      if (result != 0) {
        ss << "Failed to load multimodal runner: [" << result << "]";
      }
    } else if (wrapper->model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      result = static_cast<jint>(wrapper->runner_->load());
      if (result != 0) {
        ss << "Failed to load llm runner: [" << result << "]";
      }
    } else {
      ss << "Invalid model type category: " << wrapper->model_type_category_
         << ". Valid values are: " << MODEL_TYPE_CATEGORY_LLM << " or "
         << MODEL_TYPE_CATEGORY_MULTIMODAL;
    }
    
    if (result != 0) {
       // Using jni_helper to throw exception
       executorch::jni_helper::throwExecutorchException(
           env, ss.str().c_str());
    }
    return result; 
}

JNIEXPORT void JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeStop(
    JNIEnv* env, jobject thiz, jlong handle) {
    LlmWrapper* wrapper = reinterpret_cast<LlmWrapper*>(handle);
    if (wrapper->model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      wrapper->multi_modal_runner_->stop();
    } else if (wrapper->model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      wrapper->runner_->stop();
    }
}

JNIEXPORT void JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeResetContext(
    JNIEnv* env, jobject thiz, jlong handle) {
    LlmWrapper* wrapper = reinterpret_cast<LlmWrapper*>(handle);
    if (wrapper->runner_ != nullptr) {
      wrapper->runner_->reset();
    }
    if (wrapper->multi_modal_runner_ != nullptr) {
      wrapper->multi_modal_runner_->reset();
    }
}

JNIEXPORT jint JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendImagesInput(
    JNIEnv* env, jobject thiz, jlong handle, jintArray image, jint width, jint height, jint channels) {
    LlmWrapper* wrapper = reinterpret_cast<LlmWrapper*>(handle);
    
    if (image == nullptr) {
      return static_cast<jint>(Error::EndOfMethod);
    }
    jsize image_size = env->GetArrayLength(image);
    if (image_size != 0) {
      jint* image_data_ptr = env->GetIntArrayElements(image, nullptr);
      std::vector<uint8_t> image_data(image_size);
      for (int i = 0; i < image_size; i++) {
        image_data[i] = static_cast<uint8_t>(image_data_ptr[i]);
      }
      env->ReleaseIntArrayElements(image, image_data_ptr, JNI_ABORT);

      llm::Image image_runner{std::move(image_data), width, height, channels};
      wrapper->prefill_inputs_.emplace_back(
          llm::MultimodalInput{std::move(image_runner)});
    }
    return 0;
}

JNIEXPORT jint JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendNormalizedImagesInput(
    JNIEnv* env, jobject thiz, jlong handle, jfloatArray image, jint width, jint height, jint channels) {
    LlmWrapper* wrapper = reinterpret_cast<LlmWrapper*>(handle);
    if (image == nullptr) {
      return static_cast<jint>(Error::EndOfMethod);
    }
    jsize image_size = env->GetArrayLength(image);
    if (image_size != 0) {
      jfloat* image_data_ptr = env->GetFloatArrayElements(image, nullptr);
      std::vector<float> image_data(image_size);
      for (int i = 0; i < image_size; i++) {
        image_data[i] = image_data_ptr[i];
      }
      env->ReleaseFloatArrayElements(image, image_data_ptr, JNI_ABORT);

      llm::Image image_runner{std::move(image_data), width, height, channels};
      wrapper->prefill_inputs_.emplace_back(
          llm::MultimodalInput{std::move(image_runner)});
    }
    return 0;
}

JNIEXPORT jint JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendAudioInput(
    JNIEnv* env, jobject thiz, jlong handle, jbyteArray data, jint batch_size, jint n_bins, jint n_frames) {
    LlmWrapper* wrapper = reinterpret_cast<LlmWrapper*>(handle);
    if (data == nullptr) {
        return static_cast<jint>(Error::EndOfMethod);
    }
    jsize data_size = env->GetArrayLength(data);
    if (data_size != 0) {
        jbyte* data_ptr = env->GetByteArrayElements(data, nullptr);
        std::vector<uint8_t> data_u8(data_size);
        for(int i=0; i<data_size; ++i) {
            data_u8[i] = static_cast<uint8_t>(data_ptr[i]);
        }
        env->ReleaseByteArrayElements(data, data_ptr, JNI_ABORT);
        llm::Audio audio{std::move(data_u8), batch_size, n_bins, n_frames};
        wrapper->prefill_inputs_.emplace_back(llm::MultimodalInput{std::move(audio)});
    }
    return 0;
}

JNIEXPORT jint JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendAudioInputFloat(
    JNIEnv* env, jobject thiz, jlong handle, jfloatArray data, jint batch_size, jint n_bins, jint n_frames) {
    LlmWrapper* wrapper = reinterpret_cast<LlmWrapper*>(handle);
    if (data == nullptr) {
        return static_cast<jint>(Error::EndOfMethod);
    }
    jsize data_size = env->GetArrayLength(data);
    if (data_size != 0) {
        jfloat* data_ptr = env->GetFloatArrayElements(data, nullptr);
        std::vector<float> data_f(data_size);
        for (int i = 0; i < data_size; i++) {
            data_f[i] = data_ptr[i];
        }
        env->ReleaseFloatArrayElements(data, data_ptr, JNI_ABORT);
        llm::Audio audio{std::move(data_f), batch_size, n_bins, n_frames};
        wrapper->prefill_inputs_.emplace_back(llm::MultimodalInput{std::move(audio)});
    }
    return 0;
}

JNIEXPORT jint JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendRawAudioInput(
    JNIEnv* env, jobject thiz, jlong handle, jbyteArray data, jint batch_size, jint n_channels, jint n_samples) {
    LlmWrapper* wrapper = reinterpret_cast<LlmWrapper*>(handle);
    if (data == nullptr) {
        return static_cast<jint>(Error::EndOfMethod);
    }
    jsize data_size = env->GetArrayLength(data);
    if (data_size != 0) {
        jbyte* data_ptr = env->GetByteArrayElements(data, nullptr);
        std::vector<uint8_t> data_u8(data_size);
        for(int i=0; i<data_size; ++i) {
            data_u8[i] = static_cast<uint8_t>(data_ptr[i]);
        }
        env->ReleaseByteArrayElements(data, data_ptr, JNI_ABORT);
        llm::RawAudio audio{std::move(data_u8), batch_size, n_channels, n_samples};
        wrapper->prefill_inputs_.emplace_back(llm::MultimodalInput{std::move(audio)});
    }
    return 0;
}

JNIEXPORT jint JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeAppendTextInput(
    JNIEnv* env, jobject thiz, jlong handle, jstring prompt) {
    LlmWrapper* wrapper = reinterpret_cast<LlmWrapper*>(handle);
    const char* prompt_ptr = env->GetStringUTFChars(prompt, nullptr);
    if (prompt_ptr) {
        wrapper->prefill_inputs_.emplace_back(llm::MultimodalInput{std::string(prompt_ptr)});
        env->ReleaseStringUTFChars(prompt, prompt_ptr);
    }
    return 0;
}

JNIEXPORT jint JNICALL Java_org_pytorch_executorch_extension_llm_LlmModule_nativeGenerate(
    JNIEnv* env, jobject thiz, jlong handle, jstring prompt, jint seq_len, jobject callback, jboolean echo, jfloat temperature) {
    
    LlmWrapper* wrapper = reinterpret_cast<LlmWrapper*>(handle);
    float effective_temperature = temperature >= 0 ? temperature : wrapper->temperature_;
    
    const char* prompt_ptr = env->GetStringUTFChars(prompt, nullptr);
    std::string prompt_str = prompt_ptr ? std::string(prompt_ptr) : "";
    if (prompt_ptr) env->ReleaseStringUTFChars(prompt, prompt_ptr);

    // Prepare callback
    // Note: To be safe with threads, we should ensure the callback object is accessible. 
    // LLM runner might be synchronous or asynchronous. The original code used lambda. 
    // Assuming synchronous for now or that env remains valid if on same thread. 
    // If background thread, we need JVM attachment. TextLLMRunner seems sync.
    
    jclass callback_class = env->GetObjectClass(callback);
    jmethodID on_result_method = env->GetMethodID(callback_class, "onResult", "(Ljava/lang/String;)V");
    jmethodID on_stats_method = env->GetMethodID(callback_class, "onStats", "(Ljava/lang/String;)V");

    auto on_result = [env, callback, on_result_method, wrapper](std::string result) {
        // Accumulate and validate UTF-8
        wrapper->token_buffer += result;
        if (!utf8_check_validity(wrapper->token_buffer.c_str(), wrapper->token_buffer.size())) {
             ET_LOG(Info, "Current token buffer is not valid UTF-8. Waiting for more.");
             return;
        }
        std::string valid_result = wrapper->token_buffer;
        wrapper->token_buffer = "";
        
        jstring jres = env->NewStringUTF(valid_result.c_str());
        env->CallVoidMethod(callback, on_result_method, jres);
        env->DeleteLocalRef(jres);
    };

    auto on_stats = [env, callback, on_stats_method](const llm::Stats& stats) {
        std::string stats_str = executorch::extension::llm::stats_to_json_string(stats);
        jstring jstats = env->NewStringUTF(stats_str.c_str());
        env->CallVoidMethod(callback, on_stats_method, jstats);
        env->DeleteLocalRef(jstats);
    };

    if (wrapper->model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      std::vector<llm::MultimodalInput> inputs = wrapper->prefill_inputs_;
      wrapper->prefill_inputs_.clear();
      if (!prompt_str.empty()) {
        inputs.emplace_back(llm::MultimodalInput{prompt_str});
      }
      executorch::extension::llm::GenerationConfig config{
          .echo = static_cast<bool>(echo),
          .seq_len = seq_len,
          .temperature = effective_temperature,
      };
      wrapper->multi_modal_runner_->generate(
          std::move(inputs),
          config,
          on_result,
          on_stats);
    } else if (wrapper->model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      executorch::extension::llm::GenerationConfig config{
          .echo = static_cast<bool>(echo),
          .seq_len = seq_len,
          .temperature = effective_temperature,
      };
      wrapper->runner_->generate(
          prompt_str,
          config,
          on_result,
          on_stats);
    }
    return 0;
}

} // extern "C"
