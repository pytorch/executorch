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
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;

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

  void onStats(const llm::Stats& result) const {
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
  std::unique_ptr<example::Runner> runner_;
  std::unique_ptr<llm::MultimodalRunner> multi_modal_runner_;

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
    return makeCxxInstance(
        model_type_category, model_path, tokenizer_path, temperature);
  }

  ExecuTorchLlamaJni(
      jint model_type_category,
      facebook::jni::alias_ref<jstring> model_path,
      facebook::jni::alias_ref<jstring> tokenizer_path,
      jfloat temperature) {
#if defined(ET_USE_THREADPOOL)
    // Reserve 1 thread for the main thread.
    uint32_t num_performant_cores =
        ::executorch::extension::cpuinfo::get_num_performant_cores() - 1;
    if (num_performant_cores > 0) {
      ET_LOG(Info, "Resetting threadpool to %d threads", num_performant_cores);
      ::executorch::extension::threadpool::get_threadpool()
          ->_unsafe_reset_threadpool(num_performant_cores);
    }
#endif

    model_type_category_ = model_type_category;
    if (model_type_category == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      multi_modal_runner_ = std::make_unique<example::LlavaRunner>(
          model_path->toStdString().c_str(),
          tokenizer_path->toStdString().c_str(),
          temperature);
    } else if (model_type_category == MODEL_TYPE_CATEGORY_LLM) {
      runner_ = std::make_unique<example::Runner>(
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
      facebook::jni::alias_ref<ExecuTorchLlamaCallbackJni> callback,
      jboolean echo) {
    if (model_type_category_ == MODEL_TYPE_CATEGORY_MULTIMODAL) {
      auto image_size = image->size();
      std::vector<llm::Image> images;
      if (image_size != 0) {
        std::vector<jint> image_data_jint(image_size);
        std::vector<uint8_t> image_data(image_size);
        image->getRegion(0, image_size, image_data_jint.data());
        for (int i = 0; i < image_size; i++) {
          image_data[i] = image_data_jint[i];
        }
        llm::Image image_runner{image_data, width, height, channels};
        images.push_back(image_runner);
      }
      multi_modal_runner_->generate(
          std::move(images),
          prompt->toStdString(),
          seq_len,
          [callback](std::string result) { callback->onResult(result); },
          [callback](const llm::Stats& result) { callback->onStats(result); },
          echo);
    } else if (model_type_category_ == MODEL_TYPE_CATEGORY_LLM) {
      runner_->generate(
          prompt->toStdString(),
          seq_len,
          [callback](std::string result) { callback->onResult(result); },
          [callback](const llm::Stats& result) { callback->onStats(result); },
          echo);
    }
    return 0;
  }

  // Returns a tuple of (error, start_pos)
  // Contract is valid within an AAR (JNI + corresponding Java code)
  // If the first element is not Error::Ok, the other element is undefined.
  facebook::jni::local_ref<jlongArray> prefill_prompt(
      facebook::jni::alias_ref<jstring> prompt,
      jlong start_pos,
      jint bos,
      jint eos) {
    facebook::jni::local_ref<jlongArray> tuple_result =
        facebook::jni::make_long_array(2);
    if (model_type_category_ != MODEL_TYPE_CATEGORY_MULTIMODAL) {
      tuple_result->pin()[0] = static_cast<jint>(Error::NotSupported);
      return tuple_result;
    }

    auto&& result = multi_modal_runner_->prefill_prompt(
        prompt->toStdString(), start_pos, bos, eos);
    tuple_result->pin()[0] = static_cast<jint>(Error::Ok);
    if (result.ok()) {
      tuple_result->pin()[1] = static_cast<jlong>(start_pos);
    }
    return tuple_result;
  }

  // Returns a tuple of (error, start_pos)
  // Contract is valid within an AAR (JNI + corresponding Java code)
  // If the first element is not Error::Ok, the other element is undefined.

  facebook::jni::local_ref<jlongArray> prefill_images(
      facebook::jni::alias_ref<jintArray> image,
      jint width,
      jint height,
      jint channels,
      jlong start_pos) {
    facebook::jni::local_ref<jlongArray> tuple_result =
        facebook::jni::make_long_array(2);

    if (model_type_category_ != MODEL_TYPE_CATEGORY_MULTIMODAL) {
      tuple_result->pin()[0] = static_cast<jint>(Error::NotSupported);
      return tuple_result;
    }

    auto image_size = image->size();
    std::vector<llm::Image> images;
    if (image_size != 0) {
      std::vector<jint> image_data_jint(image_size);
      std::vector<uint8_t> image_data(image_size);
      image->getRegion(0, image_size, image_data_jint.data());
      for (int i = 0; i < image_size; i++) {
        image_data[i] = image_data_jint[i];
      }
      llm::Image image_runner{image_data, width, height, channels};
      images.push_back(image_runner);
    }
    // TODO(hsz): make  start_pos a reference and update it here
    jint result = static_cast<jint>(
        multi_modal_runner_->prefill_images(images, start_pos));
    tuple_result->pin()[0] = result;
    tuple_result->pin()[1] = static_cast<jlong>(start_pos);
    return tuple_result;
  }

  jint generate_from_pos(
      facebook::jni::alias_ref<jstring> prompt,
      jint seq_len,
      jlong start_pos,
      facebook::jni::alias_ref<ExecuTorchLlamaCallbackJni> callback,
      jboolean echo) {
    if (model_type_category_ != MODEL_TYPE_CATEGORY_MULTIMODAL) {
      return static_cast<jint>(Error::NotSupported);
    }
    return static_cast<jint>(multi_modal_runner_->generate_from_pos(
        prompt->toStdString(),
        seq_len,
        start_pos,
        [callback](const std::string& result) { callback->onResult(result); },
        [callback](const llm::Stats& stats) { callback->onStats(stats); },
        echo));
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
        makeNativeMethod(
            "prefillImagesNative", ExecuTorchLlamaJni::prefill_images),
        makeNativeMethod(
            "prefillPromptNative", ExecuTorchLlamaJni::prefill_prompt),
        makeNativeMethod(
            "generateFromPos", ExecuTorchLlamaJni::generate_from_pos),
    });
  }
};

} // namespace executorch_jni

void register_natives_for_llama() {
  executorch_jni::ExecuTorchLlamaJni::registerNatives();
}
