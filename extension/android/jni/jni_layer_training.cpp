/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/android/jni/jni_layer_constants.h>
#include <executorch/extension/android/jni/log.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/extension/training/optimizer/sgd.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/log.h>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

using namespace executorch::extension;
using namespace executorch::extension::training;
using namespace torch::executor;

namespace executorch::extension {

// Forward declarations from jni_layer.cpp
class TensorHybrid : public facebook::jni::HybridClass<TensorHybrid> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/Tensor;";

  static facebook::jni::local_ref<TensorHybrid::javaobject>
  newJTensorFromTensor(const executorch::aten::Tensor& tensor);

  static TensorPtr newTensorFromJTensor(
      facebook::jni::alias_ref<TensorHybrid::javaobject> jtensor);
};

class JEValue : public facebook::jni::JavaClass<JEValue> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/EValue;";

  constexpr static int kTypeCodeTensor = 1;
  constexpr static int kTypeCodeString = 2;
  constexpr static int kTypeCodeDouble = 3;
  constexpr static int kTypeCodeInt = 4;
  constexpr static int kTypeCodeBool = 5;

  static facebook::jni::local_ref<JEValue> newJEValueFromEValue(
      runtime::EValue evalue);

  static TensorPtr JEValueToTensorImpl(
      facebook::jni::alias_ref<JEValue> JEValue);
};

class ExecuTorchTrainingJni
    : public facebook::jni::HybridClass<ExecuTorchTrainingJni> {
 private:
  friend HybridBase;
  std::unique_ptr<training::TrainingModule> module_;

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/executorch/TrainingModule;";

  ExecuTorchTrainingJni(
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<jstring> dataPath) {
    auto modelLoader = FileDataLoader::from(modelPath->toStdString().c_str());
    auto dataLoader = FileDataLoader::from(dataPath->toStdString().c_str());
    module_ = std::make_unique<training::TrainingModule>(
        std::make_unique<FileDataLoader>(std::move(modelLoader.get())),
        nullptr,
        nullptr,
        nullptr,
        std::make_unique<FileDataLoader>(std::move(dataLoader.get())));
  }

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<jstring> dataPath) {
    return makeCxxInstance(modelPath, dataPath);
  }

  facebook::jni::local_ref<facebook::jni::JArrayClass<JEValue>>
  executeForwardBackward(
      facebook::jni::alias_ref<jstring> methodName,
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
          jinputs) {
    std::vector<runtime::EValue> evalues;
    std::vector<TensorPtr> tensors;

    static const auto typeCodeField =
        JEValue::javaClassStatic()->getField<jint>("mTypeCode");

    for (int i = 0; i < jinputs->size(); i++) {
      auto jevalue = jinputs->getElement(i);
      const auto typeCode = jevalue->getFieldValue(typeCodeField);
      if (typeCode == JEValue::kTypeCodeTensor) {
        tensors.emplace_back(JEValue::JEValueToTensorImpl(jevalue));
        evalues.emplace_back(tensors.back());
      } else if (typeCode == JEValue::kTypeCodeInt) {
        int64_t value = jevalue->getFieldValue(typeCodeField);
        evalues.emplace_back(value);
      } else if (typeCode == JEValue::kTypeCodeDouble) {
        double value = jevalue->getFieldValue(typeCodeField);
        evalues.emplace_back(value);
      } else if (typeCode == JEValue::kTypeCodeBool) {
        bool value = jevalue->getFieldValue(typeCodeField);
        evalues.emplace_back(value);
      }
    }

    auto result =
        module_->execute_forward_backward(methodName->toStdString(), evalues);
    if (!result.ok()) {
      facebook::jni::throwNewJavaException(
          "java/lang/Exception",
          "Execution of forward_backward for method %s failed with status 0x%" PRIx32,
          methodName->toStdString().c_str(),
          static_cast<error_code_t>(result.error()));
      return {};
    }

    facebook::jni::local_ref<facebook::jni::JArrayClass<JEValue>> jresult =
        facebook::jni::JArrayClass<JEValue>::newArray(result.get().size());

    for (int i = 0; i < result.get().size(); i++) {
      auto jevalue = JEValue::newJEValueFromEValue(result.get()[i]);
      jresult->setElement(i, *jevalue);
    }
    return jresult;
  }

  facebook::jni::local_ref<
      facebook::jni::JMap<jstring, TensorHybrid::javaobject>>
  namedParameters(facebook::jni::alias_ref<jstring> methodName) {
    auto method = methodName->toStdString();
    auto result = module_->named_parameters(method);
    if (!result.ok()) {
      facebook::jni::throwNewJavaException(
          "java/lang/Exception",
          "Getting named parameters for method %s failed with status 0x%" PRIx32,
          method.c_str(),
          static_cast<error_code_t>(result.error()));
      return {};
    }
    facebook::jni::local_ref<
        facebook::jni::JHashMap<jstring, TensorHybrid::javaobject>>
        parameters = facebook::jni::
            JHashMap<jstring, TensorHybrid::javaobject>::create();
    for (auto& [layer, tensor] : result.get()) {
      parameters->put(
          facebook::jni::make_jstring(layer.data()),
          TensorHybrid::newJTensorFromTensor(tensor));
    }
    return parameters;
  }

  facebook::jni::local_ref<
      facebook::jni::JMap<jstring, TensorHybrid::javaobject>>
  namedGradients(facebook::jni::alias_ref<jstring> methodName) {
    auto method = methodName->toStdString();
    auto result = module_->named_gradients(method);
    if (!result.ok()) {
      facebook::jni::throwNewJavaException(
          "java/lang/Exception",
          "Getting named gradients for method %s failed with status 0x%" PRIx32,
          method.c_str(),
          static_cast<error_code_t>(result.error()));
      return {};
    }
    facebook::jni::local_ref<
        facebook::jni::JHashMap<jstring, TensorHybrid::javaobject>>
        gradients = facebook::jni::JHashMap<jstring, TensorHybrid::javaobject>::
            create();
    for (auto& [layer, tensor] : result.get()) {
      gradients->put(
          facebook::jni::make_jstring(layer.data()),
          TensorHybrid::newJTensorFromTensor(tensor));
    }
    return gradients;
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchTrainingJni::initHybrid),
        makeNativeMethod(
            "executeForwardBackwardNative",
            ExecuTorchTrainingJni::executeForwardBackward),
        makeNativeMethod(
            "namedParametersNative", ExecuTorchTrainingJni::namedParameters),
        makeNativeMethod(
            "namedGradientsNative", ExecuTorchTrainingJni::namedGradients),
    });
  }
};

class SGDHybrid : public facebook::jni::HybridClass<SGDHybrid> {
 public:
  constexpr static const char* kJavaDescriptor = "Lorg/pytorch/executorch/SGD;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<
          facebook::jni::JMap<jstring, TensorHybrid::javaobject>>
          namedParameters,
      jdouble learningRate,
      jdouble momentum,
      jdouble dampening,
      jdouble weightDecay,
      jboolean nesterov) {
    return makeCxxInstance(
        namedParameters,
        learningRate,
        momentum,
        dampening,
        weightDecay,
        nesterov);
  }

  SGDHybrid(
      facebook::jni::alias_ref<
          facebook::jni::JMap<jstring, TensorHybrid::javaobject>>
          namedParameters,
      jdouble learningRate,
      jdouble momentum,
      jdouble dampening,
      jdouble weightDecay,
      jboolean nesterov) {
    // Convert Java Map to C++ map
    std::map<std::string_view, executorch::aten::Tensor> cpp_named_parameters;

    // Avoid vector reallocation to keep string_views valid.
    parameter_names_.reserve(namedParameters->size());
    tensor_ptrs_.reserve(namedParameters->size());

    // Iterate through the map using JNI methods
    auto iterator = namedParameters->begin();
    auto end = namedParameters->end();

    while (iterator != end) {
      auto key = iterator->first;
      auto value = iterator->second;

      std::string param_name = key->toStdString();
      TensorPtr tensor = TensorHybrid::newTensorFromJTensor(value);

      // Store the parameter name and tensor
      parameter_names_.push_back(param_name);
      tensor_ptrs_.push_back(tensor);
      cpp_named_parameters.emplace(
          std::string_view(parameter_names_.back()), *tensor);

      ++iterator;
    }

    // Create SGD options
    optimizer::SGDOptions options(
        learningRate, momentum, dampening, weightDecay, nesterov);

    // Create the SGD optimizer
    sgd_optimizer_ =
        std::make_unique<optimizer::SGD>(cpp_named_parameters, options);
  }

  void
  step(facebook::jni::alias_ref<
       facebook::jni::JMap<jstring, TensorHybrid::javaobject>> namedGradients) {
    // Convert Java Map to C++ map
    std::map<std::string_view, executorch::aten::Tensor> cpp_named_gradients;

    // Iterate through the map using JNI methods
    auto iterator = namedGradients->begin();
    auto end = namedGradients->end();

    std::vector<std::string> gradient_names;
    std::vector<TensorPtr> tensor_keepalives;

    gradient_names.reserve(namedGradients->size());
    tensor_keepalives.reserve(namedGradients->size());

    while (iterator != end) {
      auto key = iterator->first;
      auto value = iterator->second;

      std::string grad_name = key->toStdString();
      TensorPtr tensor = TensorHybrid::newTensorFromJTensor(value);

      // Store the gradient name and tensor
      gradient_names.push_back(grad_name);
      tensor_keepalives.push_back(tensor);
      cpp_named_gradients.emplace(
          std::string_view(gradient_names.back()), *tensor);

      ++iterator;
    }

    // Perform the optimization step
    auto result = sgd_optimizer_->step(cpp_named_gradients);
    if (result != ::executorch::runtime::Error::Ok) {
      facebook::jni::throwNewJavaException(
          "java/lang/Exception",
          "SGD optimization step failed with status 0x%" PRIx32,
          static_cast<error_code_t>(result));
    }
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", SGDHybrid::initHybrid),
        makeNativeMethod("stepNative", SGDHybrid::step),
    });
  }

 private:
  friend HybridBase;
  std::unique_ptr<optimizer::SGD> sgd_optimizer_;
  std::vector<std::string>
      parameter_names_; // Store parameter names to keep string_view valid
  std::vector<TensorPtr>
      tensor_ptrs_; // Store tensors to keep TensorPtrs valid.
};

} // namespace executorch::extension

// Function to register training module natives
void register_natives_for_training() {
  executorch::extension::ExecuTorchTrainingJni::registerNatives();
  executorch::extension::SGDHybrid::registerNatives();
};
