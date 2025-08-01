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
      "Lorg/pytorch/executorch/training/TrainingModule;";

  ExecuTorchTrainingJni(
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<jstring> dataPath) {
    auto modelPathString = modelPath->toStdString();
    auto modelLoaderRes = FileDataLoader::from(modelPathString.c_str());
    if (modelLoaderRes.error() != Error::Ok) {
      facebook::jni::throwNewJavaException(
          "java/lang/Exception",
          "Failed to open model file: %s",
          modelPathString.c_str());
    }
    auto modelLoader =
        std::make_unique<FileDataLoader>(std::move(modelLoaderRes.get()));

    std::unique_ptr<FileDataLoader> dataLoader = nullptr;
    auto dataPathString = dataPath->toStdString();
    if (!dataPathString.empty()) {
      auto dataLoaderRes = FileDataLoader::from(dataPathString.c_str());
      if (dataLoaderRes.error() != Error::Ok) {
        facebook::jni::throwNewJavaException(
            "java/lang/Exception",
            "Failed to open ptd file: %s",
            dataPathString.c_str());
      }
      dataLoader =
          std::make_unique<FileDataLoader>(std::move(dataLoaderRes.get()));
    }

    module_ = std::make_unique<training::TrainingModule>(
        std::move(modelLoader),
        nullptr,
        nullptr,
        nullptr,
        std::move(dataLoader));
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
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/training/SGD;";

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
    std::map<std::string_view, executorch::aten::Tensor> cppNamedParameters;

    // Avoid vector reallocation to keep string_views valid.
    parameterNames_.reserve(namedParameters->size());
    paramTensorPtrs_.reserve(namedParameters->size());

    auto iterator = namedParameters->begin();
    auto end = namedParameters->end();

    while (iterator != end) {
      auto key = iterator->first;
      auto value = iterator->second;

      std::string paramName = key->toStdString();
      TensorPtr tensor = TensorHybrid::newTensorFromJTensor(value);

      // Store the parameter name and tensor
      parameterNames_.push_back(paramName);
      paramTensorPtrs_.push_back(tensor);
      cppNamedParameters.emplace(
          std::string_view(parameterNames_.back()), *tensor);

      ++iterator;
    }

    optimizer::SGDOptions options(
        learningRate, momentum, dampening, weightDecay, nesterov);
    sgdOptimizer_ =
        std::make_unique<optimizer::SGD>(cppNamedParameters, options);
  }

  void
  step(facebook::jni::alias_ref<
       facebook::jni::JMap<jstring, TensorHybrid::javaobject>> namedGradients) {
    std::map<std::string_view, executorch::aten::Tensor> cppNamedGradients;
    std::vector<std::string> gradientNames;
    std::vector<TensorPtr> tensorKeepalives;

    gradientNames.reserve(namedGradients->size());
    tensorKeepalives.reserve(namedGradients->size());

    auto iterator = namedGradients->begin();
    auto end = namedGradients->end();

    while (iterator != end) {
      auto key = iterator->first;
      auto value = iterator->second;

      std::string gradName = key->toStdString();
      TensorPtr tensor = TensorHybrid::newTensorFromJTensor(value);

      // Store the gradient name and tensor
      gradientNames.push_back(gradName);
      tensorKeepalives.push_back(tensor);
      cppNamedGradients.emplace(
          std::string_view(gradientNames.back()), *tensor);

      ++iterator;
    }

    auto result = sgdOptimizer_->step(cppNamedGradients);
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
  std::unique_ptr<optimizer::SGD> sgdOptimizer_;
  std::vector<std::string>
      parameterNames_; // Store parameter names to keep string_view valid
  std::vector<TensorPtr>
      paramTensorPtrs_; // Store parameter tensors to keep TensorPtrs valid.
};

} // namespace executorch::extension

// Function to register training module natives
void register_natives_for_training() {
  executorch::extension::ExecuTorchTrainingJni::registerNatives();
  executorch::extension::SGDHybrid::registerNatives();
};
