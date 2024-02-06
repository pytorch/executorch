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

#include "jni_layer_constants.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/managed_tensor.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

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

  __android_log_print(android_log_level, "ExecuTorch", "%s", message);
}
#endif

using namespace torch::executor;

namespace executorch_jni {
class TensorHybrid : public facebook::jni::HybridClass<TensorHybrid> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/Tensor;";

  explicit TensorHybrid(exec_aten::Tensor tensor) {}

  static facebook::jni::local_ref<TensorHybrid::javaobject>
  newJTensorFromTensor(const exec_aten::Tensor& tensor) {
    // Java wrapper currently only supports contiguous tensors.

    const auto scalarType = tensor.scalar_type();

    if (scalar_type_to_java_dtype.count(scalarType) == 0) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "exec_aten::Tensor scalar type %d is not supported on java side",
          scalarType);
    }
    int jdtype = scalar_type_to_java_dtype.at(scalarType);

    const auto& tensor_shape = tensor.sizes();
    std::vector<jlong> tensor_shape_vec;
    for (const auto& s : tensor_shape) {
      tensor_shape_vec.push_back(s);
    }
    facebook::jni::local_ref<jlongArray> jTensorShape =
        facebook::jni::make_long_array(tensor_shape_vec.size());
    jTensorShape->setRegion(
        0, tensor_shape_vec.size(), tensor_shape_vec.data());

    static auto cls = TensorHybrid::javaClassStatic();
    // Note: this is safe as long as the data stored in tensor is valid; the
    // data won't go out of scope as long as the Method for the inference is
    // valid and there is no other inference call. Java layer picks up this
    // value immediately so the data is valid.
    facebook::jni::local_ref<facebook::jni::JByteBuffer> jTensorBuffer =
        facebook::jni::JByteBuffer::wrapBytes(
            (uint8_t*)tensor.data_ptr(), tensor.nbytes());
    jTensorBuffer->order(facebook::jni::JByteOrder::nativeOrder());

    static const auto jMethodNewTensor =
        cls->getStaticMethod<facebook::jni::local_ref<TensorHybrid::javaobject>(
            facebook::jni::alias_ref<facebook::jni::JByteBuffer>,
            facebook::jni::alias_ref<jlongArray>,
            jint,
            facebook::jni::alias_ref<jhybriddata>)>("nativeNewTensor");
    return jMethodNewTensor(
        cls, jTensorBuffer, jTensorShape, jdtype, makeCxxInstance(tensor));
  }

 private:
  friend HybridBase;
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

  static facebook::jni::local_ref<JEValue> newJEValueFromEValue(EValue evalue) {
    if (evalue.isTensor()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                  facebook::jni::local_ref<TensorHybrid::javaobject>)>("from");
      return jMethodTensor(
          JEValue::javaClassStatic(),
          TensorHybrid::newJTensorFromTensor(evalue.toTensor()));
    } else if (evalue.isInt()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(jlong)>(
                  "from");
      return jMethodTensor(JEValue::javaClassStatic(), evalue.toInt());
    } else if (evalue.isDouble()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(jdouble)>(
                  "from");
      return jMethodTensor(JEValue::javaClassStatic(), evalue.toDouble());
    } else if (evalue.isBool()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(jboolean)>(
                  "from");
      return jMethodTensor(JEValue::javaClassStatic(), evalue.toBool());
    } else if (evalue.isString()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                  facebook::jni::local_ref<jstring>)>("from");
      std::string str =
          std::string(evalue.toString().begin(), evalue.toString().end());
      return jMethodTensor(
          JEValue::javaClassStatic(), facebook::jni::make_jstring(str));
    }
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Unsupported EValue type: %d",
        evalue.tag);
  }

  static ManagedTensor JEValueToTensorImpl(
      facebook::jni::alias_ref<JEValue> JEValue) {
    static const auto typeCodeField =
        JEValue::javaClassStatic()->getField<jint>("mTypeCode");
    const auto typeCode = JEValue->getFieldValue(typeCodeField);
    if (JEValue::kTypeCodeTensor == typeCode) {
      static const auto jMethodGetTensor =
          JEValue::javaClassStatic()
              ->getMethod<facebook::jni::alias_ref<TensorHybrid::javaobject>()>(
                  "toTensor");
      auto jtensor = jMethodGetTensor(JEValue);

      static auto cls = TensorHybrid::javaClassStatic();
      static const auto dtypeMethod = cls->getMethod<jint()>("dtypeJniCode");
      jint jdtype = dtypeMethod(jtensor);

      static const auto shapeField = cls->getField<jlongArray>("shape");
      auto jshape = jtensor->getFieldValue(shapeField);

      static auto dataBufferMethod = cls->getMethod<
          facebook::jni::local_ref<facebook::jni::JBuffer::javaobject>()>(
          "getRawDataBuffer");
      facebook::jni::local_ref<facebook::jni::JBuffer> jbuffer =
          dataBufferMethod(jtensor);

      const auto rank = jshape->size();

      const auto shapeArr = jshape->getRegion(0, rank);
      std::vector<exec_aten::SizesType> shape_vec;
      shape_vec.reserve(rank);

      auto numel = 1;
      for (int i = 0; i < rank; i++) {
        shape_vec.push_back(shapeArr[i]);
      }
      for (int i = rank - 1; i >= 0; --i) {
        numel *= shapeArr[i];
      }
      JNIEnv* jni = facebook::jni::Environment::current();
      if (java_dtype_to_scalar_type.count(jdtype) == 0) {
        facebook::jni::throwNewJavaException(
            facebook::jni::gJavaLangIllegalArgumentException,
            "Unknown Tensor jdtype %d",
            jdtype);
      }
      ScalarType scalar_type = java_dtype_to_scalar_type.at(jdtype);
      const auto dataCapacity = jni->GetDirectBufferCapacity(jbuffer.get());
      if (dataCapacity != numel) {
        facebook::jni::throwNewJavaException(
            facebook::jni::gJavaLangIllegalArgumentException,
            "Tensor dimensions(elements number:%d inconsistent with buffer capacity(%d)",
            numel,
            dataCapacity);
      }
      return ManagedTensor(
          jni->GetDirectBufferAddress(jbuffer.get()),
          numel,
          shape_vec,
          scalar_type);
    }
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Unknown EValue typeCode %d",
        typeCode);
  }
};

class ExecuTorchJni : public facebook::jni::HybridClass<ExecuTorchJni> {
 private:
  friend HybridBase;
  std::unique_ptr<torch::executor::Module> module_;

 public:
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/executorch/NativePeer;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
          extraFiles) {
    return makeCxxInstance(modelPath, extraFiles);
  }

  ExecuTorchJni(
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
          extraFiles) {
    module_ = std::make_unique<torch::executor::Module>(
        modelPath->toStdString(),
        torch::executor::Module::MlockConfig::NoMlock);
  }

  facebook::jni::local_ref<JEValue> forward(
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
          jinputs) {
    return execute_method("forward", jinputs);
  }

  facebook::jni::local_ref<JEValue> execute(
      facebook::jni::alias_ref<jstring> methodName,
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
          jinputs) {
    return execute_method(methodName->toStdString(), jinputs);
  }

  facebook::jni::local_ref<JEValue> execute_method(
      std::string method,
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
          jinputs) {
    std::vector<EValue> evalues = {};

    std::vector<ManagedTensor> managed_tensors = {};

    static const auto typeCodeField =
        JEValue::javaClassStatic()->getField<jint>("mTypeCode");

    for (int i = 0; i < jinputs->size(); i++) {
      auto jevalue = jinputs->getElement(i);
      const auto typeCode = jevalue->getFieldValue(typeCodeField);
      if (typeCode == JEValue::kTypeCodeTensor) {
        managed_tensors.emplace_back(JEValue::JEValueToTensorImpl(jevalue));
        evalues.emplace_back(
            EValue(managed_tensors.back().get_aliasing_tensor()));
      } else if (typeCode == JEValue::kTypeCodeInt) {
        int64_t value = jevalue->getFieldValue(typeCodeField);
        evalues.emplace_back(EValue(value));
      } else if (typeCode == JEValue::kTypeCodeDouble) {
        double value = jevalue->getFieldValue(typeCodeField);
        evalues.emplace_back(EValue(value));
      } else if (typeCode == JEValue::kTypeCodeBool) {
        bool value = jevalue->getFieldValue(typeCodeField);
        evalues.emplace_back(EValue(value));
      }
    }

#ifdef EXECUTORCH_ANDROID_PROFILING
    auto start = std::chrono::high_resolution_clock::now();
    auto result = module_->execute(method, evalues);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    ET_LOG(Debug, "Execution time: %lld ms.", duration);

#else
    auto result = module_->execute(method, evalues);

#endif

    ET_CHECK_MSG(
        result.ok(),
        "Execution of method %s failed with status 0x%" PRIx32,
        method.c_str(),
        static_cast<error_code_t>(result.error()));
    ET_LOG(Info, "Model executed successfully.");

    return JEValue::newJEValueFromEValue(result.get()[0]);
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchJni::initHybrid),
        makeNativeMethod("forward", ExecuTorchJni::forward),
        makeNativeMethod("execute", ExecuTorchJni::execute),
    });
  }
};

} // namespace executorch_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { executorch_jni::ExecuTorchJni::registerNatives(); });
}
