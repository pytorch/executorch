/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <jni.h>

#include <executorch/extension/android/jni/jni_helper.h>
#include <executorch/extension/android/jni/jni_layer_constants.h>
#include <executorch/extension/android/jni/log.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/extension/training/optimizer/sgd.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/log.h>
#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace executorch::extension;
using namespace executorch::extension::training;
using namespace torch::executor;

namespace {

// Helper to convert jstring to std::string
static std::string jstring_to_string(JNIEnv* env, jstring jstr) {
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

// EValue type codes (must match Java EValue class)
constexpr int kTypeCodeNone = 0;
constexpr int kTypeCodeTensor = 1;
constexpr int kTypeCodeString = 2;
constexpr int kTypeCodeDouble = 3;
constexpr int kTypeCodeInt = 4;
constexpr int kTypeCodeBool = 5;

// Cached class and method IDs for performance
struct TrainingJniCache {
  jclass tensor_class = nullptr;
  jclass evalue_class = nullptr;
  jclass hashmap_class = nullptr;
  jclass iterator_class = nullptr;
  jclass map_entry_class = nullptr;
  
  jmethodID tensor_nativeNewTensor = nullptr;
  jmethodID tensor_dtypeJniCode = nullptr;
  jmethodID tensor_getRawDataBuffer = nullptr;
  jfieldID tensor_shape = nullptr;
  
  jmethodID evalue_from_tensor = nullptr;
  jmethodID evalue_from_long = nullptr;
  jmethodID evalue_from_double = nullptr;
  jmethodID evalue_from_bool = nullptr;
  jmethodID evalue_from_string = nullptr;
  jmethodID evalue_toTensor = nullptr;
  jfieldID evalue_mTypeCode = nullptr;
  jfieldID evalue_mData = nullptr;
  
  jmethodID hashmap_init = nullptr;
  jmethodID hashmap_put = nullptr;
  jmethodID hashmap_size = nullptr;
  jmethodID hashmap_entrySet = nullptr;
  jmethodID set_iterator = nullptr;
  jmethodID iterator_hasNext = nullptr;
  jmethodID iterator_next = nullptr;
  jmethodID entry_getKey = nullptr;
  jmethodID entry_getValue = nullptr;

  bool initialized = false;

  void init(JNIEnv* env) {
    if (initialized) {
      return;
    }

    // Cache Tensor class and methods
    jclass local_tensor_class = env->FindClass("org/pytorch/executorch/Tensor");
    if (local_tensor_class != nullptr) {
      tensor_class = static_cast<jclass>(env->NewGlobalRef(local_tensor_class));
      env->DeleteLocalRef(local_tensor_class);

      tensor_nativeNewTensor = env->GetStaticMethodID(
          tensor_class,
          "nativeNewTensor",
          "(Ljava/nio/ByteBuffer;[JIJ)Lorg/pytorch/executorch/Tensor;");
      tensor_dtypeJniCode = env->GetMethodID(tensor_class, "dtypeJniCode", "()I");
      tensor_getRawDataBuffer =
          env->GetMethodID(tensor_class, "getRawDataBuffer", "()Ljava/nio/Buffer;");
      tensor_shape = env->GetFieldID(tensor_class, "shape", "[J");
    }

    // Cache EValue class and methods
    jclass local_evalue_class = env->FindClass("org/pytorch/executorch/EValue");
    if (local_evalue_class != nullptr) {
      evalue_class = static_cast<jclass>(env->NewGlobalRef(local_evalue_class));
      env->DeleteLocalRef(local_evalue_class);

      evalue_from_tensor = env->GetStaticMethodID(
          evalue_class,
          "from",
          "(Lorg/pytorch/executorch/Tensor;)Lorg/pytorch/executorch/EValue;");
      evalue_from_long =
          env->GetStaticMethodID(evalue_class, "from", "(J)Lorg/pytorch/executorch/EValue;");
      evalue_from_double =
          env->GetStaticMethodID(evalue_class, "from", "(D)Lorg/pytorch/executorch/EValue;");
      evalue_from_bool =
          env->GetStaticMethodID(evalue_class, "from", "(Z)Lorg/pytorch/executorch/EValue;");
      evalue_from_string = env->GetStaticMethodID(
          evalue_class,
          "from",
          "(Ljava/lang/String;)Lorg/pytorch/executorch/EValue;");
      evalue_toTensor = env->GetMethodID(
          evalue_class, "toTensor", "()Lorg/pytorch/executorch/Tensor;");
      evalue_mTypeCode = env->GetFieldID(evalue_class, "mTypeCode", "I");
      evalue_mData = env->GetFieldID(evalue_class, "mData", "Ljava/lang/Object;");
    }

    // Cache HashMap class and methods
    jclass local_hashmap_class = env->FindClass("java/util/HashMap");
    if (local_hashmap_class != nullptr) {
      hashmap_class = static_cast<jclass>(env->NewGlobalRef(local_hashmap_class));
      env->DeleteLocalRef(local_hashmap_class);

      hashmap_init = env->GetMethodID(hashmap_class, "<init>", "()V");
      hashmap_put = env->GetMethodID(
          hashmap_class,
          "put",
          "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
      hashmap_size = env->GetMethodID(hashmap_class, "size", "()I");
      hashmap_entrySet = env->GetMethodID(hashmap_class, "entrySet", "()Ljava/util/Set;");
    }

    // Cache Set/Iterator/Entry classes and methods
    jclass local_set_class = env->FindClass("java/util/Set");
    if (local_set_class != nullptr) {
      set_iterator = env->GetMethodID(local_set_class, "iterator", "()Ljava/util/Iterator;");
      env->DeleteLocalRef(local_set_class);
    }

    jclass local_iterator_class = env->FindClass("java/util/Iterator");
    if (local_iterator_class != nullptr) {
      iterator_class = static_cast<jclass>(env->NewGlobalRef(local_iterator_class));
      env->DeleteLocalRef(local_iterator_class);

      iterator_hasNext = env->GetMethodID(iterator_class, "hasNext", "()Z");
      iterator_next = env->GetMethodID(iterator_class, "next", "()Ljava/lang/Object;");
    }

    jclass local_entry_class = env->FindClass("java/util/Map$Entry");
    if (local_entry_class != nullptr) {
      map_entry_class = static_cast<jclass>(env->NewGlobalRef(local_entry_class));
      env->DeleteLocalRef(local_entry_class);

      entry_getKey = env->GetMethodID(map_entry_class, "getKey", "()Ljava/lang/Object;");
      entry_getValue = env->GetMethodID(map_entry_class, "getValue", "()Ljava/lang/Object;");
    }

    initialized = true;
  }
};

TrainingJniCache g_training_cache;

// Helper to create Java Tensor from C++ tensor
jobject newJTensorFromTensor(JNIEnv* env, const executorch::aten::Tensor& tensor) {
  g_training_cache.init(env);

  const auto scalarType = tensor.scalar_type();
  if (scalar_type_to_java_dtype.count(scalarType) == 0) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        "executorch::aten::Tensor scalar type not supported on java side");
    return nullptr;
  }
  int jdtype = scalar_type_to_java_dtype.at(scalarType);

  const auto& tensor_shape = tensor.sizes();
  jlongArray jTensorShape = env->NewLongArray(tensor_shape.size());
  if (jTensorShape == nullptr) {
    return nullptr;
  }
  
  std::vector<jlong> shape_vec(tensor_shape.begin(), tensor_shape.end());
  env->SetLongArrayRegion(jTensorShape, 0, shape_vec.size(), shape_vec.data());

  jobject jTensorBuffer = env->NewDirectByteBuffer(
      const_cast<void*>(tensor.const_data_ptr()), tensor.nbytes());
  if (jTensorBuffer == nullptr) {
    env->DeleteLocalRef(jTensorShape);
    return nullptr;
  }

  jobject jTensor = env->CallStaticObjectMethod(
      g_training_cache.tensor_class,
      g_training_cache.tensor_nativeNewTensor,
      jTensorBuffer,
      jTensorShape,
      jdtype,
      (jlong)0);

  env->DeleteLocalRef(jTensorShape);
  env->DeleteLocalRef(jTensorBuffer);

  return jTensor;
}

// Helper to create C++ TensorPtr from Java Tensor
TensorPtr newTensorFromJTensor(JNIEnv* env, jobject jtensor) {
  g_training_cache.init(env);

  jint jdtype = env->CallIntMethod(jtensor, g_training_cache.tensor_dtypeJniCode);
  
  jlongArray jshape = static_cast<jlongArray>(
      env->GetObjectField(jtensor, g_training_cache.tensor_shape));
  
  jobject jbuffer = env->CallObjectMethod(jtensor, g_training_cache.tensor_getRawDataBuffer);

  const auto rank = env->GetArrayLength(jshape);
  std::vector<jlong> shapeArr(rank);
  env->GetLongArrayRegion(jshape, 0, rank, shapeArr.data());

  std::vector<executorch::aten::SizesType> sizes_vec;
  sizes_vec.reserve(rank);
  for (int i = 0; i < rank; i++) {
    sizes_vec.push_back(shapeArr[i]);
  }

  void* dataPtr = env->GetDirectBufferAddress(jbuffer);
  if (java_dtype_to_scalar_type.count(jdtype) == 0) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        "Unknown Tensor jdtype");
    return nullptr;
  }

  ScalarType scalarType = java_dtype_to_scalar_type.at(jdtype);
  return from_blob(dataPtr, sizes_vec, scalarType);
}

// Helper to create Java EValue from C++ EValue
jobject newJEValueFromEValue(JNIEnv* env, runtime::EValue evalue) {
  g_training_cache.init(env);

  if (evalue.isTensor()) {
    jobject jTensor = newJTensorFromTensor(env, evalue.toTensor());
    if (jTensor == nullptr) {
      return nullptr;
    }
    jobject jEValue = env->CallStaticObjectMethod(
        g_training_cache.evalue_class,
        g_training_cache.evalue_from_tensor,
        jTensor);
    env->DeleteLocalRef(jTensor);
    return jEValue;
  } else if (evalue.isInt()) {
    return env->CallStaticObjectMethod(
        g_training_cache.evalue_class,
        g_training_cache.evalue_from_long,
        (jlong)evalue.toInt());
  } else if (evalue.isDouble()) {
    return env->CallStaticObjectMethod(
        g_training_cache.evalue_class,
        g_training_cache.evalue_from_double,
        (jdouble)evalue.toDouble());
  } else if (evalue.isBool()) {
    return env->CallStaticObjectMethod(
        g_training_cache.evalue_class,
        g_training_cache.evalue_from_bool,
        (jboolean)evalue.toBool());
  } else if (evalue.isString()) {
    std::string str = std::string(evalue.toString().begin(), evalue.toString().end());
    jstring jStr = env->NewStringUTF(str.c_str());
    jobject jEValue = env->CallStaticObjectMethod(
        g_training_cache.evalue_class,
        g_training_cache.evalue_from_string,
        jStr);
    env->DeleteLocalRef(jStr);
    return jEValue;
  }
  
  jni_helper::throwExecutorchException(
      env,
      static_cast<uint32_t>(Error::InvalidArgument),
      "Unknown EValue type");
  return nullptr;
}

// Helper to extract TensorPtr from Java EValue
TensorPtr JEValueToTensorImpl(JNIEnv* env, jobject jevalue) {
  g_training_cache.init(env);

  jint typeCode = env->GetIntField(jevalue, g_training_cache.evalue_mTypeCode);
  if (typeCode == kTypeCodeTensor) {
    jobject jtensor = env->CallObjectMethod(jevalue, g_training_cache.evalue_toTensor);
    if (jtensor == nullptr) {
      return nullptr;
    }
    TensorPtr tensor = newTensorFromJTensor(env, jtensor);
    env->DeleteLocalRef(jtensor);
    return tensor;
  }
  
  jni_helper::throwExecutorchException(
      env,
      static_cast<uint32_t>(Error::InvalidArgument),
      "EValue is not a tensor");
  return nullptr;
}

} // namespace

extern "C" {

// TrainingModule native methods
JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_training_TrainingModule_initHybrid(
    JNIEnv* env,
    jclass /* clazz */,
    jstring modelPath,
    jstring dataPath) {
  
  std::string modelPathString = jstring_to_string(env, modelPath);
  auto modelLoaderRes = FileDataLoader::from(modelPathString.c_str());
  if (modelLoaderRes.error() != Error::Ok) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(modelLoaderRes.error()),
        "Failed to open model file: " + modelPathString);
    return 0;
  }
  auto modelLoader =
      std::make_unique<FileDataLoader>(std::move(modelLoaderRes.get()));

  std::unique_ptr<FileDataLoader> dataLoader = nullptr;
  std::string dataPathString = jstring_to_string(env, dataPath);
  if (!dataPathString.empty()) {
    auto dataLoaderRes = FileDataLoader::from(dataPathString.c_str());
    if (dataLoaderRes.error() != Error::Ok) {
      jni_helper::throwExecutorchException(
          env,
          static_cast<uint32_t>(dataLoaderRes.error()),
          "Failed to open ptd file: " + dataPathString);
      return 0;
    }
    dataLoader =
        std::make_unique<FileDataLoader>(std::move(dataLoaderRes.get()));
  }

  auto* module = new training::TrainingModule(
      std::move(modelLoader),
      nullptr,
      nullptr,
      nullptr,
      std::move(dataLoader));
  
  return reinterpret_cast<jlong>(module);
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_training_TrainingModule_nativeDestroy(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jlong nativeHandle) {
  if (nativeHandle != 0) {
    auto* module = reinterpret_cast<training::TrainingModule*>(nativeHandle);
    delete module;
  }
}

JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_training_TrainingModule_executeForwardBackwardNative(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jstring methodName,
    jobjectArray jinputs) {
  
  g_training_cache.init(env);
  
  auto* module = reinterpret_cast<training::TrainingModule*>(nativeHandle);
  if (module == nullptr) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        "Invalid native handle");
    return nullptr;
  }

  std::vector<runtime::EValue> evalues;
  std::vector<TensorPtr> tensors;

  jsize inputCount = env->GetArrayLength(jinputs);
  for (jsize i = 0; i < inputCount; i++) {
    jobject jevalue = env->GetObjectArrayElement(jinputs, i);
    if (jevalue == nullptr) {
      continue;
    }
    
    jint typeCode = env->GetIntField(jevalue, g_training_cache.evalue_mTypeCode);
    if (typeCode == kTypeCodeTensor) {
      tensors.emplace_back(JEValueToTensorImpl(env, jevalue));
      if (tensors.back() == nullptr) {
        env->DeleteLocalRef(jevalue);
        return nullptr;
      }
      evalues.emplace_back(tensors.back());
    } else if (typeCode == kTypeCodeInt) {
      jobject data = env->GetObjectField(jevalue, g_training_cache.evalue_mData);
      jclass longClass = env->FindClass("java/lang/Long");
      jmethodID longValue = env->GetMethodID(longClass, "longValue", "()J");
      jlong value = env->CallLongMethod(data, longValue);
      evalues.emplace_back(static_cast<int64_t>(value));
      env->DeleteLocalRef(longClass);
      env->DeleteLocalRef(data);
    } else if (typeCode == kTypeCodeDouble) {
      jobject data = env->GetObjectField(jevalue, g_training_cache.evalue_mData);
      jclass doubleClass = env->FindClass("java/lang/Double");
      jmethodID doubleValue = env->GetMethodID(doubleClass, "doubleValue", "()D");
      jdouble value = env->CallDoubleMethod(data, doubleValue);
      evalues.emplace_back(static_cast<double>(value));
      env->DeleteLocalRef(doubleClass);
      env->DeleteLocalRef(data);
    } else if (typeCode == kTypeCodeBool) {
      jobject data = env->GetObjectField(jevalue, g_training_cache.evalue_mData);
      jclass boolClass = env->FindClass("java/lang/Boolean");
      jmethodID boolValue = env->GetMethodID(boolClass, "booleanValue", "()Z");
      jboolean value = env->CallBooleanMethod(data, boolValue);
      evalues.emplace_back(static_cast<bool>(value));
      env->DeleteLocalRef(boolClass);
      env->DeleteLocalRef(data);
    }
    env->DeleteLocalRef(jevalue);
  }

  std::string method = jstring_to_string(env, methodName);
  auto result = module->execute_forward_backward(method, evalues);
  if (!result.ok()) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(result.error()),
        "Execution of forward_backward for method " + method + " failed");
    return nullptr;
  }

  jobjectArray jresult = env->NewObjectArray(
      result.get().size(),
      g_training_cache.evalue_class,
      nullptr);
  if (jresult == nullptr) {
    return nullptr;
  }

  for (size_t i = 0; i < result.get().size(); i++) {
    jobject jevalue = newJEValueFromEValue(env, result.get()[i]);
    if (jevalue == nullptr) {
      return nullptr;
    }
    env->SetObjectArrayElement(jresult, i, jevalue);
    env->DeleteLocalRef(jevalue);
  }
  
  return jresult;
}

JNIEXPORT jobject JNICALL
Java_org_pytorch_executorch_training_TrainingModule_namedParametersNative(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jstring methodName) {
  
  g_training_cache.init(env);
  
  auto* module = reinterpret_cast<training::TrainingModule*>(nativeHandle);
  if (module == nullptr) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        "Invalid native handle");
    return nullptr;
  }

  std::string method = jstring_to_string(env, methodName);
  auto result = module->named_parameters(method);
  if (!result.ok()) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(result.error()),
        "Getting named parameters for method " + method + " failed");
    return nullptr;
  }

  jobject parameters = env->NewObject(
      g_training_cache.hashmap_class,
      g_training_cache.hashmap_init);
  if (parameters == nullptr) {
    return nullptr;
  }

  for (auto& [layer, tensor] : result.get()) {
    jstring jKey = env->NewStringUTF(layer.data());
    jobject jTensor = newJTensorFromTensor(env, tensor);
    if (jKey == nullptr || jTensor == nullptr) {
      env->DeleteLocalRef(parameters);
      if (jKey) env->DeleteLocalRef(jKey);
      if (jTensor) env->DeleteLocalRef(jTensor);
      return nullptr;
    }
    env->CallObjectMethod(
        parameters,
        g_training_cache.hashmap_put,
        jKey,
        jTensor);
    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(jTensor);
  }

  return parameters;
}

JNIEXPORT jobject JNICALL
Java_org_pytorch_executorch_training_TrainingModule_namedGradientsNative(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jstring methodName) {
  
  g_training_cache.init(env);
  
  auto* module = reinterpret_cast<training::TrainingModule*>(nativeHandle);
  if (module == nullptr) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        "Invalid native handle");
    return nullptr;
  }

  std::string method = jstring_to_string(env, methodName);
  auto result = module->named_gradients(method);
  if (!result.ok()) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(result.error()),
        "Getting named gradients for method " + method + " failed");
    return nullptr;
  }

  jobject gradients = env->NewObject(
      g_training_cache.hashmap_class,
      g_training_cache.hashmap_init);
  if (gradients == nullptr) {
    return nullptr;
  }

  for (auto& [layer, tensor] : result.get()) {
    jstring jKey = env->NewStringUTF(layer.data());
    jobject jTensor = newJTensorFromTensor(env, tensor);
    if (jKey == nullptr || jTensor == nullptr) {
      env->DeleteLocalRef(gradients);
      if (jKey) env->DeleteLocalRef(jKey);
      if (jTensor) env->DeleteLocalRef(jTensor);
      return nullptr;
    }
    env->CallObjectMethod(
        gradients,
        g_training_cache.hashmap_put,
        jKey,
        jTensor);
    env->DeleteLocalRef(jKey);
    env->DeleteLocalRef(jTensor);
  }

  return gradients;
}

// SGD native methods
JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_training_SGD_initHybrid(
    JNIEnv* env,
    jclass /* clazz */,
    jobject namedParameters,
    jdouble learningRate,
    jdouble momentum,
    jdouble dampening,
    jdouble weightDecay,
    jboolean nesterov) {
  
  g_training_cache.init(env);

  // Extract named parameters from Java Map
  std::map<std::string_view, executorch::aten::Tensor> cppNamedParameters;
  std::vector<std::string>* parameterNames = new std::vector<std::string>();
  std::vector<TensorPtr>* paramTensorPtrs = new std::vector<TensorPtr>();

  // Get entrySet from the map
  jobject entrySet = env->CallObjectMethod(namedParameters, g_training_cache.hashmap_entrySet);
  jobject iterator = env->CallObjectMethod(entrySet, g_training_cache.set_iterator);

  while (env->CallBooleanMethod(iterator, g_training_cache.iterator_hasNext)) {
    jobject entry = env->CallObjectMethod(iterator, g_training_cache.iterator_next);
    jstring key = static_cast<jstring>(env->CallObjectMethod(entry, g_training_cache.entry_getKey));
    jobject value = env->CallObjectMethod(entry, g_training_cache.entry_getValue);

    std::string paramName = jstring_to_string(env, key);
    TensorPtr tensor = newTensorFromJTensor(env, value);

    parameterNames->push_back(paramName);
    paramTensorPtrs->push_back(tensor);
    cppNamedParameters.emplace(
        std::string_view(parameterNames->back()), *tensor);

    env->DeleteLocalRef(entry);
    env->DeleteLocalRef(key);
    env->DeleteLocalRef(value);
  }

  env->DeleteLocalRef(iterator);
  env->DeleteLocalRef(entrySet);

  optimizer::SGDOptions options(
      learningRate, momentum, dampening, weightDecay, nesterov);
  auto* sgd = new optimizer::SGD(cppNamedParameters, options);

  // Pack the SGD optimizer and auxiliary data into a structure
  struct SGDData {
    optimizer::SGD* optimizer;
    std::vector<std::string>* parameterNames;
    std::vector<TensorPtr>* paramTensorPtrs;
  };

  SGDData* data = new SGDData{sgd, parameterNames, paramTensorPtrs};
  return reinterpret_cast<jlong>(data);
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_training_SGD_nativeDestroy(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jlong nativeHandle) {
  if (nativeHandle != 0) {
    struct SGDData {
      optimizer::SGD* optimizer;
      std::vector<std::string>* parameterNames;
      std::vector<TensorPtr>* paramTensorPtrs;
    };
    
    auto* data = reinterpret_cast<SGDData*>(nativeHandle);
    delete data->optimizer;
    delete data->parameterNames;
    delete data->paramTensorPtrs;
    delete data;
  }
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_training_SGD_stepNative(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jobject namedGradients) {
  
  g_training_cache.init(env);

  struct SGDData {
    optimizer::SGD* optimizer;
    std::vector<std::string>* parameterNames;
    std::vector<TensorPtr>* paramTensorPtrs;
  };

  auto* data = reinterpret_cast<SGDData*>(nativeHandle);
  if (data == nullptr || data->optimizer == nullptr) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        "Invalid native handle");
    return;
  }

  // Extract named gradients from Java Map
  std::map<std::string_view, executorch::aten::Tensor> cppNamedGradients;
  std::vector<std::string> gradientNames;
  std::vector<TensorPtr> tensorKeepalives;

  jobject entrySet = env->CallObjectMethod(namedGradients, g_training_cache.hashmap_entrySet);
  jobject iterator = env->CallObjectMethod(entrySet, g_training_cache.set_iterator);

  while (env->CallBooleanMethod(iterator, g_training_cache.iterator_hasNext)) {
    jobject entry = env->CallObjectMethod(iterator, g_training_cache.iterator_next);
    jstring key = static_cast<jstring>(env->CallObjectMethod(entry, g_training_cache.entry_getKey));
    jobject value = env->CallObjectMethod(entry, g_training_cache.entry_getValue);

    std::string gradName = jstring_to_string(env, key);
    TensorPtr tensor = newTensorFromJTensor(env, value);

    gradientNames.push_back(gradName);
    tensorKeepalives.push_back(tensor);
    cppNamedGradients.emplace(
        std::string_view(gradientNames.back()), *tensor);

    env->DeleteLocalRef(entry);
    env->DeleteLocalRef(key);
    env->DeleteLocalRef(value);
  }

  env->DeleteLocalRef(iterator);
  env->DeleteLocalRef(entrySet);

  auto result = data->optimizer->step(cppNamedGradients);
  if (result != ::executorch::runtime::Error::Ok) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(result),
        "SGD optimization step failed");
  }
}

} // extern "C"

// Function to register training module natives (for compatibility)
void register_natives_for_training(JNIEnv* env) {
  // With pure JNI, natives are registered via JNI_OnLoad or statically
  // This function can be a no-op or used for manual registration if needed
  g_training_cache.init(env);
}
