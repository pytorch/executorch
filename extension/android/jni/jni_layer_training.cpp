/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <jni.h>

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
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace executorch::extension;
using namespace executorch::extension::training;
using namespace torch::executor;

namespace {

// EValue type codes (must match Java EValue class)
constexpr int kTypeCodeNone = 0;
constexpr int kTypeCodeTensor = 1;
constexpr int kTypeCodeString = 2;
constexpr int kTypeCodeDouble = 3;
constexpr int kTypeCodeInt = 4;
constexpr int kTypeCodeBool = 5;

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

// Helper to throw a Java exception
void throwJavaException(JNIEnv* env, const char* message) {
  jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
  if (exceptionClass != nullptr) {
    env->ThrowNew(exceptionClass, message);
    env->DeleteLocalRef(exceptionClass);
  }
}

// Cached class and method IDs for training module
struct TrainingJniCache {
  jclass tensor_class = nullptr;
  jclass evalue_class = nullptr;
  jclass hashmap_class = nullptr;
  jclass bytebuffer_class = nullptr;
  jclass byteorder_class = nullptr;
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
  jmethodID map_entrySet = nullptr;
  jmethodID set_iterator = nullptr;
  jmethodID iterator_hasNext = nullptr;
  jmethodID iterator_next = nullptr;
  jmethodID entry_getKey = nullptr;
  jmethodID entry_getValue = nullptr;
  jmethodID map_size = nullptr;
  jmethodID bytebuffer_order = nullptr;
  jmethodID byteorder_nativeOrder = nullptr;

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
      tensor_dtypeJniCode =
          env->GetMethodID(tensor_class, "dtypeJniCode", "()I");
      tensor_getRawDataBuffer = env->GetMethodID(
          tensor_class, "getRawDataBuffer", "()Ljava/nio/Buffer;");
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
      evalue_from_long = env->GetStaticMethodID(
          evalue_class, "from", "(J)Lorg/pytorch/executorch/EValue;");
      evalue_from_double = env->GetStaticMethodID(
          evalue_class, "from", "(D)Lorg/pytorch/executorch/EValue;");
      evalue_from_bool = env->GetStaticMethodID(
          evalue_class, "from", "(Z)Lorg/pytorch/executorch/EValue;");
      evalue_from_string = env->GetStaticMethodID(
          evalue_class,
          "from",
          "(Ljava/lang/String;)Lorg/pytorch/executorch/EValue;");
      evalue_toTensor = env->GetMethodID(
          evalue_class, "toTensor", "()Lorg/pytorch/executorch/Tensor;");
      evalue_mTypeCode = env->GetFieldID(evalue_class, "mTypeCode", "I");
      evalue_mData =
          env->GetFieldID(evalue_class, "mData", "Ljava/lang/Object;");
    }

    // Cache HashMap class and methods
    jclass local_hashmap_class = env->FindClass("java/util/HashMap");
    if (local_hashmap_class != nullptr) {
      hashmap_class =
          static_cast<jclass>(env->NewGlobalRef(local_hashmap_class));
      env->DeleteLocalRef(local_hashmap_class);

      hashmap_init = env->GetMethodID(hashmap_class, "<init>", "()V");
      hashmap_put = env->GetMethodID(
          hashmap_class,
          "put",
          "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    }

    // Cache Map iteration methods
    jclass map_class = env->FindClass("java/util/Map");
    if (map_class != nullptr) {
      map_entrySet =
          env->GetMethodID(map_class, "entrySet", "()Ljava/util/Set;");
      map_size = env->GetMethodID(map_class, "size", "()I");
      env->DeleteLocalRef(map_class);
    }

    jclass set_class = env->FindClass("java/util/Set");
    if (set_class != nullptr) {
      set_iterator =
          env->GetMethodID(set_class, "iterator", "()Ljava/util/Iterator;");
      env->DeleteLocalRef(set_class);
    }

    jclass iterator_class = env->FindClass("java/util/Iterator");
    if (iterator_class != nullptr) {
      iterator_hasNext = env->GetMethodID(iterator_class, "hasNext", "()Z");
      iterator_next =
          env->GetMethodID(iterator_class, "next", "()Ljava/lang/Object;");
      env->DeleteLocalRef(iterator_class);
    }

    jclass entry_class = env->FindClass("java/util/Map$Entry");
    if (entry_class != nullptr) {
      entry_getKey =
          env->GetMethodID(entry_class, "getKey", "()Ljava/lang/Object;");
      entry_getValue =
          env->GetMethodID(entry_class, "getValue", "()Ljava/lang/Object;");
      env->DeleteLocalRef(entry_class);
    }

    // Cache ByteBuffer and ByteOrder classes and methods
    jclass local_bytebuffer_class = env->FindClass("java/nio/ByteBuffer");
    if (local_bytebuffer_class != nullptr) {
      bytebuffer_class =
          static_cast<jclass>(env->NewGlobalRef(local_bytebuffer_class));
      env->DeleteLocalRef(local_bytebuffer_class);

      bytebuffer_order = env->GetMethodID(
          bytebuffer_class,
          "order",
          "(Ljava/nio/ByteOrder;)Ljava/nio/ByteBuffer;");
    }

    jclass local_byteorder_class = env->FindClass("java/nio/ByteOrder");
    if (local_byteorder_class != nullptr) {
      byteorder_class =
          static_cast<jclass>(env->NewGlobalRef(local_byteorder_class));
      env->DeleteLocalRef(local_byteorder_class);

      byteorder_nativeOrder = env->GetStaticMethodID(
          byteorder_class, "nativeOrder", "()Ljava/nio/ByteOrder;");
    }

    initialized = true;
  }
};

TrainingJniCache g_training_cache;

// Helper to create Java Tensor from native tensor
jobject newJTensorFromTensor(
    JNIEnv* env,
    const executorch::aten::Tensor& tensor) {
  g_training_cache.init(env);

  const auto scalarType = tensor.scalar_type();
  if (scalar_type_to_java_dtype.count(scalarType) == 0) {
    std::stringstream ss;
    ss << "Tensor scalar type " << static_cast<int>(scalarType)
       << " is not supported on java side";
    throwJavaException(env, ss.str().c_str());
    return nullptr;
  }
  int jdtype = scalar_type_to_java_dtype.at(scalarType);

  // Create shape array
  const auto& tensor_shape = tensor.sizes();
  jlongArray jTensorShape = env->NewLongArray(tensor_shape.size());
  if (jTensorShape == nullptr) {
    return nullptr;
  }
  std::vector<jlong> shape_vec;
  for (const auto& s : tensor_shape) {
    shape_vec.push_back(s);
  }
  env->SetLongArrayRegion(jTensorShape, 0, shape_vec.size(), shape_vec.data());

  // Create ByteBuffer wrapping tensor data
  jobject jTensorBuffer = env->NewDirectByteBuffer(
      const_cast<void*>(tensor.const_data_ptr()), tensor.nbytes());
  if (jTensorBuffer == nullptr) {
    env->DeleteLocalRef(jTensorShape);
    return nullptr;
  }

  // Set byte order to native order (using cached classes/methods)
  jobject nativeOrder = env->CallStaticObjectMethod(
      g_training_cache.byteorder_class,
      g_training_cache.byteorder_nativeOrder);
  env->CallObjectMethod(
      jTensorBuffer, g_training_cache.bytebuffer_order, nativeOrder);
  env->DeleteLocalRef(nativeOrder);

  // Call nativeNewTensor static method
  jobject result = env->CallStaticObjectMethod(
      g_training_cache.tensor_class,
      g_training_cache.tensor_nativeNewTensor,
      jTensorBuffer,
      jTensorShape,
      jdtype,
      static_cast<jlong>(0));

  env->DeleteLocalRef(jTensorBuffer);
  env->DeleteLocalRef(jTensorShape);

  return result;
}

// Helper to create native TensorPtr from Java Tensor
TensorPtr newTensorFromJTensor(JNIEnv* env, jobject jtensor) {
  g_training_cache.init(env);

  jint jdtype =
      env->CallIntMethod(jtensor, g_training_cache.tensor_dtypeJniCode);

  jlongArray jshape = static_cast<jlongArray>(
      env->GetObjectField(jtensor, g_training_cache.tensor_shape));

  jobject jbuffer =
      env->CallObjectMethod(jtensor, g_training_cache.tensor_getRawDataBuffer);

  jsize rank = env->GetArrayLength(jshape);

  std::vector<jlong> shapeArr(rank);
  env->GetLongArrayRegion(jshape, 0, rank, shapeArr.data());

  std::vector<executorch::aten::SizesType> shape_vec;
  shape_vec.reserve(rank);

  for (int i = 0; i < rank; i++) {
    shape_vec.push_back(shapeArr[i]);
  }

  if (java_dtype_to_scalar_type.count(jdtype) == 0) {
    std::stringstream ss;
    ss << "Unknown Tensor jdtype: " << jdtype;
    throwJavaException(env, ss.str().c_str());
    env->DeleteLocalRef(jshape);
    env->DeleteLocalRef(jbuffer);
    return nullptr;
  }

  ScalarType scalar_type = java_dtype_to_scalar_type.at(jdtype);
  void* data = env->GetDirectBufferAddress(jbuffer);
  TensorPtr result = from_blob(data, shape_vec, scalar_type);

  env->DeleteLocalRef(jshape);
  env->DeleteLocalRef(jbuffer);

  return result;
}

// Helper to create Java EValue from native EValue
jobject newJEValueFromEValue(JNIEnv* env, runtime::EValue evalue) {
  g_training_cache.init(env);

  if (evalue.isTensor()) {
    jobject jtensor = newJTensorFromTensor(env, evalue.toTensor());
    if (jtensor == nullptr) {
      return nullptr;
    }
    jobject result = env->CallStaticObjectMethod(
        g_training_cache.evalue_class,
        g_training_cache.evalue_from_tensor,
        jtensor);
    env->DeleteLocalRef(jtensor);
    return result;
  } else if (evalue.isInt()) {
    return env->CallStaticObjectMethod(
        g_training_cache.evalue_class,
        g_training_cache.evalue_from_long,
        evalue.toInt());
  } else if (evalue.isDouble()) {
    return env->CallStaticObjectMethod(
        g_training_cache.evalue_class,
        g_training_cache.evalue_from_double,
        evalue.toDouble());
  } else if (evalue.isBool()) {
    return env->CallStaticObjectMethod(
        g_training_cache.evalue_class,
        g_training_cache.evalue_from_bool,
        static_cast<jboolean>(evalue.toBool()));
  } else if (evalue.isString()) {
    std::string str =
        std::string(evalue.toString().begin(), evalue.toString().end());
    jstring jstr = env->NewStringUTF(str.c_str());
    jobject result = env->CallStaticObjectMethod(
        g_training_cache.evalue_class,
        g_training_cache.evalue_from_string,
        jstr);
    env->DeleteLocalRef(jstr);
    return result;
  }

  std::stringstream ss;
  ss << "Unknown EValue type: " << static_cast<int>(evalue.tag);
  throwJavaException(env, ss.str().c_str());
  return nullptr;
}

// Helper to get TensorPtr from Java EValue
TensorPtr JEValueToTensorImpl(JNIEnv* env, jobject jevalue) {
  g_training_cache.init(env);

  jint typeCode =
      env->GetIntField(jevalue, g_training_cache.evalue_mTypeCode);
  if (typeCode == kTypeCodeTensor) {
    jobject jtensor =
        env->CallObjectMethod(jevalue, g_training_cache.evalue_toTensor);
    TensorPtr result = newTensorFromJTensor(env, jtensor);
    env->DeleteLocalRef(jtensor);
    return result;
  }

  std::stringstream ss;
  ss << "Unknown EValue typeCode: " << typeCode;
  throwJavaException(env, ss.str().c_str());
  return nullptr;
}

} // anonymous namespace

namespace executorch::extension {

// Native training module handle class
class TrainingModuleNative {
 public:
  std::unique_ptr<training::TrainingModule> module_;

  TrainingModuleNative(
      JNIEnv* env,
      jstring modelPath,
      jstring dataPath) {
    std::string modelPathString = jstring_to_string(env, modelPath);
    auto modelLoaderRes = FileDataLoader::from(modelPathString.c_str());
    if (modelLoaderRes.error() != Error::Ok) {
      std::stringstream ss;
      ss << "Failed to open model file: " << modelPathString;
      throwJavaException(env, ss.str().c_str());
      return;
    }
    auto modelLoader =
        std::make_unique<FileDataLoader>(std::move(modelLoaderRes.get()));

    std::unique_ptr<FileDataLoader> dataLoader = nullptr;
    std::string dataPathString = jstring_to_string(env, dataPath);
    if (!dataPathString.empty()) {
      auto dataLoaderRes = FileDataLoader::from(dataPathString.c_str());
      if (dataLoaderRes.error() != Error::Ok) {
        std::stringstream ss;
        ss << "Failed to open ptd file: " << dataPathString;
        throwJavaException(env, ss.str().c_str());
        return;
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
};

// Native SGD optimizer handle class
class SGDNative {
 public:
  std::unique_ptr<optimizer::SGD> sgdOptimizer_;
  std::vector<std::string>
      parameterNames_; // Store parameter names to keep string_view valid
  std::vector<TensorPtr>
      paramTensorPtrs_; // Store parameter tensors to keep TensorPtrs valid.

  SGDNative(
      JNIEnv* env,
      jobject namedParameters,
      jdouble learningRate,
      jdouble momentum,
      jdouble dampening,
      jdouble weightDecay,
      jboolean nesterov) {
    g_training_cache.init(env);

    std::map<std::string_view, executorch::aten::Tensor> cppNamedParameters;

    // Get the size of the map
    jint mapSize =
        env->CallIntMethod(namedParameters, g_training_cache.map_size);

    // Reserve space
    parameterNames_.reserve(mapSize);
    paramTensorPtrs_.reserve(mapSize);

    // Get entry set and iterate
    jobject entrySet =
        env->CallObjectMethod(namedParameters, g_training_cache.map_entrySet);
    jobject iterator =
        env->CallObjectMethod(entrySet, g_training_cache.set_iterator);

    while (env->CallBooleanMethod(iterator, g_training_cache.iterator_hasNext)) {
      jobject entry =
          env->CallObjectMethod(iterator, g_training_cache.iterator_next);
      jstring key = static_cast<jstring>(
          env->CallObjectMethod(entry, g_training_cache.entry_getKey));
      jobject value =
          env->CallObjectMethod(entry, g_training_cache.entry_getValue);

      std::string paramName = jstring_to_string(env, key);
      TensorPtr tensor = newTensorFromJTensor(env, value);

      // Store the parameter name and tensor
      parameterNames_.push_back(paramName);
      paramTensorPtrs_.push_back(tensor);
      cppNamedParameters.emplace(
          std::string_view(parameterNames_.back()), *tensor);

      env->DeleteLocalRef(key);
      env->DeleteLocalRef(value);
      env->DeleteLocalRef(entry);
    }

    env->DeleteLocalRef(iterator);
    env->DeleteLocalRef(entrySet);

    optimizer::SGDOptions options(
        learningRate, momentum, dampening, weightDecay, nesterov);
    sgdOptimizer_ =
        std::make_unique<optimizer::SGD>(cppNamedParameters, options);
  }
};

} // namespace executorch::extension

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_training_TrainingModule_nativeCreate(
    JNIEnv* env,
    jclass /* clazz */,
    jstring modelPath,
    jstring dataPath) {
  auto* native =
      new executorch::extension::TrainingModuleNative(env, modelPath, dataPath);
  return reinterpret_cast<jlong>(native);
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_training_TrainingModule_nativeDestroy(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jlong nativeHandle) {
  if (nativeHandle != 0) {
    auto* native =
        reinterpret_cast<executorch::extension::TrainingModuleNative*>(
            nativeHandle);
    delete native;
  }
}

JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_training_TrainingModule_nativeExecuteForwardBackward(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jstring methodName,
    jobjectArray jinputs) {
  auto* native =
      reinterpret_cast<executorch::extension::TrainingModuleNative*>(
          nativeHandle);
  if (native == nullptr) {
    throwJavaException(env, "Native handle is null");
    return nullptr;
  }

  g_training_cache.init(env);

  std::string method = jstring_to_string(env, methodName);
  jsize inputSize = jinputs != nullptr ? env->GetArrayLength(jinputs) : 0;

  std::vector<runtime::EValue> evalues;
  std::vector<TensorPtr> tensors;

  for (jsize i = 0; i < inputSize; i++) {
    jobject jevalue = env->GetObjectArrayElement(jinputs, i);
    jint typeCode =
        env->GetIntField(jevalue, g_training_cache.evalue_mTypeCode);

    if (typeCode == kTypeCodeTensor) {
      tensors.emplace_back(JEValueToTensorImpl(env, jevalue));
      evalues.emplace_back(tensors.back());
    } else if (typeCode == kTypeCodeInt) {
      jobject mData =
          env->GetObjectField(jevalue, g_training_cache.evalue_mData);
      jclass longClass = env->FindClass("java/lang/Long");
      jmethodID longValue = env->GetMethodID(longClass, "longValue", "()J");
      jlong value = env->CallLongMethod(mData, longValue);
      evalues.emplace_back(static_cast<int64_t>(value));
      env->DeleteLocalRef(mData);
      env->DeleteLocalRef(longClass);
    } else if (typeCode == kTypeCodeDouble) {
      jobject mData =
          env->GetObjectField(jevalue, g_training_cache.evalue_mData);
      jclass doubleClass = env->FindClass("java/lang/Double");
      jmethodID doubleValue =
          env->GetMethodID(doubleClass, "doubleValue", "()D");
      jdouble value = env->CallDoubleMethod(mData, doubleValue);
      evalues.emplace_back(static_cast<double>(value));
      env->DeleteLocalRef(mData);
      env->DeleteLocalRef(doubleClass);
    } else if (typeCode == kTypeCodeBool) {
      jobject mData =
          env->GetObjectField(jevalue, g_training_cache.evalue_mData);
      jclass boolClass = env->FindClass("java/lang/Boolean");
      jmethodID boolValue = env->GetMethodID(boolClass, "booleanValue", "()Z");
      jboolean value = env->CallBooleanMethod(mData, boolValue);
      evalues.emplace_back(static_cast<bool>(value));
      env->DeleteLocalRef(mData);
      env->DeleteLocalRef(boolClass);
    }
    env->DeleteLocalRef(jevalue);
  }

  auto result = native->module_->execute_forward_backward(method, evalues);
  if (!result.ok()) {
    std::stringstream ss;
    ss << "Execution of forward_backward for method " << method
       << " failed with status 0x" << std::hex
       << static_cast<error_code_t>(result.error());
    throwJavaException(env, ss.str().c_str());
    return nullptr;
  }

  jobjectArray jresult = env->NewObjectArray(
      result.get().size(), g_training_cache.evalue_class, nullptr);

  for (size_t i = 0; i < result.get().size(); i++) {
    jobject jevalue = newJEValueFromEValue(env, result.get()[i]);
    env->SetObjectArrayElement(jresult, i, jevalue);
    if (jevalue != nullptr) {
      env->DeleteLocalRef(jevalue);
    }
  }
  return jresult;
}

JNIEXPORT jobject JNICALL
Java_org_pytorch_executorch_training_TrainingModule_nativeNamedParameters(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jstring methodName) {
  auto* native =
      reinterpret_cast<executorch::extension::TrainingModuleNative*>(
          nativeHandle);
  if (native == nullptr) {
    throwJavaException(env, "Native handle is null");
    return nullptr;
  }

  g_training_cache.init(env);

  std::string method = jstring_to_string(env, methodName);
  auto result = native->module_->named_parameters(method);
  if (!result.ok()) {
    std::stringstream ss;
    ss << "Getting named parameters for method " << method
       << " failed with status 0x" << std::hex
       << static_cast<error_code_t>(result.error());
    throwJavaException(env, ss.str().c_str());
    return nullptr;
  }

  // Create a new HashMap
  jobject hashMap = env->NewObject(
      g_training_cache.hashmap_class, g_training_cache.hashmap_init);

  for (auto& [layer, tensor] : result.get()) {
    jstring jkey = env->NewStringUTF(std::string(layer).c_str());
    jobject jtensor = newJTensorFromTensor(env, tensor);
    env->CallObjectMethod(
        hashMap, g_training_cache.hashmap_put, jkey, jtensor);
    env->DeleteLocalRef(jkey);
    if (jtensor != nullptr) {
      env->DeleteLocalRef(jtensor);
    }
  }

  return hashMap;
}

JNIEXPORT jobject JNICALL
Java_org_pytorch_executorch_training_TrainingModule_nativeNamedGradients(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jstring methodName) {
  auto* native =
      reinterpret_cast<executorch::extension::TrainingModuleNative*>(
          nativeHandle);
  if (native == nullptr) {
    throwJavaException(env, "Native handle is null");
    return nullptr;
  }

  g_training_cache.init(env);

  std::string method = jstring_to_string(env, methodName);
  auto result = native->module_->named_gradients(method);
  if (!result.ok()) {
    std::stringstream ss;
    ss << "Getting named gradients for method " << method
       << " failed with status 0x" << std::hex
       << static_cast<error_code_t>(result.error());
    throwJavaException(env, ss.str().c_str());
    return nullptr;
  }

  // Create a new HashMap
  jobject hashMap = env->NewObject(
      g_training_cache.hashmap_class, g_training_cache.hashmap_init);

  for (auto& [layer, tensor] : result.get()) {
    jstring jkey = env->NewStringUTF(std::string(layer).c_str());
    jobject jtensor = newJTensorFromTensor(env, tensor);
    env->CallObjectMethod(
        hashMap, g_training_cache.hashmap_put, jkey, jtensor);
    env->DeleteLocalRef(jkey);
    if (jtensor != nullptr) {
      env->DeleteLocalRef(jtensor);
    }
  }

  return hashMap;
}

JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_training_SGD_nativeCreate(
    JNIEnv* env,
    jclass /* clazz */,
    jobject namedParameters,
    jdouble learningRate,
    jdouble momentum,
    jdouble dampening,
    jdouble weightDecay,
    jboolean nesterov) {
  auto* native = new executorch::extension::SGDNative(
      env,
      namedParameters,
      learningRate,
      momentum,
      dampening,
      weightDecay,
      nesterov);
  return reinterpret_cast<jlong>(native);
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_training_SGD_nativeDestroy(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jlong nativeHandle) {
  if (nativeHandle != 0) {
    auto* native =
        reinterpret_cast<executorch::extension::SGDNative*>(nativeHandle);
    delete native;
  }
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_training_SGD_nativeStep(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jobject namedGradients) {
  auto* native =
      reinterpret_cast<executorch::extension::SGDNative*>(nativeHandle);
  if (native == nullptr) {
    throwJavaException(env, "Native handle is null");
    return;
  }

  g_training_cache.init(env);

  std::map<std::string_view, executorch::aten::Tensor> cppNamedGradients;
  std::vector<std::string> gradientNames;
  std::vector<TensorPtr> tensorKeepalives;

  // Get the size of the map
  jint mapSize =
      env->CallIntMethod(namedGradients, g_training_cache.map_size);

  gradientNames.reserve(mapSize);
  tensorKeepalives.reserve(mapSize);

  // Get entry set and iterate
  jobject entrySet =
      env->CallObjectMethod(namedGradients, g_training_cache.map_entrySet);
  jobject iterator =
      env->CallObjectMethod(entrySet, g_training_cache.set_iterator);

  while (env->CallBooleanMethod(iterator, g_training_cache.iterator_hasNext)) {
    jobject entry =
        env->CallObjectMethod(iterator, g_training_cache.iterator_next);
    jstring key = static_cast<jstring>(
        env->CallObjectMethod(entry, g_training_cache.entry_getKey));
    jobject value =
        env->CallObjectMethod(entry, g_training_cache.entry_getValue);

    std::string gradName = jstring_to_string(env, key);
    TensorPtr tensor = newTensorFromJTensor(env, value);

    // Store the gradient name and tensor
    gradientNames.push_back(gradName);
    tensorKeepalives.push_back(tensor);
    cppNamedGradients.emplace(
        std::string_view(gradientNames.back()), *tensor);

    env->DeleteLocalRef(key);
    env->DeleteLocalRef(value);
    env->DeleteLocalRef(entry);
  }

  env->DeleteLocalRef(iterator);
  env->DeleteLocalRef(entrySet);

  auto result = native->sgdOptimizer_->step(cppNamedGradients);
  if (result != ::executorch::runtime::Error::Ok) {
    std::stringstream ss;
    ss << "SGD optimization step failed with status 0x" << std::hex
       << static_cast<error_code_t>(result);
    throwJavaException(env, ss.str().c_str());
  }
}

} // extern "C"

// Function to register training module natives
void register_natives_for_training(JNIEnv* env) {
  // Register TrainingModule natives
  jclass training_module_class =
      env->FindClass("org/pytorch/executorch/training/TrainingModule");
  if (training_module_class == nullptr) {
    ET_LOG(Error, "Failed to find TrainingModule class");
    env->ExceptionClear();
    return;
  }

  // clang-format off
  static const JNINativeMethod training_methods[] = {
      {"nativeCreate", "(Ljava/lang/String;Ljava/lang/String;)J",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_training_TrainingModule_nativeCreate)},
      {"nativeDestroy", "(J)V",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_training_TrainingModule_nativeDestroy)},
      {"nativeExecuteForwardBackward",
       "(JLjava/lang/String;[Lorg/pytorch/executorch/EValue;)[Lorg/pytorch/executorch/EValue;",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_training_TrainingModule_nativeExecuteForwardBackward)},
      {"nativeNamedParameters", "(JLjava/lang/String;)Ljava/util/Map;",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_training_TrainingModule_nativeNamedParameters)},
      {"nativeNamedGradients", "(JLjava/lang/String;)Ljava/util/Map;",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_training_TrainingModule_nativeNamedGradients)},
  };
  // clang-format on

  int result = env->RegisterNatives(
      training_module_class,
      training_methods,
      sizeof(training_methods) / sizeof(training_methods[0]));
  if (result != JNI_OK) {
    ET_LOG(Error, "Failed to register native methods for TrainingModule");
  }

  env->DeleteLocalRef(training_module_class);

  // Register SGD natives
  jclass sgd_class = env->FindClass("org/pytorch/executorch/training/SGD");
  if (sgd_class == nullptr) {
    ET_LOG(Error, "Failed to find SGD class");
    env->ExceptionClear();
    return;
  }

  // clang-format off
  static const JNINativeMethod sgd_methods[] = {
      {"nativeCreate", "(Ljava/util/Map;DDDDZ)J",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_training_SGD_nativeCreate)},
      {"nativeDestroy", "(J)V",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_training_SGD_nativeDestroy)},
      {"nativeStep", "(JLjava/util/Map;)V",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_training_SGD_nativeStep)},
  };
  // clang-format on

  result = env->RegisterNatives(
      sgd_class, sgd_methods, sizeof(sgd_methods) / sizeof(sgd_methods[0]));
  if (result != JNI_OK) {
    ET_LOG(Error, "Failed to register native methods for SGD");
  }

  env->DeleteLocalRef(sgd_class);
}
