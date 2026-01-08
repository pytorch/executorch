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
#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef ET_USE_THREADPOOL
#include <cpuinfo.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

#ifdef EXECUTORCH_ANDROID_PROFILING
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace executorch::extension;
using namespace torch::executor;

namespace {

// Global JavaVM pointer for obtaining JNIEnv in callbacks
JavaVM* g_jvm = nullptr;

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

// EValue type codes (must match Java EValue class)
constexpr int kTypeCodeNone = 0;
constexpr int kTypeCodeTensor = 1;
constexpr int kTypeCodeString = 2;
constexpr int kTypeCodeDouble = 3;
constexpr int kTypeCodeInt = 4;
constexpr int kTypeCodeBool = 5;

// Cached class and method IDs for performance
struct JniCache {
  jclass tensor_class = nullptr;
  jclass evalue_class = nullptr;
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

    initialized = true;
  }
};

JniCache g_jni_cache;

// Native module handle class
class ExecuTorchModuleNative {
 public:
  std::unique_ptr<Module> module_;

  ExecuTorchModuleNative(
      JNIEnv* env,
      jstring modelPath,
      jint loadMode,
      jint numThreads) {
    Module::LoadMode load_mode = Module::LoadMode::Mmap;
    if (loadMode == 0) {
      load_mode = Module::LoadMode::File;
    } else if (loadMode == 1) {
      load_mode = Module::LoadMode::Mmap;
    } else if (loadMode == 2) {
      load_mode = Module::LoadMode::MmapUseMlock;
    } else if (loadMode == 3) {
      load_mode = Module::LoadMode::MmapUseMlockIgnoreErrors;
    }
#ifdef EXECUTORCH_ANDROID_PROFILING
    auto etdump_gen = std::make_unique<executorch::etdump::ETDumpGen>();
#else
    auto etdump_gen = nullptr;
#endif
    std::string path = jstring_to_string(env, modelPath);
    module_ = std::make_unique<Module>(path, load_mode, std::move(etdump_gen));

#ifdef ET_USE_THREADPOOL
    auto threadpool = executorch::extension::threadpool::get_threadpool();
    if (threadpool) {
      int thread_count =
          numThreads != 0 ? numThreads : cpuinfo_get_processors_count() / 2;
      if (thread_count > 0) {
        threadpool->_unsafe_reset_threadpool(thread_count);
      }
    }
#endif
  }
};

// Helper to create Java Tensor from native tensor
jobject newJTensorFromTensor(JNIEnv* env, const executorch::aten::Tensor& tensor) {
  g_jni_cache.init(env);

  const auto scalarType = tensor.scalar_type();
  if (scalar_type_to_java_dtype.count(scalarType) == 0) {
    std::stringstream ss;
    ss << "executorch::aten::Tensor scalar type is not supported on java side";
    jni_helper::throwExecutorchException(
        env, static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
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

  // Set byte order to native order
  jclass byteBufferClass = env->FindClass("java/nio/ByteBuffer");
  jmethodID orderMethod =
      env->GetMethodID(byteBufferClass, "order", "(Ljava/nio/ByteOrder;)Ljava/nio/ByteBuffer;");
  jclass byteOrderClass = env->FindClass("java/nio/ByteOrder");
  jmethodID nativeOrderMethod =
      env->GetStaticMethodID(byteOrderClass, "nativeOrder", "()Ljava/nio/ByteOrder;");
  jobject nativeOrder = env->CallStaticObjectMethod(byteOrderClass, nativeOrderMethod);
  env->CallObjectMethod(jTensorBuffer, orderMethod, nativeOrder);

  env->DeleteLocalRef(byteBufferClass);
  env->DeleteLocalRef(byteOrderClass);
  env->DeleteLocalRef(nativeOrder);

  // Call nativeNewTensor static method (pass 0 for nativeHandle since we don't need it)
  jobject result = env->CallStaticObjectMethod(
      g_jni_cache.tensor_class,
      g_jni_cache.tensor_nativeNewTensor,
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
  g_jni_cache.init(env);

  jint jdtype = env->CallIntMethod(jtensor, g_jni_cache.tensor_dtypeJniCode);

  jlongArray jshape =
      static_cast<jlongArray>(env->GetObjectField(jtensor, g_jni_cache.tensor_shape));

  jobject jbuffer = env->CallObjectMethod(jtensor, g_jni_cache.tensor_getRawDataBuffer);

  jsize rank = env->GetArrayLength(jshape);

  std::vector<jlong> shapeArr(rank);
  env->GetLongArrayRegion(jshape, 0, rank, shapeArr.data());

  std::vector<executorch::aten::SizesType> shape_vec;
  shape_vec.reserve(rank);

  int64_t numel = 1;
  for (int i = 0; i < rank; i++) {
    shape_vec.push_back(shapeArr[i]);
  }
  for (int i = rank - 1; i >= 0; --i) {
    numel *= shapeArr[i];
  }

  if (java_dtype_to_scalar_type.count(jdtype) == 0) {
    std::stringstream ss;
    ss << "Unknown Tensor jdtype: [" << jdtype << "]";
    jni_helper::throwExecutorchException(
        env, static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    env->DeleteLocalRef(jshape);
    env->DeleteLocalRef(jbuffer);
    return nullptr;
  }

  ScalarType scalar_type = java_dtype_to_scalar_type.at(jdtype);
  const jlong dataCapacity = env->GetDirectBufferCapacity(jbuffer);
  if (dataCapacity < 0) {
    std::stringstream ss;
    ss << "Tensor buffer is not direct or has invalid capacity";
    jni_helper::throwExecutorchException(
        env, static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    env->DeleteLocalRef(jshape);
    env->DeleteLocalRef(jbuffer);
    return nullptr;
  }

  const size_t elementSize = executorch::runtime::elementSize(scalar_type);
  const jlong expectedElements = static_cast<jlong>(numel);
  const jlong expectedBytes = expectedElements * static_cast<jlong>(elementSize);
  const bool matchesElements = dataCapacity == expectedElements;
  const bool matchesBytes = dataCapacity == expectedBytes;

  if (!matchesElements && !matchesBytes) {
    std::stringstream ss;
    ss << "Tensor dimensions(elements number: " << numel
       << ") inconsistent with buffer capacity " << dataCapacity
       << " (element size bytes: " << elementSize << ")";
    jni_helper::throwExecutorchException(
        env, static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    env->DeleteLocalRef(jshape);
    env->DeleteLocalRef(jbuffer);
    return nullptr;
  }

  void* data = env->GetDirectBufferAddress(jbuffer);
  TensorPtr result = from_blob(data, shape_vec, scalar_type);

  env->DeleteLocalRef(jshape);
  env->DeleteLocalRef(jbuffer);

  return result;
}

// Helper to create Java EValue from native EValue
jobject newJEValueFromEValue(JNIEnv* env, EValue evalue) {
  g_jni_cache.init(env);

  if (evalue.isTensor()) {
    jobject jtensor = newJTensorFromTensor(env, evalue.toTensor());
    if (jtensor == nullptr) {
      return nullptr;
    }
    jobject result = env->CallStaticObjectMethod(
        g_jni_cache.evalue_class, g_jni_cache.evalue_from_tensor, jtensor);
    env->DeleteLocalRef(jtensor);
    return result;
  } else if (evalue.isInt()) {
    return env->CallStaticObjectMethod(
        g_jni_cache.evalue_class, g_jni_cache.evalue_from_long, evalue.toInt());
  } else if (evalue.isDouble()) {
    return env->CallStaticObjectMethod(
        g_jni_cache.evalue_class, g_jni_cache.evalue_from_double, evalue.toDouble());
  } else if (evalue.isBool()) {
    return env->CallStaticObjectMethod(
        g_jni_cache.evalue_class,
        g_jni_cache.evalue_from_bool,
        static_cast<jboolean>(evalue.toBool()));
  } else if (evalue.isString()) {
    std::string str =
        std::string(evalue.toString().begin(), evalue.toString().end());
    jstring jstr = env->NewStringUTF(str.c_str());
    jobject result = env->CallStaticObjectMethod(
        g_jni_cache.evalue_class, g_jni_cache.evalue_from_string, jstr);
    env->DeleteLocalRef(jstr);
    return result;
  }

  std::stringstream ss;
  ss << "Unknown EValue type: [" << static_cast<int>(evalue.tag) << "]";
  jni_helper::throwExecutorchException(
      env, static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
  return nullptr;
}

// Helper to get TensorPtr from Java EValue
TensorPtr JEValueToTensorImpl(JNIEnv* env, jobject jevalue) {
  g_jni_cache.init(env);

  jint typeCode = env->GetIntField(jevalue, g_jni_cache.evalue_mTypeCode);
  if (typeCode == kTypeCodeTensor) {
    jobject jtensor =
        env->CallObjectMethod(jevalue, g_jni_cache.evalue_toTensor);
    TensorPtr result = newTensorFromJTensor(env, jtensor);
    env->DeleteLocalRef(jtensor);
    return result;
  }

  std::stringstream ss;
  ss << "Unknown EValue typeCode: " << typeCode;
  jni_helper::throwExecutorchException(
      env, static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
  return nullptr;
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_Module_nativeCreate(
    JNIEnv* env,
    jclass /* clazz */,
    jstring modelPath,
    jint loadMode,
    jint numThreads) {
  auto* native = new ExecuTorchModuleNative(env, modelPath, loadMode, numThreads);
  return reinterpret_cast<jlong>(native);
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_Module_nativeDestroy(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jlong nativeHandle) {
  auto* native = reinterpret_cast<ExecuTorchModuleNative*>(nativeHandle);
  delete native;
}

JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_Module_nativeExecute(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jstring methodName,
    jobjectArray jinputs) {
  auto* native = reinterpret_cast<ExecuTorchModuleNative*>(nativeHandle);
  if (native == nullptr) {
    return nullptr;
  }

  g_jni_cache.init(env);

  std::string method = jstring_to_string(env, methodName);
  jsize inputSize = jinputs != nullptr ? env->GetArrayLength(jinputs) : 0;

  // If no inputs is given, it will run with sample inputs (ones)
  if (inputSize == 0) {
    auto result = native->module_->load_method(method);
    if (result != Error::Ok) {
      std::stringstream ss;
      ss << "Cannot get method names [Native Error: 0x" << std::hex
         << std::uppercase << static_cast<uint32_t>(result) << "]";
      jni_helper::throwExecutorchException(
          env, static_cast<uint32_t>(result), ss.str());
      return nullptr;
    }
    auto&& underlying_method = native->module_->methods_[method].method;
    auto&& buf = prepare_input_tensors(*underlying_method);
    result = underlying_method->execute();
    if (result != Error::Ok) {
      jni_helper::throwExecutorchException(
          env, static_cast<uint32_t>(result), "Execution failed for method: " + method);
      return nullptr;
    }

    jobjectArray jresult =
        env->NewObjectArray(underlying_method->outputs_size(), g_jni_cache.evalue_class, nullptr);

    for (int i = 0; i < underlying_method->outputs_size(); i++) {
      jobject jevalue = newJEValueFromEValue(env, underlying_method->get_output(i));
      env->SetObjectArrayElement(jresult, i, jevalue);
      if (jevalue != nullptr) {
        env->DeleteLocalRef(jevalue);
      }
    }
    return jresult;
  }

  std::vector<EValue> evalues;
  std::vector<TensorPtr> tensors;

  for (int i = 0; i < inputSize; i++) {
    jobject jevalue = env->GetObjectArrayElement(jinputs, i);
    jint typeCode = env->GetIntField(jevalue, g_jni_cache.evalue_mTypeCode);

    if (typeCode == kTypeCodeTensor) {
      tensors.emplace_back(JEValueToTensorImpl(env, jevalue));
      evalues.emplace_back(tensors.back());
    } else if (typeCode == kTypeCodeInt) {
      jobject mData = env->GetObjectField(jevalue, g_jni_cache.evalue_mData);
      jclass longClass = env->FindClass("java/lang/Long");
      jmethodID longValue = env->GetMethodID(longClass, "longValue", "()J");
      jlong value = env->CallLongMethod(mData, longValue);
      evalues.emplace_back(static_cast<int64_t>(value));
      env->DeleteLocalRef(mData);
      env->DeleteLocalRef(longClass);
    } else if (typeCode == kTypeCodeDouble) {
      jobject mData = env->GetObjectField(jevalue, g_jni_cache.evalue_mData);
      jclass doubleClass = env->FindClass("java/lang/Double");
      jmethodID doubleValue = env->GetMethodID(doubleClass, "doubleValue", "()D");
      jdouble value = env->CallDoubleMethod(mData, doubleValue);
      evalues.emplace_back(static_cast<double>(value));
      env->DeleteLocalRef(mData);
      env->DeleteLocalRef(doubleClass);
    } else if (typeCode == kTypeCodeBool) {
      jobject mData = env->GetObjectField(jevalue, g_jni_cache.evalue_mData);
      jclass boolClass = env->FindClass("java/lang/Boolean");
      jmethodID boolValue = env->GetMethodID(boolClass, "booleanValue", "()Z");
      jboolean value = env->CallBooleanMethod(mData, boolValue);
      evalues.emplace_back(static_cast<bool>(value));
      env->DeleteLocalRef(mData);
      env->DeleteLocalRef(boolClass);
    }
    env->DeleteLocalRef(jevalue);
  }

#ifdef EXECUTORCH_ANDROID_PROFILING
  auto start = std::chrono::high_resolution_clock::now();
  auto result = native->module_->execute(method, evalues);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  ET_LOG(Debug, "Execution time: %lld ms.", duration);
#else
  auto result = native->module_->execute(method, evalues);
#endif

  if (!result.ok()) {
    jni_helper::throwExecutorchException(
        env,
        static_cast<uint32_t>(result.error()),
        "Execution failed for method: " + method);
    return nullptr;
  }

  jobjectArray jresult =
      env->NewObjectArray(result.get().size(), g_jni_cache.evalue_class, nullptr);

  for (size_t i = 0; i < result.get().size(); i++) {
    jobject jevalue = newJEValueFromEValue(env, result.get()[i]);
    env->SetObjectArrayElement(jresult, i, jevalue);
    if (jevalue != nullptr) {
      env->DeleteLocalRef(jevalue);
    }
  }
  return jresult;
}

JNIEXPORT jint JNICALL
Java_org_pytorch_executorch_Module_nativeLoadMethod(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jstring methodName) {
  auto* native = reinterpret_cast<ExecuTorchModuleNative*>(nativeHandle);
  if (native == nullptr) {
    return -1;
  }
  std::string method = jstring_to_string(env, methodName);
  return static_cast<jint>(native->module_->load_method(method));
}

JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_Module_nativeGetMethods(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle) {
  auto* native = reinterpret_cast<ExecuTorchModuleNative*>(nativeHandle);
  if (native == nullptr) {
    return nullptr;
  }

  const auto& names_result = native->module_->method_names();
  if (!names_result.ok()) {
    std::stringstream ss;
    ss << "Cannot get load module [Native Error: 0x" << std::hex
       << std::uppercase << static_cast<uint32_t>(names_result.error()) << "]";
    jni_helper::throwExecutorchException(
        env, static_cast<uint32_t>(Error::InvalidArgument), ss.str());
    return nullptr;
  }

  const auto& methods = names_result.get();
  jclass stringClass = env->FindClass("java/lang/String");
  jobjectArray ret = env->NewObjectArray(methods.size(), stringClass, nullptr);

  int i = 0;
  for (auto s : methods) {
    jstring method_name = env->NewStringUTF(s.c_str());
    env->SetObjectArrayElement(ret, i, method_name);
    env->DeleteLocalRef(method_name);
    i++;
  }
  env->DeleteLocalRef(stringClass);
  return ret;
}

JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_Module_nativeGetUsedBackends(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jstring methodName) {
  auto* native = reinterpret_cast<ExecuTorchModuleNative*>(nativeHandle);
  if (native == nullptr) {
    return nullptr;
  }

  std::string method = jstring_to_string(env, methodName);
  auto methodMeta = native->module_->method_meta(method).get();
  std::unordered_set<std::string> backends;
  for (auto i = 0; i < methodMeta.num_backends(); i++) {
    backends.insert(methodMeta.get_backend_name(i).get());
  }

  jclass stringClass = env->FindClass("java/lang/String");
  jobjectArray ret = env->NewObjectArray(backends.size(), stringClass, nullptr);

  int i = 0;
  for (auto s : backends) {
    jstring backend_name = env->NewStringUTF(s.c_str());
    env->SetObjectArrayElement(ret, i, backend_name);
    env->DeleteLocalRef(backend_name);
    i++;
  }
  env->DeleteLocalRef(stringClass);
  return ret;
}

JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_Module_nativeReadLogBuffer(
    JNIEnv* env,
    jclass /* clazz */,
    jlong /* nativeHandle */) {
#ifdef __ANDROID__
  jclass stringClass = env->FindClass("java/lang/String");
  jobjectArray ret = nullptr;

  access_log_buffer([&](std::vector<log_entry>& buffer) {
    const auto size = buffer.size();
    ret = env->NewObjectArray(size, stringClass, nullptr);
    for (auto i = 0u; i < size; i++) {
      const auto& entry = buffer[i];
      std::stringstream ss;
      ss << "[" << entry.timestamp << " " << entry.function << " "
         << entry.filename << ":" << entry.line << "] "
         << static_cast<char>(entry.level) << " " << entry.message;
      jstring jstr_message = env->NewStringUTF(ss.str().c_str());
      env->SetObjectArrayElement(ret, i, jstr_message);
      env->DeleteLocalRef(jstr_message);
    }
  });

  env->DeleteLocalRef(stringClass);
  return ret;
#else
  jclass stringClass = env->FindClass("java/lang/String");
  jobjectArray ret = env->NewObjectArray(0, stringClass, nullptr);
  env->DeleteLocalRef(stringClass);
  return ret;
#endif
}

JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_Module_nativeReadLogBufferStatic(
    JNIEnv* env,
    jclass clazz) {
  return Java_org_pytorch_executorch_Module_nativeReadLogBuffer(env, clazz, 0);
}

JNIEXPORT jboolean JNICALL
Java_org_pytorch_executorch_Module_nativeEtdump(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jlong nativeHandle) {
#ifdef EXECUTORCH_ANDROID_PROFILING
  auto* native = reinterpret_cast<ExecuTorchModuleNative*>(nativeHandle);
  if (native == nullptr) {
    return JNI_FALSE;
  }

  executorch::etdump::ETDumpGen* etdumpgen =
      (executorch::etdump::ETDumpGen*)native->module_->event_tracer();
  auto etdump_data = etdumpgen->get_etdump_data();

  if (etdump_data.buf != nullptr && etdump_data.size > 0) {
    int etdump_file =
        open("/data/local/tmp/result.etdump", O_WRONLY | O_CREAT, 0644);
    if (etdump_file == -1) {
      ET_LOG(Error, "Cannot create result.etdump error: %d", errno);
      return JNI_FALSE;
    }
    ssize_t bytes_written =
        write(etdump_file, (uint8_t*)etdump_data.buf, etdump_data.size);
    if (bytes_written == -1) {
      ET_LOG(Error, "Cannot write result.etdump error: %d", errno);
      return JNI_FALSE;
    } else {
      ET_LOG(Info, "ETDump written %d bytes to file.", bytes_written);
    }
    close(etdump_file);
    free(etdump_data.buf);
    return JNI_TRUE;
  } else {
    ET_LOG(Error, "No ETDump data available!");
  }
#endif
  return JNI_FALSE;
}

} // extern "C"

#ifdef EXECUTORCH_BUILD_LLAMA_JNI
extern void register_natives_for_llm(JNIEnv* env);
#else
// No op if we don't build LLM
void register_natives_for_llm(JNIEnv* /* env */) {}
#endif

#ifdef EXECUTORCH_BUILD_EXTENSION_TRAINING
extern void register_natives_for_training(JNIEnv* env);
#else
// No op if we don't build training JNI
void register_natives_for_training(JNIEnv* /* env */) {}
#endif

void register_natives_for_runtime(JNIEnv* env);

void register_natives_for_module(JNIEnv* env) {
  jclass module_class = env->FindClass("org/pytorch/executorch/Module");
  if (module_class == nullptr) {
    ET_LOG(Error, "Failed to find Module class");
    env->ExceptionClear();
    return;
  }

  // clang-format off
  static const JNINativeMethod methods[] = {
      {"nativeCreate", "(Ljava/lang/String;II)J",
       reinterpret_cast<void*>(Java_org_pytorch_executorch_Module_nativeCreate)},
      {"nativeDestroy", "(J)V",
       reinterpret_cast<void*>(Java_org_pytorch_executorch_Module_nativeDestroy)},
      {"nativeExecute",
       "(JLjava/lang/String;[Lorg/pytorch/executorch/EValue;)[Lorg/pytorch/executorch/EValue;",
       reinterpret_cast<void*>(Java_org_pytorch_executorch_Module_nativeExecute)},
      {"nativeLoadMethod", "(JLjava/lang/String;)I",
       reinterpret_cast<void*>(Java_org_pytorch_executorch_Module_nativeLoadMethod)},
      {"nativeGetMethods", "(J)[Ljava/lang/String;",
       reinterpret_cast<void*>(Java_org_pytorch_executorch_Module_nativeGetMethods)},
      {"nativeGetUsedBackends", "(JLjava/lang/String;)[Ljava/lang/String;",
       reinterpret_cast<void*>(Java_org_pytorch_executorch_Module_nativeGetUsedBackends)},
      {"nativeReadLogBuffer", "(J)[Ljava/lang/String;",
       reinterpret_cast<void*>(Java_org_pytorch_executorch_Module_nativeReadLogBuffer)},
      {"nativeReadLogBufferStatic", "()[Ljava/lang/String;",
       reinterpret_cast<void*>(Java_org_pytorch_executorch_Module_nativeReadLogBufferStatic)},
      {"nativeEtdump", "(J)Z",
       reinterpret_cast<void*>(Java_org_pytorch_executorch_Module_nativeEtdump)},
  };
  // clang-format on

  int num_methods = sizeof(methods) / sizeof(methods[0]);
  int result = env->RegisterNatives(module_class, methods, num_methods);
  if (result != JNI_OK) {
    ET_LOG(Error, "Failed to register native methods for Module");
  }

  env->DeleteLocalRef(module_class);
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  g_jvm = vm;
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  // Initialize the JNI cache
  g_jni_cache.init(env);

  // Register native methods
  register_natives_for_module(env);
  register_natives_for_llm(env);
  register_natives_for_runtime(env);
  register_natives_for_training(env);

  return JNI_VERSION_1_6;
}
