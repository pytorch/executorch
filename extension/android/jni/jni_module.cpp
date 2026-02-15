/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

using namespace executorch::extension;
using namespace torch::executor;

namespace executorch::extension {
class TensorHybrid : public facebook::jni::HybridClass<TensorHybrid> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/Tensor;";

  explicit TensorHybrid(executorch::aten::Tensor tensor) {}

  static facebook::jni::local_ref<TensorHybrid::javaobject>
  newJTensorFromTensor(const executorch::aten::Tensor& tensor) {
    // Java wrapper currently only supports contiguous tensors.

    const auto scalarType = tensor.scalar_type();
    int jdtype = scalar_type_to_java_dtype.at(scalarType);
    if (scalar_type_to_java_dtype.count(scalarType) == 0) {
      std::stringstream ss;
      ss << "executorch::aten::Tensor scalar [java] type: " << jdtype
         << " is not supported on java side";
      jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    }

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

  static TensorPtr newTensorFromJTensor(
      facebook::jni::alias_ref<TensorHybrid::javaobject> jtensor) {
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
    std::vector<executorch::aten::SizesType> shape_vec;
    shape_vec.reserve(rank);

    int64_t numel = 1;
    for (int i = 0; i < rank; i++) {
      shape_vec.push_back(shapeArr[i]);
    }
    for (int i = rank - 1; i >= 0; --i) {
      numel *= shapeArr[i];
    }
    JNIEnv* jni = facebook::jni::Environment::current();
    if (java_dtype_to_scalar_type.count(jdtype) == 0) {
      std::stringstream ss;
      ss << "Unknown Tensor jdtype: [" << jdtype << "]";
      jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    }
    ScalarType scalar_type = java_dtype_to_scalar_type.at(jdtype);
    const jlong dataCapacity = jni->GetDirectBufferCapacity(jbuffer.get());
    if (dataCapacity < 0) {
      std::stringstream ss;
      ss << "Tensor buffer is not direct or has invalid capacity";
      jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    }
    const size_t elementSize = executorch::runtime::elementSize(scalar_type);
    const jlong expectedElements = static_cast<jlong>(numel);
    const jlong expectedBytes =
        expectedElements * static_cast<jlong>(elementSize);
    const bool matchesElements = dataCapacity == expectedElements;
    const bool matchesBytes = dataCapacity == expectedBytes;
    if (!matchesElements && !matchesBytes) {
      std::stringstream ss;
      ss << "Tensor dimensions(elements number: " << numel
         << ") inconsistent with buffer capacity " << dataCapacity
         << " (element size bytes: " << elementSize << ")";
      jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    }
    return from_blob(
        jni->GetDirectBufferAddress(jbuffer.get()), shape_vec, scalar_type);
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
    std::stringstream ss;
    ss << "Unknown EValue type: [" << static_cast<int>(evalue.tag) << "]";
    jni_helper::throwExecutorchException(
        static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    return {};
  }

  static TensorPtr JEValueToTensorImpl(
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
      return TensorHybrid::newTensorFromJTensor(jtensor);
    }
    std::stringstream ss;
    ss << "Unknown EValue typeCode: " << typeCode;
    jni_helper::throwExecutorchException(
        static_cast<uint32_t>(Error::InvalidArgument), ss.str().c_str());
    return {};
  }
};

class ExecuTorchJni : public facebook::jni::HybridClass<ExecuTorchJni> {
 private:
  friend HybridBase;
  std::unique_ptr<Module> module_;

 public:
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/executorch/Module;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> modelPath,
      jint loadMode,
      jint numThreads) {
    return makeCxxInstance(modelPath, loadMode, numThreads);
  }

  ExecuTorchJni(
      facebook::jni::alias_ref<jstring> modelPath,
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
    module_ = std::make_unique<Module>(
        modelPath->toStdString(), load_mode, std::move(etdump_gen));

#ifdef ET_USE_THREADPOOL
    // Default to using cores/2 threadpool threads. The long-term plan is to
    // improve performant core detection in CPUInfo, but for now we can use
    // cores/2 as a sane default.
    //
    // Based on testing, this is almost universally faster than using all
    // cores, as efficiency cores can be quite slow. In extreme cases, using
    // all cores can be 10x slower than using cores/2.
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

  facebook::jni::local_ref<facebook::jni::JArrayClass<JEValue>> execute(
      facebook::jni::alias_ref<jstring> methodName,
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
          jinputs) {
    return execute_method(methodName->toStdString(), jinputs);
  }

  jint load_method(facebook::jni::alias_ref<jstring> methodName) {
    return static_cast<jint>(module_->load_method(methodName->toStdString()));
  }

  facebook::jni::local_ref<facebook::jni::JArrayClass<JEValue>> execute_method(
      std::string method,
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
          jinputs) {
    // If no inputs is given, it will run with sample inputs (ones)
    if (jinputs->size() == 0) {
      auto result = module_->load_method(method);
      if (result != Error::Ok) {
        // Format hex string
        std::stringstream ss;
        ss << "Cannot get method names [Native Error: 0x" << std::hex
           << std::uppercase << static_cast<uint32_t>(result) << "]";

        jni_helper::throwExecutorchException(
            static_cast<uint32_t>(result), ss.str());
        return {};
      }
      auto&& underlying_method = module_->methods_[method].method;
      auto&& buf = prepare_input_tensors(*underlying_method);
      result = underlying_method->execute();
      if (result != Error::Ok) {
        jni_helper::throwExecutorchException(
            static_cast<uint32_t>(result),
            "Execution failed for method: " + method);
        return {};
      }
      facebook::jni::local_ref<facebook::jni::JArrayClass<JEValue>> jresult =
          facebook::jni::JArrayClass<JEValue>::newArray(
              underlying_method->outputs_size());

      for (int i = 0; i < underlying_method->outputs_size(); i++) {
        auto jevalue =
            JEValue::newJEValueFromEValue(underlying_method->get_output(i));
        jresult->setElement(i, *jevalue);
      }
      return jresult;
    }

    std::vector<EValue> evalues;
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

    if (!result.ok()) {
      jni_helper::throwExecutorchException(
          static_cast<uint32_t>(result.error()),
          "Execution failed for method: " + method);
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

  facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>>
  readLogBuffer() {
    return readLogBufferUtil();
  }

  static facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>>
  readLogBufferStatic(facebook::jni::alias_ref<jclass>) {
    return readLogBufferUtil();
  }

  static facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>>
  readLogBufferUtil() {
#ifdef __ANDROID__

    facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>> ret;

    access_log_buffer([&](std::vector<log_entry>& buffer) {
      const auto size = buffer.size();
      ret = facebook::jni::JArrayClass<jstring>::newArray(size);
      for (auto i = 0u; i < size; i++) {
        const auto& entry = buffer[i];
        // Format the log entry as "[TIMESTAMP FUNCTION FILE:LINE] LEVEL
        // MESSAGE".
        std::stringstream ss;
        ss << "[" << entry.timestamp << " " << entry.function << " "
           << entry.filename << ":" << entry.line << "] "
           << static_cast<char>(entry.level) << " " << entry.message;

        facebook::jni::local_ref<facebook::jni::JString> jstr_message =
            facebook::jni::make_jstring(ss.str().c_str());
        (*ret)[i] = jstr_message;
      }
    });

    return ret;
#else
    return facebook::jni::JArrayClass<String>::newArray(0);
#endif
  }

  jboolean etdump() {
#ifdef EXECUTORCH_ANDROID_PROFILING
    executorch::etdump::ETDumpGen* etdumpgen =
        (executorch::etdump::ETDumpGen*)module_->event_tracer();
    auto etdump_data = etdumpgen->get_etdump_data();

    if (etdump_data.buf != nullptr && etdump_data.size > 0) {
      int etdump_file =
          open("/data/local/tmp/result.etdump", O_WRONLY | O_CREAT, 0644);
      if (etdump_file == -1) {
        ET_LOG(Error, "Cannot create result.etdump error: %d", errno);
        return false;
      }
      ssize_t bytes_written =
          write(etdump_file, (uint8_t*)etdump_data.buf, etdump_data.size);
      if (bytes_written == -1) {
        ET_LOG(Error, "Cannot write result.etdump error: %d", errno);
        return false;
      } else {
        ET_LOG(Info, "ETDump written %d bytes to file.", bytes_written);
      }
      close(etdump_file);
      free(etdump_data.buf);
      return true;
    } else {
      ET_LOG(Error, "No ETDump data available!");
    }
#endif
    return false;
  }

  facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>> getMethods() {
    const auto& names_result = module_->method_names();
    if (!names_result.ok()) {
      // Format hex string
      std::stringstream ss;
      ss << "Cannot get load module [Native Error: 0x" << std::hex
         << std::uppercase << static_cast<uint32_t>(names_result.error())
         << "]";

      jni_helper::throwExecutorchException(
          static_cast<uint32_t>(Error::InvalidArgument), ss.str());
      return {};
    }
    const auto& methods = names_result.get();
    facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>> ret =
        facebook::jni::JArrayClass<jstring>::newArray(methods.size());
    int i = 0;
    for (auto s : methods) {
      facebook::jni::local_ref<facebook::jni::JString> method_name =
          facebook::jni::make_jstring(s.c_str());
      (*ret)[i] = method_name;
      i++;
    }
    return ret;
  }

  facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>> getUsedBackends(
      facebook::jni::alias_ref<jstring> methodName) {
    auto methodMeta = module_->method_meta(methodName->toStdString()).get();
    std::unordered_set<std::string> backends;
    for (auto i = 0; i < methodMeta.num_backends(); i++) {
      backends.insert(methodMeta.get_backend_name(i).get());
    }

    facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>> ret =
        facebook::jni::JArrayClass<jstring>::newArray(backends.size());
    int i = 0;
    for (auto s : backends) {
      facebook::jni::local_ref<facebook::jni::JString> backend_name =
          facebook::jni::make_jstring(s.c_str());
      (*ret)[i] = backend_name;
      i++;
    }
    return ret;
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchJni::initHybrid),
        makeNativeMethod("executeNative", ExecuTorchJni::execute),
        makeNativeMethod("loadMethodNative", ExecuTorchJni::load_method),
        makeNativeMethod("readLogBufferNative", ExecuTorchJni::readLogBuffer),
        makeNativeMethod(
            "readLogBufferStaticNative", ExecuTorchJni::readLogBufferStatic),
        makeNativeMethod("etdump", ExecuTorchJni::etdump),
        makeNativeMethod("getMethods", ExecuTorchJni::getMethods),
        makeNativeMethod("getUsedBackends", ExecuTorchJni::getUsedBackends),
    });
  }
};
} // namespace executorch::extension

#ifdef EXECUTORCH_BUILD_LLAMA_JNI
extern void register_natives_for_llm();
#else
// No op if we don't build LLM
void register_natives_for_llm() {}
#endif
extern void register_natives_for_runtime();

#ifdef EXECUTORCH_BUILD_EXTENSION_TRAINING
extern void register_natives_for_training();
#else
// No op if we don't build training JNI
void register_natives_for_training() {}
#endif

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(vm, [] {
    executorch::extension::ExecuTorchJni::registerNatives();
    register_natives_for_llm();
    register_natives_for_runtime();
    register_natives_for_training();
  });
}
