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

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
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
          jni->GetDirectBufferAddress(jbuffer.get()), shape_vec, scalar_type);
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
  std::unique_ptr<Module> module_;
  std::string model_path_;

 public:
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/executorch/NativePeer;";

  static facebook::jni::local_ref<jhybriddata> initHybrid(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
          extraFiles,
      jint loadMode) {
    return makeCxxInstance(modelPath, extraFiles, loadMode);
  }

  ExecuTorchJni(
      facebook::jni::alias_ref<jstring> modelPath,
      facebook::jni::alias_ref<
          facebook::jni::JMap<facebook::jni::JString, facebook::jni::JString>>
          extraFiles,
      jint loadMode) {
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

    module_ = std::make_unique<Module>(modelPath->toStdString(), load_mode);
    model_path_ = modelPath->toStdString();
  }

  facebook::jni::local_ref<facebook::jni::JArrayClass<JEValue>> forward(
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
          jinputs) {
    return execute_method("forward", jinputs);
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

    if (!result.ok()) {
      facebook::jni::throwNewJavaException(
          "java/lang/Exception",
          "Execution of method %s failed with status 0x%" PRIx32,
          method.c_str(),
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

  jint forward_ones() {
      // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point to data that's already in memory, and
  // users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = model_path_.c_str();
  Result<torch::executor::util::FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      (uint32_t)loader.error());

  // Parse the program file. This is immutable, and can also be reused between
  // multiple execution invocations across multiple threads.
  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  // Use the first method in the program.
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Using method %s", method_name);

  // MethodMeta describes the memory requirements of the method.
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      method_name,
      (uint32_t)method_meta.error());

  //
  // The runtime does not use malloc/new; it allocates all memory using the
  // MemoryManger provided by the client. Clients are responsible for allocating
  // the memory ahead of time, or providing MemoryAllocator subclasses that can
  // do it dynamically.
  //

  // The method allocator is used to allocate all dynamic C++ metadata/objects
  // used to represent the loaded method. This allocator is only used during
  // loading a method of the program, which will return an error if there was
  // not enough memory.
  //
  // The amount of memory required depends on the loaded method and the runtime
  // code itself. The amount of memory here is usually determined by running the
  // method and seeing how much memory is actually used, though it's possible to
  // subclass MemoryAllocator so that it calls malloc() under the hood (see
  // MallocMemoryAllocator).
  //
  // In this example we use a statically allocated memory pool.
  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

  // The memory-planned buffers will back the mutable tensors used by the
  // method. The sizes of these buffers were determined ahead of time during the
  // memory-planning pasees.
  //
  // Each buffer typically corresponds to a different hardware memory bank. Most
  // mobile environments will only have a single buffer. Some embedded
  // environments may have more than one for, e.g., slow/large DRAM and
  // fast/small SRAM, or for memory associated with particular cores.
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the memory
  std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    // .get() will always succeed because id < num_memory_planned_buffers.
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  // Assemble all of the allocators into the MemoryManager that the Executor
  // will use.
  MemoryManager memory_manager(&method_allocator, &planned_memory);

  Result<Method> method =
      program->load_method(method_name, &memory_manager, event_tracer_ptr);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      (uint32_t)method.error());
  ET_LOG(Info, "Method loaded.");

  // Allocate input tensors and set all of their elements to 1. The `inputs`
  // variable owns the allocated memory and must live past the last call to
  // `execute()`.
  auto inputs = util::prepare_input_tensors(*method);
  ET_CHECK_MSG(
      inputs.ok(),
      "Could not prepare inputs: 0x%" PRIx32,
      (uint32_t)inputs.error());
  ET_LOG(Info, "Inputs prepared.");

  // Run the model.
  for (uint32_t i = 0; i < 10; i++) {
    Error status = method->execute();
    ET_CHECK_MSG(
        status == Error::Ok,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        (uint32_t)status);
  }
  ET_LOG(Info, "Model executed successfully %i time(s).", 10);

  // Print the outputs.
  std::vector<EValue> outputs(method->outputs_size());
  ET_LOG(Info, "%zu outputs: ", outputs.size());
  Error status = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);
  // Print the first and last 100 elements of long lists of scalars.
  std::cout << torch::executor::util::evalue_edge_items(100);
  for (int i = 0; i < outputs.size(); ++i) {
    std::cout << "Output " << i << ": " << outputs[i] << std::endl;
  }

  return 0;
  }


  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchJni::initHybrid),
        makeNativeMethod("forward", ExecuTorchJni::forward),
        makeNativeMethod("execute", ExecuTorchJni::execute),
        makeNativeMethod("loadMethod", ExecuTorchJni::load_method),
        makeNativeMethod("forwardOnes", ExecuTorchJni::forward_ones),
    });
  }
};

} // namespace executorch_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { executorch_jni::ExecuTorchJni::registerNatives(); });
}
