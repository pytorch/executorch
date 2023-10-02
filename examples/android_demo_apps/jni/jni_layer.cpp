/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/util.h>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#ifdef __ANDROID__
#include <android/log.h>

void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
  __android_log_print(ANDROID_LOG_INFO, "ETLOG", "%s", message);
}

#endif

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

namespace executorch_jni {

constexpr static int kTensorDTypeUInt8 = 1;
constexpr static int kTensorDTypeInt8 = 2;
constexpr static int kTensorDTypeInt32 = 3;
constexpr static int kTensorDTypeFloat32 = 4;
constexpr static int kTensorDTypeInt64 = 5;
constexpr static int kTensorDTypeFloat64 = 6;

class TensorHybrid : public facebook::jni::HybridClass<TensorHybrid> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lcom/example/executorchdemo/executor/Tensor;";

  explicit TensorHybrid(exec_aten::Tensor tensor) : tensor_(tensor) {}

  static facebook::jni::local_ref<TensorHybrid::javaobject>
  newJTensorFromTensor(const exec_aten::Tensor& tensor) {
    // Java wrapper currently only supports contiguous tensors.

    const auto scalarType = tensor.scalar_type();
    int jdtype = 0;

    if (ScalarType::Float == scalarType) {
      jdtype = kTensorDTypeFloat32;
    } else if (ScalarType::Int == scalarType) {
      jdtype = kTensorDTypeInt32;
    } else if (ScalarType::Byte == scalarType) {
      jdtype = kTensorDTypeUInt8;
    } else if (ScalarType::Char == scalarType) {
      jdtype = kTensorDTypeInt8;
    } else if (ScalarType::Long == scalarType) {
      jdtype = kTensorDTypeInt64;
    } else if (ScalarType::Double == scalarType) {
      jdtype = kTensorDTypeFloat64;
    } else {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "exec_aten::Tensor scalar type is not supported on java side");
    }

    const auto& tensorShape = tensor.sizes();
    std::vector<jlong> tensorShapeVec;
    for (const auto& s : tensorShape) {
      tensorShapeVec.push_back(s);
    }
    facebook::jni::local_ref<jlongArray> jTensorShape =
        facebook::jni::make_long_array(tensorShapeVec.size());
    jTensorShape->setRegion(0, tensorShapeVec.size(), tensorShapeVec.data());

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
            jint,
            facebook::jni::alias_ref<jhybriddata>)>("nativeNewTensor");
    constexpr int kMemoryFormat = 1;
    return jMethodNewTensor(
        cls,
        jTensorBuffer,
        jTensorShape,
        jdtype,
        kMemoryFormat,
        makeCxxInstance(tensor));
  }

 private:
  friend HybridBase;
  exec_aten::Tensor tensor_;
};

class JEValue : public facebook::jni::JavaClass<JEValue> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lcom/example/executorchdemo/executor/EValue;";

  constexpr static int kTypeCodeTensor = 2;

  static facebook::jni::local_ref<JEValue> newJEValueFromEValue(EValue evalue) {
    // Note: evalue is valid as long as Method is valid before next execution.
    if (evalue.isTensor()) {
      static auto jMethodTensor =
          JEValue::javaClassStatic()
              ->getStaticMethod<facebook::jni::local_ref<JEValue>(
                  facebook::jni::local_ref<TensorHybrid::javaobject>)>("from");
      const auto& tensor = evalue.toTensor();
      return jMethodTensor(
          JEValue::javaClassStatic(),
          TensorHybrid::newJTensorFromTensor(tensor));
    }
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Unsupported EValue type: %d",
        evalue.tag);
  }

  static TensorImpl JEValueToEValue(
      facebook::jni::alias_ref<JEValue> JEValue,
      std::vector<exec_aten::SizesType>& shapeVec,
      std::vector<uint8_t>& dim_order,
      std::vector<int32_t>& strides) {
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

      static const auto memoryFormatMethod =
          cls->getMethod<jint()>("memoryFormatJniCode");
      jint jmemoryFormat = memoryFormatMethod(jtensor);

      static const auto shapeField = cls->getField<jlongArray>("shape");
      auto jshape = jtensor->getFieldValue(shapeField);

      static auto dataBufferMethod = cls->getMethod<
          facebook::jni::local_ref<facebook::jni::JBuffer::javaobject>()>(
          "getRawDataBuffer");
      facebook::jni::local_ref<facebook::jni::JBuffer> jbuffer =
          dataBufferMethod(jtensor);

      const auto rank = jshape->size();

      const auto shapeArr = jshape->getRegion(0, rank);
      shapeVec.reserve(rank);

      auto numel = 1;
      for (int i = 0; i < rank; i++) {
        shapeVec.push_back(shapeArr[i]);
        numel *= shapeArr[i];
      }
      JNIEnv* jni = facebook::jni::Environment::current();
      ScalarType scalar_type;
      if (kTensorDTypeFloat32 == jdtype) {
        scalar_type = ScalarType::Float;
      } else if (kTensorDTypeInt32 == jdtype) {
        scalar_type = ScalarType::Int;
      } else if (kTensorDTypeInt8 == jdtype) {
        scalar_type = ScalarType::Char;
      } else if (kTensorDTypeUInt8 == jdtype) {
        scalar_type = ScalarType::Byte;
      } else if (kTensorDTypeFloat64 == jdtype) {
        scalar_type = ScalarType::Double;
      } else if (kTensorDTypeInt64 == jdtype) {
        scalar_type = ScalarType::Long;
      } else {
        facebook::jni::throwNewJavaException(
            facebook::jni::gJavaLangIllegalArgumentException,
            "Unknown Tensor jdtype %d",
            jdtype);
      }
      const auto dataCapacity = jni->GetDirectBufferCapacity(jbuffer.get());
      if (dataCapacity != numel) {
        facebook::jni::throwNewJavaException(
            facebook::jni::gJavaLangIllegalArgumentException,
            "Tensor dimensions(elements number:%d inconsistent with buffer capacity(%d)",
            numel,
            dataCapacity);
      }
      return TensorImpl(
          scalar_type,
          shapeVec.size(),
          shapeVec.data(),
          jni->GetDirectBufferAddress(jbuffer.get()),
          dim_order.data(),
          strides.data(),
          TensorShapeDynamism::DYNAMIC_UNBOUND);
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
  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the memory
  std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
  HierarchicalAllocator planned_memory{
      {planned_spans.data(), planned_spans.size()}};
  MemoryManager memory_manager{&method_allocator, &planned_memory};
  std::unique_ptr<const Program> program_;
  std::unique_ptr<MemoryManager> memoryManager_;
  std::unique_ptr<Method> method_;

 public:
  constexpr static auto kJavaDescriptor =
      "Lcom/example/executorchdemo/executor/NativePeer;";

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
    // Loads a model file (pte) from the Java model path. extraFiles are not
    // used for now. The function loads the program and use `forward` method for
    // now. It extracts the metadata for the `forward` method and initialize the
    // memory allocator from the input/output data format. It only supports 1
    // tensor input and 1 tensor output for now. This initializes ExecuTorchJni
    // data members such as `planned_memory`, `memory_manager`, `method_`, etc.
    std::string model_path_str = modelPath->toStdString();
    const char* model_path = model_path_str.c_str();
    Result<FileDataLoader> loader = FileDataLoader::from(model_path);
    ET_CHECK_MSG(
        loader.ok(),
        "FileDataLoader::from() failed: 0x%" PRIx32,
        loader.error());

    Result<Program> program = Program::load(&loader.get());
    if (!program.ok()) {
      ET_LOG(Error, "Failed to parse model file %s", model_path);
    }
    ET_LOG(Info, "Model file %s is loaded.", model_path);
    program_ = std::make_unique<Program>(std::move(program.get()));

    // MethodMeta describes the memory requirements of the method.
    Result<MethodMeta> method_meta = program_->method_meta("forward");
    ET_CHECK_MSG(
        method_meta.ok(),
        "Failed to get method_meta for %s: 0x%x",
        "forward",
        (unsigned int)method_meta.error());

    size_t num_memory_planned_buffers =
        method_meta->num_memory_planned_buffers();
    for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
      // .get() will always succeed because id < num_memory_planned_buffers.
      size_t buffer_size = static_cast<size_t>(
          method_meta->memory_planned_buffer_size(id).get());
      ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
      planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
      planned_spans.push_back({planned_buffers.back().get(), buffer_size});
    }
    planned_memory =
        HierarchicalAllocator({planned_spans.data(), planned_spans.size()});

    memory_manager = MemoryManager(&method_allocator, &planned_memory);

    // Note: We only support "forward" method for now.
    auto method = program_->load_method("forward", &memory_manager);
    method_ = std::make_unique<Method>(std::move(method.get()));
  }

  facebook::jni::local_ref<JEValue> forward(
      facebook::jni::alias_ref<
          facebook::jni::JArrayClass<JEValue::javaobject>::javaobject>
          jinputs) {
    size_t n = jinputs->size();
    auto kDimOrder = std::vector<uint8_t>({0, 1, 2, 3});
    auto kStrides = std::vector<int32_t>({3 * 224 * 224, 224 * 224, 224, 1});
    std::vector<exec_aten::SizesType> shapeVec;
    auto tensor_impl = JEValue::JEValueToEValue(
        jinputs->getElement(0), shapeVec, kDimOrder, kStrides);
    EValue atEValue = EValue(exec_aten::Tensor(&tensor_impl));
    Error set_input_status = method_->set_input(atEValue, 0);
    ET_CHECK(set_input_status == Error::Ok);
    ET_LOG(Info, "Inputs prepared.");

    // Run the model.
    Error status = method_->execute();
    ET_CHECK_MSG(
        status == Error::Ok,
        "Execution of method forward failed with status 0x%" PRIx32,
        status);
    ET_LOG(Info, "Model executed successfully.");

    // Print the outputs.
    auto outputs = std::vector<EValue>(method_->outputs_size());
    status = method_->get_outputs(outputs.data(), outputs.size());
    ET_CHECK(status == Error::Ok);
    return JEValue::newJEValueFromEValue(outputs[0]);
  }

  static void registerNatives() {
    registerHybrid({
        makeNativeMethod("initHybrid", ExecuTorchJni::initHybrid),
        makeNativeMethod("forward", ExecuTorchJni::forward),
    });
  }
};

} // namespace executorch_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { executorch_jni::ExecuTorchJni::registerNatives(); });
}
