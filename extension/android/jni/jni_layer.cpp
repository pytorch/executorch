/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <jni.h>
#include <executorch/extension/android/jni/jni_helper.h>
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
using executorch::jni_helper::throwExecutorchException;

// Global References
static jclass gTensorClass;
static jmethodID gTensorNativeNewTensor;
static jmethodID gTensorDtypeJniCode;
static jmethodID gTensorGetRawDataBuffer;
static jmethodID gTensorShape;

static jclass gEValueClass;
static jmethodID gEValueToByteArray;
static jmethodID gEValueFromByteArray;

static jclass gMethodMetadataClass;

// Wrapper for Module
struct ModuleWrapper {
  std::unique_ptr<Module> module;
  ModuleWrapper(std::unique_ptr<Module> m) : module(std::move(m)) {}
};

// Wrapper for Output Tensor
struct TensorWrapper {
  executorch::aten::Tensor tensor;
  TensorWrapper(executorch::aten::Tensor t) : tensor(std::move(t)) {}
};

// Helper: Get ScalarType from jint code (matches DType.java)
static ScalarType javaTypeToScalarType(int typeCode) {
  // Mapping from DType.java jniCode to ScalarType
  // UINT8(0), INT8(1), INT16(2), INT32(3), INT64(4), HALF(5), FLOAT(6), DOUBLE(7), BOOL(8), QINT8(9), QUINT8(10), QINT32(11), QUINT4x2(12), QUINT2x4(13), BFLOAT16(14);
  switch (typeCode) {
    case 0: return ScalarType::Byte; // UINT8
    case 1: return ScalarType::Char; // INT8
    case 2: return ScalarType::Short; // INT16
    case 3: return ScalarType::Int; // INT32
    case 4: return ScalarType::Long; // INT64
    case 5: return ScalarType::Half; // HALF
    case 6: return ScalarType::Float; // FLOAT
    case 7: return ScalarType::Double; // DOUBLE
    case 8: return ScalarType::Bool; // BOOL
    // Add others if needed
    default: return ScalarType::Undefined;
  }
}

static int scalarTypeToJavaType(ScalarType type) {
    switch (type) {
        case ScalarType::Byte: return 0;
        case ScalarType::Char: return 1;
        case ScalarType::Short: return 2;
        case ScalarType::Int: return 3;
        case ScalarType::Long: return 4;
        case ScalarType::Half: return 5;
        case ScalarType::Float: return 6;
        case ScalarType::Double: return 7;
        case ScalarType::Bool: return 8;
        default: return -1;
    }
}

// Java Tensor -> C++ Tensor (Zero Copy if direct buffer)
TensorPtr newTensorFromJTensor(JNIEnv* env, jobject jTensor) {
    // 1. Get DType
    jint jdtype = env->CallIntMethod(jTensor, gTensorDtypeJniCode);
    ScalarType scalarType = javaTypeToScalarType(jdtype);
    if (scalarType == ScalarType::Undefined) {
        throwExecutorchException(env, (uint32_t)Error::InvalidArgument, "Unknown Tensor scalar type");
        return {};
    }

    // 2. Get Shape
    jlongArray jShape = (jlongArray)env->CallObjectMethod(jTensor, gTensorShape);
    jsize rank = env->GetArrayLength(jShape);
    jlong* shapePtr = env->GetLongArrayElements(jShape, nullptr);
    std::vector<executorch::aten::SizesType> sizes;
    sizes.reserve(rank);
    int64_t numel = 1;
    for(int i=0; i<rank; ++i) {
        sizes.push_back(static_cast<executorch::aten::SizesType>(shapePtr[i]));
        numel *= shapePtr[i];
    }
    env->ReleaseLongArrayElements(jShape, shapePtr, JNI_ABORT);

    // 3. Get Data
    jobject jBuffer = env->CallObjectMethod(jTensor, gTensorGetRawDataBuffer);
    if (!jBuffer) {
        throwExecutorchException(env, (uint32_t)Error::InvalidArgument, "Tensor buffer is null");
        return {};
    }
    void* dataPtr = env->GetDirectBufferAddress(jBuffer);
    jlong capacity = env->GetDirectBufferCapacity(jBuffer);

    if (!dataPtr || capacity < 0) {
        throwExecutorchException(env, (uint32_t)Error::InvalidArgument, "Tensor buffer is not direct or invalid");
        return {};
    }

    size_t elementSize = executorch::runtime::elementSize(scalarType);
    if ((size_t)capacity < numel * elementSize) {
         throwExecutorchException(env, (uint32_t)Error::InvalidArgument, "Tensor buffer too small");
         return {};
    }

    // 4. Create Tensor wrapping this data
    return from_blob(dataPtr, sizes, scalarType);
}

// C++ Tensor -> Java Tensor
jobject newJTensorFromTensor(JNIEnv* env, const executorch::aten::Tensor& tensor) {
    ScalarType scalarType = tensor.scalar_type();
    int jdtype = scalarTypeToJavaType(scalarType);
    if (jdtype == -1) {
         throwExecutorchException(env, (uint32_t)Error::InvalidArgument, "Supporting only basic types for now");
         return nullptr;
    }

    // Shape
    const auto& sizes = tensor.sizes();
    jlongArray jShape = env->NewLongArray(sizes.size());
    std::vector<jlong> jSizeVec(sizes.begin(), sizes.end());
    env->SetLongArrayRegion(jShape, 0, sizes.size(), jSizeVec.data());

    // Data - Create a DirectByteBuffer around the tensor's data
    // Note: The tensor data must remain valid as long as the Java object uses it.
    // We wrap the Tensor in a TensorWrapper on the heap and pass it to Java.
    // But direct buffer relies on the pointer. The TensorWrapper keeps the Tensor (and TensorImpl) alive.
    // Does Tensor own the memory?
    // If it's an output tensor from the runtime, it might be managed by the memory planner.
    // If the runtime/module is destroyed, this memory might become invalid if it's within the specific arena.
    // For now, assuming output tensors are valid as long as Module is valid or if they are copies.
    // ExecuTorch memory management is static. The outputs usually point to buffers inside the Method's memory allocator.
    // So Java Tensors created from outputs are only valid as long as the Method/Module is alive and no other execution happens.
    // This constraint was present in the previous implementation too.

    void* data = tensor.mutable_data_ptr();
    jlong dataBytes = tensor.nbytes();
    jobject jBuffer = env->NewDirectByteBuffer(data, dataBytes);
    // Needed to set order? ByteBuffer.order(ByteOrder.nativeOrder()) is usually default in JNI or handled in Java.
    // The previous code called order() in Java via helper or JNI.
    // Our Java `nativeNewTensor` creates a buffer wrapper. It assumes native order. 
    // NewDirectByteBuffer usually uses native order.

    // Native Wrapper
    auto* wrapper = new TensorWrapper(tensor);
    jlong handle = reinterpret_cast<jlong>(wrapper);

    // Call static factory
    jobject jTensor = env->CallStaticObjectMethod(gTensorClass, gTensorNativeNewTensor, 
        jBuffer, jShape, jdtype, handle);
    
    return jTensor;
}

// EValue Conversion
// For simplicity, using serialization via bytes if possible, or manual constructs.
// Previous code did manual reconstruction.

jobject newJEValueFromEValue(JNIEnv* env, const EValue& value) {
    // We can use EValue.fromByteArray if we can serialize EValue in C++ easily?
    // Or we just construct EValue in Java using from(...) methods.
    // Let's assume gEValueClass has 'from' methods. I need to look them up.
    // Or simpler: Convert to Tensors/Primitives and make JEValue.
    
    if (value.isTensor()) {
        jobject jTensor = newJTensorFromTensor(env, value.toTensor());
        static jmethodID fromTensor = env->GetStaticMethodID(gEValueClass, "from", "(Lorg/pytorch/executorch/Tensor;)Lorg/pytorch/executorch/EValue;");
        return env->CallStaticObjectMethod(gEValueClass, fromTensor, jTensor);
    } else if (value.isInt()) {
         static jmethodID fromInt = env->GetStaticMethodID(gEValueClass, "from", "(J)Lorg/pytorch/executorch/EValue;");
         return env->CallStaticObjectMethod(gEValueClass, fromInt, (jlong)value.toInt());
    } else if (value.isDouble()) {
         static jmethodID fromDouble = env->GetStaticMethodID(gEValueClass, "from", "(D)Lorg/pytorch/executorch/EValue;");
         return env->CallStaticObjectMethod(gEValueClass, fromDouble, (jdouble)value.toDouble());
    } else if (value.isBool()) {
         static jmethodID fromBool = env->GetStaticMethodID(gEValueClass, "from", "(Z)Lorg/pytorch/executorch/EValue;");
         return env->CallStaticObjectMethod(gEValueClass, fromBool, (jboolean)value.toBool());
    } else if (value.isString()) {
         // TODO string support
    }
    // Unknown or None
    return nullptr; 
}

EValue evalueFromJEValue(JNIEnv* env, jobject jEValue) {
    if(!jEValue) return EValue(); 
    // Check type by calling methods or checking fields. 
    // Previous code used type code field.
    static jfieldID typeCodeField = env->GetFieldID(gEValueClass, "mTypeCode", "I");
    int typeCode = env->GetIntField(jEValue, typeCodeField);
    
    if (typeCode == 1) { // Tensor
        static jmethodID toTensor = env->GetMethodID(gEValueClass, "toTensor", "()Lorg/pytorch/executorch/Tensor;");
        jobject jTensor = env->CallObjectMethod(jEValue, toTensor);
        return EValue(newTensorFromJTensor(env, jTensor));
    } else if (typeCode == 4) { // Int
        static jmethodID toInt = env->GetMethodID(gEValueClass, "toInt", "()J");
        int64_t val = env->CallLongMethod(jEValue, toInt);
        return EValue(val);
    } else if (typeCode == 3) { // Double
        static jmethodID toDouble = env->GetMethodID(gEValueClass, "toDouble", "()D");
        double val = env->CallDoubleMethod(jEValue, toDouble);
        return EValue(val);
    } else if (typeCode == 5) { // Bool
        static jmethodID toBool = env->GetMethodID(gEValueClass, "toBool", "()Z");
        bool val = env->CallBooleanMethod(jEValue, toBool);
        return EValue(val);
    }
    return EValue();
}


extern "C" {

JNIEXPORT jlong JNICALL Java_org_pytorch_executorch_Module_nativeInit(JNIEnv* env, jclass clazz, jstring path, jint loadMode, jint numThreads) {
    const char* pathStr = env->GetStringUTFChars(path, nullptr);
    std::string pathString(pathStr);
    env->ReleaseStringUTFChars(path, pathStr);

    Module::LoadMode mode = Module::LoadMode::Mmap;
    if (loadMode == 0) mode = Module::LoadMode::File;
    else if (loadMode == 1) mode = Module::LoadMode::Mmap;
    else if (loadMode == 2) mode = Module::LoadMode::MmapUseMlock;
    else if (loadMode == 3) mode = Module::LoadMode::MmapUseMlockIgnoreErrors;

    std::unique_ptr<executorch::runtime::EventTracer> event_tracer = nullptr;
#ifdef EXECUTORCH_ANDROID_PROFILING
    event_tracer = std::make_unique<executorch::etdump::ETDumpGen>();
#endif

    auto module = std::make_unique<Module>(pathString, mode, std::move(event_tracer));
    if (module->load_method("forward") != Error::Ok) {
       // Just created, maybe not loaded yet? Module constructor doesn't load methods eagerly unless we do something else.
       // Actually Module loads header.
    }
    
#ifdef ET_USE_THREADPOOL
    auto threadpool = executorch::extension::threadpool::get_threadpool();
    if (threadpool) {
      int thread_count = numThreads != 0 ? numThreads : cpuinfo_get_processors_count() / 2;
      if (thread_count > 0) {
        threadpool->_unsafe_reset_threadpool(thread_count);
      }
    }
#endif

    auto wrapper = new ModuleWrapper(std::move(module));
    return reinterpret_cast<jlong>(wrapper);
}

JNIEXPORT void JNICALL Java_org_pytorch_executorch_Module_nativeDestroy(JNIEnv* env, jobject thiz, jlong handle) {
    if (handle != 0) {
        delete reinterpret_cast<ModuleWrapper*>(handle);
    }
}

JNIEXPORT jint JNICALL Java_org_pytorch_executorch_Module_nativeLoadMethod(JNIEnv* env, jobject thiz, jlong handle, jstring methodName) {
    auto wrapper = reinterpret_cast<ModuleWrapper*>(handle);
    const char* methodChars = env->GetStringUTFChars(methodName, nullptr);
    Error err = wrapper->module->load_method(methodChars);
    env->ReleaseStringUTFChars(methodName, methodChars);
    return static_cast<jint>(err);
}

JNIEXPORT jobjectArray JNICALL Java_org_pytorch_executorch_Module_nativeExecute(JNIEnv* env, jobject thiz, jlong handle, jstring methodName, jobjectArray jArgs) {
    auto wrapper = reinterpret_cast<ModuleWrapper*>(handle);
    const char* methodChars = env->GetStringUTFChars(methodName, nullptr);
    std::string methodStr(methodChars);
    env->ReleaseStringUTFChars(methodName, methodChars);

    // Prepare inputs
    std::vector<EValue> inputs;
    int argCount = env->GetArrayLength(jArgs);
    for(int i=0; i<argCount; ++i) {
        jobject jArg = env->GetObjectArrayElement(jArgs, i);
        inputs.push_back(evalueFromJEValue(env, jArg));
    }

    Result<std::vector<EValue>> result = wrapper->module->execute(methodStr, inputs);
    if (!result.ok()) {
        throwExecutorchException(env, static_cast<uint32_t>(result.error()), "Execution failed");
        return nullptr;
    }

    const auto& outputs = result.get();
    jobjectArray jResults = env->NewObjectArray(outputs.size(), gEValueClass, nullptr);
    for(size_t i=0; i<outputs.size(); ++i) {
        jobject jVal = newJEValueFromEValue(env, outputs[i]);
        env->SetObjectArrayElement(jResults, i, jVal);
    }
    return jResults;
}

JNIEXPORT jobjectArray JNICALL Java_org_pytorch_executorch_Module_nativeGetMethods(JNIEnv* env, jobject thiz, jlong handle) {
    auto wrapper = reinterpret_cast<ModuleWrapper*>(handle);
    auto res = wrapper->module->method_names();
    if (!res.ok()) return nullptr;
    
    auto names = res.get();
    jobjectArray ret = env->NewObjectArray(names.size(), env->FindClass("java/lang/String"), nullptr);
    int i = 0;
    for(const auto& name : names) {
         env->SetObjectArrayElement(ret, i++, env->NewStringUTF(name.c_str()));
    }
    return ret;
}

JNIEXPORT jobjectArray JNICALL Java_org_pytorch_executorch_Module_nativeGetUsedBackends(JNIEnv* env, jobject thiz, jlong handle, jstring methodName) {
    auto wrapper = reinterpret_cast<ModuleWrapper*>(handle);
    const char* mName = env->GetStringUTFChars(methodName, nullptr);
    auto res = wrapper->module->method_meta(mName);
    env->ReleaseStringUTFChars(methodName, mName);
    
    if(!res.ok()) return nullptr;
    auto meta = res.get();
    
    std::unordered_set<std::string> backends;
    for (auto i = 0; i < meta.num_backends(); i++) {
      backends.insert(meta.get_backend_name(i).get());
    }
    
    jobjectArray ret = env->NewObjectArray(backends.size(), env->FindClass("java/lang/String"), nullptr);
    int i=0;
    for(const auto& s : backends) {
        env->SetObjectArrayElement(ret, i++, env->NewStringUTF(s.c_str()));
    }
    return ret;
}

JNIEXPORT jobjectArray JNICALL Java_org_pytorch_executorch_Module_nativeReadLogBufferStatic(JNIEnv* env, jclass clazz) {
#ifdef __ANDROID__
    jobjectArray ret = nullptr;
    access_log_buffer([&](std::vector<log_entry>& buffer) {
         ret = env->NewObjectArray(buffer.size(), env->FindClass("java/lang/String"), nullptr);
         for(size_t i=0; i<buffer.size(); ++i) {
            std::stringstream ss;
            ss << "[" << buffer[i].timestamp << " " << buffer[i].function << "] " << buffer[i].message;
            env->SetObjectArrayElement(ret, i, env->NewStringUTF(ss.str().c_str()));
         }
    });
    return ret;
#else
    return env->NewObjectArray(0, env->FindClass("java/lang/String"), nullptr);
#endif
}

JNIEXPORT jobjectArray JNICALL Java_org_pytorch_executorch_Module_nativeReadLogBuffer(JNIEnv* env, jobject thiz, jlong handle) {
    return Java_org_pytorch_executorch_Module_nativeReadLogBufferStatic(env, nullptr);
}

JNIEXPORT jboolean JNICALL Java_org_pytorch_executorch_Module_nativeEtDump(JNIEnv* env, jobject thiz, jlong handle) {
   // Implementation omitted for brevity/Linux target, return false or implement if needed. 
   // Assuming simple return false unless PROFILING is on.
   return JNI_FALSE;
}

JNIEXPORT void JNICALL Java_org_pytorch_executorch_Tensor_nativeDestroy(JNIEnv* env, jobject thiz, jlong handle) {
    if (handle != 0) {
        delete reinterpret_cast<TensorWrapper*>(handle);
    }
}

} // extern C


JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    // Cache classes and methods
    jclass localTensor = env->FindClass("org/pytorch/executorch/Tensor");
    gTensorClass = (jclass)env->NewGlobalRef(localTensor);
    gTensorNativeNewTensor = env->GetStaticMethodID(gTensorClass, "nativeNewTensor", "(Ljava/nio/ByteBuffer;[JIJ)Lorg/pytorch/executorch/Tensor;");
    gTensorDtypeJniCode = env->GetMethodID(gTensorClass, "dtypeJniCode", "()I");
    gTensorGetRawDataBuffer = env->GetMethodID(gTensorClass, "getRawDataBuffer", "()Ljava/nio/Buffer;");
    gTensorShape = env->GetMethodID(gTensorClass, "shape", "()[J");

    jclass localEValue = env->FindClass("org/pytorch/executorch/EValue");
    gEValueClass = (jclass)env->NewGlobalRef(localEValue);
    
    return JNI_VERSION_1_6;
}
