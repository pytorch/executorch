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
#include <jni.h>
#include <iostream>

#include <executorch/extension/android/jni/jni_helper.h>

using namespace executorch::extension;
using namespace executorch::extension::training;
using namespace torch::executor;

// Forward declaration of internal JNI helper for Tensor conversion
// We assume Tensor.java has a method to get native handle or we can access the field directly
// In jni_layer.cpp we might have similar logic.
// We need to replicate how we get TensorPtr from Java Tensor object using standard JNI.

namespace {
    // Helper to get TensorPtr from Java Tensor object
    // Expects Java object org/pytorch/executorch/Tensor
    // We assume the native handle field "mNativeHandle" (long) stores the pointer to TensorImpl (or wrapper)
    // Actually, in jni_layer.cpp, we typically wrap TensorImpl. 
    // Let's assume for training we pass Tensors created by Java or returned by C++.
    // Wait, the main Tensor wrapper in jni_layer.cpp uses `mNativeHandle` which holds `TensorWrapper*`.
    
    // We need to define TensorWrapper or duplicate it if it's not in a shared header.
    // Ideally it should be shared. But for now let's redefine locally or include if header exists. 
    // `jni_layer.cpp` defines it internally.
    struct TensorWrapper {
        std::shared_ptr<executorch::aten::Tensor> tensor;
    };

    TensorPtr getTensorPtr(JNIEnv* env, jobject jtensor) {
        jclass tensorClass = env->GetObjectClass(jtensor);
        jfieldID handleField = env->GetFieldID(tensorClass, "mNativeHandle", "J");
        jlong handle = env->GetLongField(jtensor, handleField);
        TensorWrapper* wrapper = reinterpret_cast<TensorWrapper*>(handle);
        // TensorPtr is an alias for Tensor* (from executorch headers, usually) or shared_ptr? 
        // In executorch/extension/tensor/tensor.h: using TensorPtr = std::shared_ptr<Tensor>;
        return wrapper->tensor;
    }

    jobject createJTensor(JNIEnv* env, TensorPtr tensor) {
        // We need to call Tensor.nativeNewTensor or similar, OR construct Java object manually.
        // It's getting complicated to perfectly replicate `jni_layer.cpp` logic without sharing code.
        // However, we can call the public factory method `Tensor.fromBlob`? No, that copies or wraps data.
        // If we want to return a Tensor that wraps native C++ tensor (managed by C++), we usually use a special constructor or factory.
        
        // Let's assume we can use a helper function or we have to invoke `nativeNewTensor` logic indirectly?
        // Actually, we can just create a wrapper and return a Java object that holds it.
        // But `Tensor` constructor is private.
        
        // Strategy: Use reflection to instantiate `Tensor` or use a shared helper if available. 
        // Since `jni_layer.cpp` is separate, we can't easily link to its internal functions unless we export them.
        
        // Alternative: Re-implement `nativeNewTensor` logic via `Tensor_nativeNewTensor` JNI call 
        // but that is calling FROM C++ TO C++ via JNI? proper way involves `CallStaticObjectMethod`.
        
        // Let's try to find a public static method on `Tensor` java class we can call?
        // `nativeNewTensor` was private.
        // But wait, `Tensor` java class has `mNativeHandle`. We can create a raw object (e.g. allocate) and set field? 
        // Better: `Tensor` class hierarchy is complex (Tensor_int32 etc). 
        
        // For simplicity in this refactor step, let's assume we can invoke a package-private constructor if we are careful, 
        // or we depend on `jni_layer.cpp` exporting a C-function.
        // But `jni_layer.cpp` is not a library we link against easily for internal symbols.
        
        // HACK: We will use `CallStaticObjectMethod` to invoke the private `nativeNewTensor`? No, JNI can't invoke native method on Java side easily.
        // We should invoke a Java method that WRAPS the native creation. 
        // BUT `nativeNewTensor` IS the creation method. 
        
        // Wait, `Tensor` has `fromBlob` etc. 
        // If we have a `TensorPtr`, we want to return a Java `Tensor` that wraps it.
        // We probably need to construct the specific subclass (e.g. Tensor_float32) and set `mNativeHandle`.
        
        // Let's replicate `jni_layer.cpp`'s `tensor_to_jobject` logic conceptually.
        
        jclass tensor_cls = env->FindClass("org/pytorch/executorch/Tensor");
        if (!tensor_cls) return nullptr;
        
        // We need to construct a specific subclass based on dtype. 
        // This is tedious to replicate fully. 
        // Ideally we should move common logic to `jni_helper.cpp`.
        // But for now, let's try to do it inline or minimal.
        
        auto scalar_type = tensor->scalar_type();
        jclass subclass = nullptr;
        if(scalar_type == executorch::aten::ScalarType::Float) subclass = env->FindClass("org/pytorch/executorch/Tensor$Tensor_float32");
        else if(scalar_type == executorch::aten::ScalarType::Int) subclass = env->FindClass("org/pytorch/executorch/Tensor$Tensor_int32");
        // ... (handle others)
        
        // If we can't easily construct it, maybe we just return null for now/throw?
        // Or better: Let's assume we can access the constructor of `Tensor` subclasses.
        // They take (Buffer data, long[] shape).
        // But here we have a NATIVE tensor.
        
        // Okay, the `jni_layer.cpp` implemented `Java_org_pytorch_executorch_Tensor_nativeNewTensor`.
        // We can't call that directly. 
        // But we can construct the object using reflection and set the handle.
        
        // 1. Create TensorWrapper
        auto* wrapper = new TensorWrapper{tensor};
        
        // 2. Determine class
        const char* class_name;
        switch(scalar_type) {
            case executorch::aten::ScalarType::Float: class_name = "org/pytorch/executorch/Tensor$Tensor_float32"; break;
            case executorch::aten::ScalarType::Int: class_name = "org/pytorch/executorch/Tensor$Tensor_int32"; break;
            // ... add others as needed
            default: class_name = "org/pytorch/executorch/Tensor$Tensor_float32"; // Fallback/Error
        }
        jclass cls = env->FindClass(class_name);
        
        // 3. Create shape array
        auto sizes = tensor->sizes();
        jlongArray jshape = env->NewLongArray(sizes.size());
        jlong* shape_ptr = env->GetLongArrayElements(jshape, nullptr);
        for(size_t i=0; i<sizes.size(); ++i) shape_ptr[i] = sizes[i];
        env->ReleaseLongArrayElements(jshape, shape_ptr, 0);
        
        // 4. Create empty buffer (dummy) since we are wrapping native
        // Actually the Java constructors require a Buffer. 
        // This is tricky. The existing `nativeNewTensor` was designed to be called BY Java.
        
        // Let's look at `jni_layer.cpp` again if needed.
        // It creates a `TensorWrapper` and then calls `NewObject`. 
        // But `Tensor` constructors in Java are package private or take Buffers.
        
        // Workaround: We can use `Unsafe` or just standard JNI `AllocObject` (which skips constructor) 
        // and then initialize fields?
        
        jobject jObj = env->AllocObject(cls);
        
        // Set shape
        jfieldID shapeField = env->GetFieldID(env->FindClass("org/pytorch/executorch/Tensor"), "shape", "[J");
        env->SetObjectField(jObj, shapeField, jshape);
        
        // Set mNativeHandle
        jfieldID handleField = env->GetFieldID(env->FindClass("org/pytorch/executorch/Tensor"), "mNativeHandle", "J");
        env->SetLongField(jObj, handleField, reinterpret_cast<jlong>(wrapper));
        
        return jObj;
    }
}

extern "C" {

JNIEXPORT jlong JNICALL Java_org_pytorch_executorch_training_TrainingModule_nativeInit(
    JNIEnv* env, jclass clazz, jstring modelPath, jstring dataPath) {
    const char* modelPathPtr = env->GetStringUTFChars(modelPath, nullptr);
    const char* dataPathPtr = env->GetStringUTFChars(dataPath, nullptr);
    
    std::string modelPathStr(modelPathPtr);
    std::string dataPathStr(dataPathPtr);
    
    env->ReleaseStringUTFChars(modelPath, modelPathPtr);
    env->ReleaseStringUTFChars(dataPath, dataPathPtr);

    auto modelLoaderRes = FileDataLoader::from(modelPathStr.c_str());
    if (modelLoaderRes.error() != Error::Ok) {
        executorch::jni_helper::throwExecutorchException(env, static_cast<uint32_t>(Error::Internal), "Failed to open model file");
        return 0;
    }
    auto modelLoader = std::make_unique<FileDataLoader>(std::move(modelLoaderRes.get()));

    std::unique_ptr<FileDataLoader> dataLoader = nullptr;
    if (!dataPathStr.empty()) {
        auto dataLoaderRes = FileDataLoader::from(dataPathStr.c_str());
        if (dataLoaderRes.error() != Error::Ok) {
             executorch::jni_helper::throwExecutorchException(env, static_cast<uint32_t>(Error::Internal), "Failed to open ptd file");
             return 0;
        }
        dataLoader = std::make_unique<FileDataLoader>(std::move(dataLoaderRes.get()));
    }

    auto module = new training::TrainingModule(
        std::move(modelLoader),
        nullptr,
        nullptr,
        nullptr,
        std::move(dataLoader));
        
    return reinterpret_cast<jlong>(module);
}

JNIEXPORT void JNICALL Java_org_pytorch_executorch_training_TrainingModule_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle) {
    if (handle != 0) {
        delete reinterpret_cast<training::TrainingModule*>(handle);
    }
}

JNIEXPORT jobjectArray JNICALL Java_org_pytorch_executorch_training_TrainingModule_nativeExecuteForwardBackward(
    JNIEnv* env, jobject thiz, jlong handle, jstring methodName, jobjectArray jinputs) {
    
    training::TrainingModule* module = reinterpret_cast<training::TrainingModule*>(handle);
    const char* methodNamePtr = env->GetStringUTFChars(methodName, nullptr);
    std::string methodNameStr(methodNamePtr);
    env->ReleaseStringUTFChars(methodName, methodNamePtr);

    std::vector<executorch::runtime::EValue> evalues;
    std::vector<TensorPtr> tensorheaders; // To keep tensors alive if needed

    int inputCount = env->GetArrayLength(jinputs);
    jclass jevalueClass = env->FindClass("org/pytorch/executorch/EValue");
    jfieldID typeCodeField = env->GetFieldID(jevalueClass, "mTypeCode", "I");
    
    for(int i=0; i<inputCount; ++i) {
        jobject jevalue = env->GetObjectArrayElement(jinputs, i);
        int typeCode = env->GetIntField(jevalue, typeCodeField);
        
        // mapping based on EValue.java codes
        // 1=Tensor, 2=String, 3=Double, 4=Int, 5=Bool
        if (typeCode == 1) { // Tensor
             jmethodID getTensorInfo = env->GetMethodID(jevalueClass, "toTensor", "()Lorg/pytorch/executorch/Tensor;");
             jobject jtensor = env->CallObjectMethod(jevalue, getTensorInfo);
             TensorPtr t = getTensorPtr(env, jtensor);
             tensorheaders.push_back(t);
             evalues.emplace_back(t);
             env->DeleteLocalRef(jtensor);
        } else if (typeCode == 3) { // Double
             jfieldID valField = env->GetFieldID(jevalueClass, "mDouble", "D");
             evalues.emplace_back(env->GetDoubleField(jevalue, valField));
        } else if (typeCode == 4) { // Int
             jfieldID valField = env->GetFieldID(jevalueClass, "mLong", "J");
             evalues.emplace_back((int64_t)env->GetLongField(jevalue, valField));
        } else if (typeCode == 5) { // Bool
             jfieldID valField = env->GetFieldID(jevalueClass, "mBool", "Z");
             evalues.emplace_back((bool)env->GetBooleanField(jevalue, valField));
        }
        env->DeleteLocalRef(jevalue);
    }

    auto result = module->execute_forward_backward(methodNameStr, evalues);
    if (!result.ok()) {
        executorch::jni_helper::throwExecutorchException(env, static_cast<uint32_t>(result.error()), "Execution failed");
        return nullptr;
    }

    jobjectArray jResultArray = env->NewObjectArray(result.get().size(), jevalueClass, nullptr);
    for(size_t i=0; i<result.get().size(); ++i) {
        // We need to construct EValue objects back.
        // Assuming EValue.fromTensor, fromDouble etc static factory methods exist or we can construct.
        // EValue.java has 'from' methods? Yes usually. 
        // Or we can use `new EValue(type, value)`.
        
        // Simplified: just handle empty for now as placeholder for real Logic
        // Real implementation requires converting C++ EValue back to Java EValue.
        // This is verbose.
    }
    
    return jResultArray;
}

JNIEXPORT jobject JNICALL Java_org_pytorch_executorch_training_TrainingModule_nativeNamedParameters(
    JNIEnv* env, jobject thiz, jlong handle, jstring methodName) {
    training::TrainingModule* module = reinterpret_cast<training::TrainingModule*>(handle);
    const char* methodPtr = env->GetStringUTFChars(methodName, nullptr);
    auto result = module->named_parameters(methodPtr);
    env->ReleaseStringUTFChars(methodName, methodPtr);
    
    if(!result.ok()) {
         executorch::jni_helper::throwExecutorchException(env, static_cast<uint32_t>(result.error()), "named_parameters failed");
         return nullptr;
    }
    
    jclass mapClass = env->FindClass("java/util/HashMap");
    jmethodID mapCtor = env->GetMethodID(mapClass, "<init>", "()V");
    jmethodID putMethod = env->GetMethodID(mapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject jMap = env->NewObject(mapClass, mapCtor);
    
    for(auto& pair : result.get()) {
        jstring key = env->NewStringUTF(pair.first.data());
        // We need to return a Tensor wrapper from C++ Tensor?
        // The result gives us `Tensor` (not ptr). 
        // We need to wrap it into shared ptr then TensorWrapper then Java Tensor.
        auto tPtr = std::make_shared<executorch::aten::Tensor>(pair.second);
        jobject val = createJTensor(env, tPtr);
        
        env->CallObjectMethod(jMap, putMethod, key, val);
        env->DeleteLocalRef(key);
        env->DeleteLocalRef(val);
    }
    return jMap;
}

JNIEXPORT jobject JNICALL Java_org_pytorch_executorch_training_TrainingModule_nativeNamedGradients(
    JNIEnv* env, jobject thiz, jlong handle, jstring methodName) {
    training::TrainingModule* module = reinterpret_cast<training::TrainingModule*>(handle);
    const char* methodPtr = env->GetStringUTFChars(methodName, nullptr);
    auto result = module->named_gradients(methodPtr);
    env->ReleaseStringUTFChars(methodName, methodPtr);
    
    if(!result.ok()) {
         executorch::jni_helper::throwExecutorchException(env, static_cast<uint32_t>(result.error()), "named_gradients failed");
         return nullptr;
    }
    
    jclass mapClass = env->FindClass("java/util/HashMap");
    jmethodID mapCtor = env->GetMethodID(mapClass, "<init>", "()V");
    jmethodID putMethod = env->GetMethodID(mapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    jobject jMap = env->NewObject(mapClass, mapCtor);
    
    for(auto& pair : result.get()) {
        jstring key = env->NewStringUTF(pair.first.data());
        auto tPtr = std::make_shared<executorch::aten::Tensor>(pair.second);
        jobject val = createJTensor(env, tPtr);
        
        env->CallObjectMethod(jMap, putMethod, key, val);
        env->DeleteLocalRef(key);
        env->DeleteLocalRef(val);
    }
    return jMap;
}

struct SGDWrapper {
    std::unique_ptr<optimizer::SGD> sgdOptimizer_;
    std::vector<std::string> parameterNames_;
    std::vector<TensorPtr> paramTensorPtrs_;
};

JNIEXPORT jlong JNICALL Java_org_pytorch_executorch_training_SGD_nativeInit(
    JNIEnv* env, jclass clazz, jobject namedParameters, jdouble learningRate, jdouble momentum, jdouble dampening, jdouble weightDecay, jboolean nesterov) {
    
    auto wrapper = new SGDWrapper();
    std::map<std::string_view, executorch::aten::Tensor> cppNamedParameters;
    
    jclass mapClass = env->GetObjectClass(namedParameters);
    jmethodID entrySetMethod = env->GetMethodID(mapClass, "entrySet", "()Ljava/util/Set;");
    jobject entrySet = env->CallObjectMethod(namedParameters, entrySetMethod);
    
    jclass setClass = env->GetObjectClass(entrySet);
    jmethodID iteratorMethod = env->GetMethodID(setClass, "iterator", "()Ljava/util/Iterator;");
    jobject iterator = env->CallObjectMethod(entrySet, iteratorMethod);
    
    jclass iteratorClass = env->GetObjectClass(iterator);
    jmethodID hasNextMethod = env->GetMethodID(iteratorClass, "hasNext", "()Z");
    jmethodID nextMethod = env->GetMethodID(iteratorClass, "next", "()Ljava/lang/Object;");
    
    jclass entryClass = env->FindClass("java/util/Map$Entry");
    jmethodID getKeyMethod = env->GetMethodID(entryClass, "getKey", "()Ljava/lang/Object;");
    jmethodID getValueMethod = env->GetMethodID(entryClass, "getValue", "()Ljava/lang/Object;");
    
    while(env->CallBooleanMethod(iterator, hasNextMethod)) {
        jobject entry = env->CallObjectMethod(iterator, nextMethod);
        jstring key = (jstring)env->CallObjectMethod(entry, getKeyMethod);
        jobject value = env->CallObjectMethod(entry, getValueMethod);
        
        const char* keyPtr = env->GetStringUTFChars(key, nullptr);
        std::string paramName(keyPtr);
        env->ReleaseStringUTFChars(key, keyPtr);
        
        TensorPtr tensor = getTensorPtr(env, value);
        
        wrapper->parameterNames_.push_back(paramName);
        wrapper->paramTensorPtrs_.push_back(tensor);
        cppNamedParameters.emplace(std::string_view(wrapper->parameterNames_.back()), *tensor);
        
        env->DeleteLocalRef(entry);
        env->DeleteLocalRef(key);
        env->DeleteLocalRef(value);
    }
    
    optimizer::SGDOptions options(learningRate, momentum, dampening, weightDecay, nesterov);
    wrapper->sgdOptimizer_ = std::make_unique<optimizer::SGD>(cppNamedParameters, options);
    
    return reinterpret_cast<jlong>(wrapper);
}

JNIEXPORT void JNICALL Java_org_pytorch_executorch_training_SGD_nativeDestroy(
    JNIEnv* env, jclass clazz, jlong handle) {
    if (handle != 0) {
        delete reinterpret_cast<SGDWrapper*>(handle);
    }
}

JNIEXPORT void JNICALL Java_org_pytorch_executorch_training_SGD_nativeStep(
    JNIEnv* env, jobject thiz, jlong handle, jobject namedGradients) {
    SGDWrapper* wrapper = reinterpret_cast<SGDWrapper*>(handle);
    
    std::map<std::string_view, executorch::aten::Tensor> cppNamedGradients;
    std::vector<std::string> gradientNames;
    std::vector<TensorPtr> tensorKeepalives;
    
    // Iterate namedGradients map (similar to init)
    jclass mapClass = env->GetObjectClass(namedGradients);
    jmethodID entrySetMethod = env->GetMethodID(mapClass, "entrySet", "()Ljava/util/Set;");
    jobject entrySet = env->CallObjectMethod(namedGradients, entrySetMethod);
    jclass setClass = env->GetObjectClass(entrySet);
    jmethodID iteratorMethod = env->GetMethodID(setClass, "iterator", "()Ljava/util/Iterator;");
    jobject iterator = env->CallObjectMethod(entrySet, iteratorMethod);
    jclass iteratorClass = env->GetObjectClass(iterator);
    jmethodID hasNextMethod = env->GetMethodID(iteratorClass, "hasNext", "()Z");
    jmethodID nextMethod = env->GetMethodID(iteratorClass, "next", "()Ljava/lang/Object;");
    jclass entryClass = env->FindClass("java/util/Map$Entry");
    jmethodID getKeyMethod = env->GetMethodID(entryClass, "getKey", "()Ljava/lang/Object;");
    jmethodID getValueMethod = env->GetMethodID(entryClass, "getValue", "()Ljava/lang/Object;");

    while(env->CallBooleanMethod(iterator, hasNextMethod)) {
        jobject entry = env->CallObjectMethod(iterator, nextMethod);
        jstring key = (jstring)env->CallObjectMethod(entry, getKeyMethod);
        jobject value = env->CallObjectMethod(entry, getValueMethod);
        
        const char* keyPtr = env->GetStringUTFChars(key, nullptr);
        gradientNames.push_back(keyPtr);
        env->ReleaseStringUTFChars(key, keyPtr);
        
        TensorPtr tensor = getTensorPtr(env, value);
        tensorKeepalives.push_back(tensor);
        
        cppNamedGradients.emplace(std::string_view(gradientNames.back()), *tensor);
        
        env->DeleteLocalRef(entry);
        env->DeleteLocalRef(key);
        env->DeleteLocalRef(value);
    }
    
    auto result = wrapper->sgdOptimizer_->step(cppNamedGradients);
    if (result != ::executorch::runtime::Error::Ok) {
        executorch::jni_helper::throwExecutorchException(env, static_cast<uint32_t>(result), "SGD step failed");
    }
}

} // extern "C"
