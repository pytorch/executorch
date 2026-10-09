/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <jni.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/kernel/operator_registry.h>

#ifndef EXECUTORCH_USE_GENERIC_JNI
#include <fbjni/fbjni.h>
#endif

namespace runtime = ::executorch::ET_RUNTIME_NAMESPACE;

#ifdef EXECUTORCH_USE_GENERIC_JNI

namespace {

jobjectArray new_string_array(JNIEnv* env, jsize size) {
  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    return nullptr;
  }

  jobjectArray result = env->NewObjectArray(size, string_class, nullptr);
  env->DeleteLocalRef(string_class);
  return result;
}

bool set_string_array_element(
    JNIEnv* env,
    jobjectArray array,
    jsize index,
    const char* value) {
  jstring string = env->NewStringUTF(value);
  if (string == nullptr) {
    return false;
  }

  env->SetObjectArrayElement(array, index, string);
  env->DeleteLocalRef(string);
  return !env->ExceptionCheck();
}

} // namespace

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_ExecuTorchRuntime_getRegisteredOps(
    JNIEnv* env,
    jclass /* clazz */) {
  auto kernels = runtime::get_registered_kernels();
  auto result = new_string_array(env, static_cast<jsize>(kernels.size()));
  if (result == nullptr) {
    return nullptr;
  }

  for (size_t i = 0; i < kernels.size(); ++i) {
    if (!set_string_array_element(
            env, result, static_cast<jsize>(i), kernels[i].name_)) {
      return nullptr;
    }
  }

  return result;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_ExecuTorchRuntime_getRegisteredBackends(
    JNIEnv* env,
    jclass /* clazz */) {
  const int num_backends = runtime::get_num_registered_backends();
  auto result = new_string_array(env, static_cast<jsize>(num_backends));
  if (result == nullptr) {
    return nullptr;
  }

  for (int i = 0; i < num_backends; ++i) {
    auto name_result = runtime::get_backend_name(i);
    const char* name = "";

    if (name_result.ok()) {
      name = *name_result;
    }

    if (!set_string_array_element(env, result, static_cast<jsize>(i), name)) {
      return nullptr;
    }
  }

  return result;
}

void register_natives_for_runtime() {}

#else

namespace executorch_jni {

class AndroidRuntimeJni : public facebook::jni::JavaClass<AndroidRuntimeJni> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/ExecuTorchRuntime;";

  static void registerNatives() {
    javaClassStatic()->registerNatives({
        makeNativeMethod(
            "getRegisteredOps", AndroidRuntimeJni::getRegisteredOps),
        makeNativeMethod(
            "getRegisteredBackends", AndroidRuntimeJni::getRegisteredBackends),
    });
  }

  static facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>>
  getRegisteredOps(facebook::jni::alias_ref<jclass>) {
    auto kernels = runtime::get_registered_kernels();
    auto result = facebook::jni::JArrayClass<jstring>::newArray(kernels.size());

    for (size_t i = 0; i < kernels.size(); ++i) {
      auto op = facebook::jni::make_jstring(kernels[i].name_);
      result->setElement(i, op.get());
    }

    return result;
  }

  static facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>>
  getRegisteredBackends(facebook::jni::alias_ref<jclass>) {
    int num_backends = runtime::get_num_registered_backends();
    auto result = facebook::jni::JArrayClass<jstring>::newArray(num_backends);

    for (int i = 0; i < num_backends; ++i) {
      auto name_result = runtime::get_backend_name(i);
      const char* name = "";

      if (name_result.ok()) {
        name = *name_result;
      }

      auto backend_str = facebook::jni::make_jstring(name);
      result->setElement(i, backend_str.get());
    }

    return result;
  }
};

} // namespace executorch_jni

void register_natives_for_runtime() {
  executorch_jni::AndroidRuntimeJni::registerNatives();
}

#endif
