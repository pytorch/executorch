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
#include <executorch/runtime/platform/log.h>

namespace executorch_jni {
namespace runtime = ::executorch::ET_RUNTIME_NAMESPACE;

} // namespace executorch_jni

extern "C" {

JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_ExecuTorchRuntime_nativeGetRegisteredOps(
    JNIEnv* env,
    jclass /* clazz */) {
  auto kernels = executorch_jni::runtime::get_registered_kernels();

  jclass stringClass = env->FindClass("java/lang/String");
  jobjectArray result = env->NewObjectArray(kernels.size(), stringClass, nullptr);

  for (size_t i = 0; i < kernels.size(); ++i) {
    jstring op = env->NewStringUTF(kernels[i].name_);
    env->SetObjectArrayElement(result, i, op);
    env->DeleteLocalRef(op);
  }

  env->DeleteLocalRef(stringClass);
  return result;
}

JNIEXPORT jobjectArray JNICALL
Java_org_pytorch_executorch_ExecuTorchRuntime_nativeGetRegisteredBackends(
    JNIEnv* env,
    jclass /* clazz */) {
  int num_backends = executorch_jni::runtime::get_num_registered_backends();

  jclass stringClass = env->FindClass("java/lang/String");
  jobjectArray result = env->NewObjectArray(num_backends, stringClass, nullptr);

  for (int i = 0; i < num_backends; ++i) {
    auto name_result = executorch_jni::runtime::get_backend_name(i);
    const char* name = "";

    if (name_result.ok()) {
      name = *name_result;
    }

    jstring backend_str = env->NewStringUTF(name);
    env->SetObjectArrayElement(result, i, backend_str);
    env->DeleteLocalRef(backend_str);
  }

  env->DeleteLocalRef(stringClass);
  return result;
}

} // extern "C"

void register_natives_for_runtime(JNIEnv* env) {
  jclass runtime_class = env->FindClass("org/pytorch/executorch/ExecuTorchRuntime");
  if (runtime_class == nullptr) {
    ET_LOG(Error, "Failed to find ExecuTorchRuntime class");
    env->ExceptionClear();
    return;
  }

  // clang-format off
  static const JNINativeMethod methods[] = {
      {"nativeGetRegisteredOps", "()[Ljava/lang/String;",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_ExecuTorchRuntime_nativeGetRegisteredOps)},
      {"nativeGetRegisteredBackends", "()[Ljava/lang/String;",
       reinterpret_cast<void*>(
           Java_org_pytorch_executorch_ExecuTorchRuntime_nativeGetRegisteredBackends)},
  };
  // clang-format on

  int num_methods = sizeof(methods) / sizeof(methods[0]);
  int result = env->RegisterNatives(runtime_class, methods, num_methods);
  if (result != JNI_OK) {
    ET_LOG(Error, "Failed to register native methods for ExecuTorchRuntime");
  }

  env->DeleteLocalRef(runtime_class);
}
