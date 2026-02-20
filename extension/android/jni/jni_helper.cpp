/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "jni_helper.h"

namespace executorch::jni_helper {

void throwExecutorchException(
    JNIEnv* env,
    uint32_t errorCode,
    const std::string& details) {
  if (!env) {
    return;
  }

  // Find the exception class
  jclass exceptionClass =
      env->FindClass("org/pytorch/executorch/ExecutorchRuntimeException");
  if (exceptionClass == nullptr) {
    // Class not found, clear the exception and return
    env->ExceptionClear();
    return;
  }

  // Find the static factory method: makeExecutorchException(int, String)
  jmethodID makeExceptionMethod = env->GetStaticMethodID(
      exceptionClass,
      "makeExecutorchException",
      "(ILjava/lang/String;)Ljava/lang/RuntimeException;");
  if (makeExceptionMethod == nullptr) {
    env->ExceptionClear();
    env->DeleteLocalRef(exceptionClass);
    return;
  }

  // Create the details string
  jstring jDetails = env->NewStringUTF(details.c_str());
  if (jDetails == nullptr) {
    env->ExceptionClear();
    env->DeleteLocalRef(exceptionClass);
    return;
  }

  // Call the factory method to create the exception object
  jobject exception = env->CallStaticObjectMethod(
      exceptionClass,
      makeExceptionMethod,
      static_cast<jint>(errorCode),
      jDetails);

  env->DeleteLocalRef(jDetails);

  if (exception != nullptr) {
    env->Throw(static_cast<jthrowable>(exception));
    env->DeleteLocalRef(exception);
  }

  env->DeleteLocalRef(exceptionClass);
}

#if EXECUTORCH_HAS_FBJNI
void throwExecutorchException(uint32_t errorCode, const std::string& details) {
  // Get the current JNI environment
  auto env = facebook::jni::Environment::current();
  if (!env) {
    return;
  }

  // stable/global class ref — safe to cache
  static const auto exceptionClass =
      JExecutorchRuntimeException::javaClassStatic();

  // Find the static factory method: makeExecutorchException(int, String)
  static auto makeExceptionMethod =
      exceptionClass
          ->getStaticMethod<facebook::jni::local_ref<facebook::jni::JThrowable>(
              int, facebook::jni::alias_ref<facebook::jni::JString>)>(
              "makeExecutorchException",
              "(ILjava/lang/String;)Ljava/lang/RuntimeException;");

  auto jDetails = facebook::jni::make_jstring(details);
  // Call the factory method to create the exception object
  auto exception = makeExceptionMethod(exceptionClass, errorCode, jDetails);
  facebook::jni::throwNewJavaException(exception.get());
}
#endif

} // namespace executorch::jni_helper
