/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "jni_helper.h"

namespace executorch::jni_helper {

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

void setExecutorchPendingException(
    JNIEnv* env,
    uint32_t errorCode,
    const std::string& details) {
  if (!env) {
    return;
  }
  if (env->ExceptionCheck()) {
    // Preserve any preexisting pending exception; do not overwrite it here.
    return;
  }

  // If an exception is already pending, preserve it and do not overwrite.
  if (env->ExceptionCheck()) {
    return;
  }

  jclass exceptionClass =
      env->FindClass("org/pytorch/executorch/ExecutorchRuntimeException");
  if (env->ExceptionCheck()) {
    if (exceptionClass) {
      env->DeleteLocalRef(exceptionClass);
    }
    // Preserve the original exception; do not clear or overwrite it.
    return;
  }

  if (!exceptionClass) {
    // FindClass failed. It should have set a pending exception; leave it as is.
    return;
  }

  jmethodID factoryMethod = env->GetStaticMethodID(
      exceptionClass,
      "makeExecutorchException",
      "(ILjava/lang/String;)Ljava/lang/RuntimeException;");
  if (env->ExceptionCheck()) {
    // Preserve the original exception; do not overwrite it.
    env->DeleteLocalRef(exceptionClass);
    return;
  }

  if (!factoryMethod) {
    // If the factory method cannot be found but no exception is pending,
    // fall back to throwing the base Executorch exception.
    env->ThrowNew(exceptionClass, details.c_str());
    env->DeleteLocalRef(exceptionClass);
    return;
  }

  jstring jDetails = env->NewStringUTF(details.c_str());
  if (env->ExceptionCheck()) {
    // Preserve the original exception; do not overwrite it.
    if (jDetails) {
      env->DeleteLocalRef(jDetails);
    }
    env->DeleteLocalRef(exceptionClass);
    return;
  }

  if (!jDetails) {
    // NewStringUTF returned null without setting an exception; fall back.
    env->ThrowNew(exceptionClass, details.c_str());
    env->DeleteLocalRef(exceptionClass);
    return;
  }

  auto exception = static_cast<jthrowable>(env->CallStaticObjectMethod(
      exceptionClass, factoryMethod, static_cast<jint>(errorCode), jDetails));
  if (env->ExceptionCheck() || !exception) {
    // If a Java exception was thrown, it is already pending; just clean up.
    if (exception) {
      env->DeleteLocalRef(exception);
    }
    env->DeleteLocalRef(jDetails);
    env->DeleteLocalRef(exceptionClass);
    return;
  }

  env->Throw(exception);
  env->DeleteLocalRef(exception);
  env->DeleteLocalRef(jDetails);
  env->DeleteLocalRef(exceptionClass);
}

bool utf8_check_validity(const char* str, size_t length) {
  for (size_t i = 0; i < length; ++i) {
    uint8_t byte = static_cast<uint8_t>(str[i]);
    if (byte >= 0x80) {
      if (i + 1 >= length) {
        return false;
      }
      uint8_t next_byte = static_cast<uint8_t>(str[i + 1]);
      if ((byte & 0xE0) == 0xC0 && (next_byte & 0xC0) == 0x80) {
        i += 1;
      } else if (
          (byte & 0xF0) == 0xE0 && (next_byte & 0xC0) == 0x80 &&
          (i + 2 < length) &&
          (static_cast<uint8_t>(str[i + 2]) & 0xC0) == 0x80) {
        i += 2;
      } else if (
          (byte & 0xF8) == 0xF0 && (next_byte & 0xC0) == 0x80 &&
          (i + 2 < length) &&
          (static_cast<uint8_t>(str[i + 2]) & 0xC0) == 0x80 &&
          (i + 3 < length) &&
          (static_cast<uint8_t>(str[i + 3]) & 0xC0) == 0x80) {
        i += 3;
      } else {
        return false;
      }
    }
  }
  return true;
}

} // namespace executorch::jni_helper
