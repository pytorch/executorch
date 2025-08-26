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

  // Find the Java ExecutorchRuntimeException class
  static auto exceptionClass = facebook::jni::findClassLocal(
      "org/pytorch/executorch/ExecutorchRuntimeException");

  // Find the static factory method: makeExecutorchException(int, String)
  static auto makeExceptionMethod = exceptionClass->getStaticMethod<
      facebook::jni::local_ref<facebook::jni::JThrowable>(
          int, facebook::jni::alias_ref<facebook::jni::JString>)>(
      "makeExecutorchException",
      "(ILjava/lang/String;)Lorg/pytorch/executorch/ExecutorchRuntimeException;");

  auto jDetails = facebook::jni::make_jstring(details);
  // Call the factory method to create the exception object
  auto exception = makeExceptionMethod(exceptionClass, errorCode, jDetails);
  facebook::jni::throwNewJavaException(exception.get());
}

} // namespace executorch::jni_helper
