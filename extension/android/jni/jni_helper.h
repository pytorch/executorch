/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <jni.h>
#include <string>

#if __has_include(<fbjni/fbjni.h>)
#include <fbjni/fbjni.h>
#define EXECUTORCH_HAS_FBJNI 1
#else
#define EXECUTORCH_HAS_FBJNI 0
#endif

namespace executorch::jni_helper {

/**
 * Throws a Java ExecutorchRuntimeException corresponding to the given error
 * code and details. Uses the Java factory method
 * ExecutorchRuntimeException.makeExecutorchException(int, String).
 *
 * This version takes JNIEnv* directly and works with pure JNI.
 *
 * @param env The JNI environment.
 * @param errorCode The error code from the C++ Executorch runtime.
 * @param details Additional details to include in the exception message.
 */
void throwExecutorchException(
    JNIEnv* env,
    uint32_t errorCode,
    const std::string& details);

#if EXECUTORCH_HAS_FBJNI
/**
 * Throws a Java ExecutorchRuntimeException corresponding to the given error
 * code and details. Uses the Java factory method
 * ExecutorchRuntimeException.makeExecutorchException(int, String).
 *
 * This version uses fbjni to get the current JNI environment.
 *
 * @param errorCode The error code from the C++ Executorch runtime.
 * @param details Additional details to include in the exception message.
 */
void throwExecutorchException(uint32_t errorCode, const std::string& details);

// Define the JavaClass wrapper
struct JExecutorchRuntimeException
    : public facebook::jni::JavaClass<JExecutorchRuntimeException> {
  static constexpr auto kJavaDescriptor =
      "Lorg/pytorch/executorch/ExecutorchRuntimeException;";
};
#endif

} // namespace executorch::jni_helper
