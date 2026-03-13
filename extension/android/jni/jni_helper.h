/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fbjni/fbjni.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace executorch::jni_helper {

/**
 * Throws a Java ExecutorchRuntimeException corresponding to the given error
 * code and details. Uses the Java factory method
 * ExecutorchRuntimeException.makeExecutorchException(int, String).
 *
 * IMPORTANT: This throws a C++ exception (via fbjni). Only use in fbjni
 * HybridClass methods where fbjni catches it at the JNI boundary.
 * For plain extern "C" JNIEXPORT functions, use setExecutorchPendingException.
 *
 * @param errorCode The error code from the C++ Executorch runtime.
 * @param details Additional details to include in the exception message.
 */
void throwExecutorchException(uint32_t errorCode, const std::string& details);

/**
 * Sets a pending Java ExecutorchRuntimeException without throwing a C++
 * exception. Safe to call from plain extern "C" JNIEXPORT functions.
 * After calling this, the caller must return from the JNI function promptly;
 * the Java exception will be delivered when control returns to the JVM.
 *
 * @param env The JNI environment pointer.
 * @param errorCode The error code from the C++ Executorch runtime.
 * @param details Additional details to include in the exception message.
 */
void setExecutorchPendingException(
    JNIEnv* env,
    uint32_t errorCode,
    const std::string& details);

// Define the JavaClass wrapper
struct JExecutorchRuntimeException
    : public facebook::jni::JavaClass<JExecutorchRuntimeException> {
  static constexpr auto kJavaDescriptor =
      "Lorg/pytorch/executorch/ExecutorchRuntimeException;";
};

/**
 * Returns true if the given byte sequence is valid UTF-8.
 */
bool utf8_check_validity(const char* str, size_t length);

} // namespace executorch::jni_helper
