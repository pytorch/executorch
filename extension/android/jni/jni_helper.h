/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fbjni/fbjni.h>
#include <string>

namespace executorch::jni_helper {

/**
 * Throws a Java ExecutorchRuntimeException corresponding to the given error
 * code and details. Uses the Java factory method
 * ExecutorchRuntimeException.makeExecutorchException(int, String).
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

} // namespace executorch::jni_helper
