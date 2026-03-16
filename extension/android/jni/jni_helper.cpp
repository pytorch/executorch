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
