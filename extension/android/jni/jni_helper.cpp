/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "jni_helper.h"
#include <executorch/extension/android/jni/log.h>

namespace executorch::jni_helper {

void throwExecutorchException(JNIEnv* env, uint32_t errorCode, const char* details) {
  if (!env) {
    ET_LOG(Error, "JNIEnv is null, cannot throw exception");
    return;
  }

  jclass exceptionClass = env->FindClass("org/pytorch/executorch/ExecutorchRuntimeException");
  if (!exceptionClass) {
    ET_LOG(Error, "Could not find ExecutorchRuntimeException class");
    return;
  }

  jmethodID makeExceptionMethod = env->GetStaticMethodID(
      exceptionClass,
      "makeExecutorchException",
      "(ILjava/lang/String;)Ljava/lang/RuntimeException;");
  
  if (!makeExceptionMethod) {
    ET_LOG(Error, "Could not find makeExecutorchException method");
    return;
  }

  jstring jDetails = env->NewStringUTF(details);
  jobject exception = env->CallStaticObjectMethod(exceptionClass, makeExceptionMethod, (jint)errorCode, jDetails);
  
  if (exception) {
    env->Throw(static_cast<jthrowable>(exception));
  }
}

} // namespace executorch::jni_helper
