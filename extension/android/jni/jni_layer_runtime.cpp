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

namespace executorch_jni {
namespace runtime = ::executorch::ET_RUNTIME_NAMESPACE;
} // namespace executorch_jni

extern "C" {

JNIEXPORT jobjectArray JNICALL Java_org_pytorch_executorch_ExecuTorchRuntime_getRegisteredOps(JNIEnv* env, jclass clazz) {
    auto kernels = executorch_jni::runtime::get_registered_kernels();
    jobjectArray result = env->NewObjectArray(kernels.size(), env->FindClass("java/lang/String"), nullptr);

    for (size_t i = 0; i < kernels.size(); ++i) {
        jstring op = env->NewStringUTF(kernels[i].name_);
        env->SetObjectArrayElement(result, i, op);
    }
    return result;
}

JNIEXPORT jobjectArray JNICALL Java_org_pytorch_executorch_ExecuTorchRuntime_getRegisteredBackends(JNIEnv* env, jclass clazz) {
    int num_backends = executorch_jni::runtime::get_num_registered_backends();
    jobjectArray result = env->NewObjectArray(num_backends, env->FindClass("java/lang/String"), nullptr);

    for (int i = 0; i < num_backends; ++i) {
        auto name_result = executorch_jni::runtime::get_backend_name(i);
        const char* name = "";
        if (name_result.ok()) {
            name = *name_result;
        }
        jstring backend_str = env->NewStringUTF(name);
        env->SetObjectArrayElement(result, i, backend_str);
    }
    return result;
}

} // extern "C"
