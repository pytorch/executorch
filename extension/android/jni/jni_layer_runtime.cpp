/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fbjni/fbjni.h>
#include <jni.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/kernel/operator_registry.h>

namespace executorch_jni {
namespace runtime = ::executorch::ET_RUNTIME_NAMESPACE;

class AndroidRuntimeJni : public facebook::jni::JavaClass<AndroidRuntimeJni> {
 public:
  constexpr static const char* kJavaDescriptor =
      "Lorg/pytorch/executorch/ExecuTorchRuntime;";

  static void registerNatives() {
    javaClassStatic()->registerNatives({
        makeNativeMethod(
            "getRegisteredOps", AndroidRuntimeJni::getRegisteredOps),
        makeNativeMethod(
            "getRegisteredBackends", AndroidRuntimeJni::getRegisteredBackends),
    });
  }

  // Returns a string array of all registered ops
  static facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>>
  getRegisteredOps(facebook::jni::alias_ref<jclass>) {
    auto kernels = runtime::get_registered_kernels();
    auto result = facebook::jni::JArrayClass<jstring>::newArray(kernels.size());

    for (size_t i = 0; i < kernels.size(); ++i) {
      auto op = facebook::jni::make_jstring(kernels[i].name_);
      result->setElement(i, op.get());
    }

    return result;
  }

  // Returns a string array of all registered backends
  static facebook::jni::local_ref<facebook::jni::JArrayClass<jstring>>
  getRegisteredBackends(facebook::jni::alias_ref<jclass>) {
    int num_backends = runtime::get_num_registered_backends();
    auto result = facebook::jni::JArrayClass<jstring>::newArray(num_backends);

    for (int i = 0; i < num_backends; ++i) {
      auto name_result = runtime::get_backend_name(i);
      const char* name = "";

      if (name_result.ok()) {
        name = *name_result;
      }

      auto backend_str = facebook::jni::make_jstring(name);
      result->setElement(i, backend_str.get());
    }

    return result;
  }
};

} // namespace executorch_jni

void register_natives_for_runtime() {
  executorch_jni::AndroidRuntimeJni::registerNatives();
}
