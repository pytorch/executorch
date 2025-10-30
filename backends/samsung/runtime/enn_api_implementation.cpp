/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#include <dlfcn.h>
#include <executorch/backends/samsung/runtime/enn_api_implementation.h>
#include <executorch/backends/samsung/runtime/logging.h>

#define ENN_LOAD_API_FUNC(handle, name, enn_api_ptr)                         \
  enn_api_ptr->name =                                                        \
      reinterpret_cast<name##_fn>(loadApiFunction(handle, #name, false));    \
  if (enn_api_ptr->name == nullptr) {                                        \
    ENN_LOG_ERROR("Unable to access symbols in enn api library: %s", #name); \
    dlclose(handle);                                                         \
    return Error::Internal;                                                  \
  }

namespace torch {
namespace executor {
namespace enn {
void* loadApiFunction(void* handle, const char* name, bool optional) {
  if (handle == nullptr) {
    return nullptr;
  }
  void* fn = dlsym(handle, name);
  if (fn == nullptr && !optional) {
    ENN_LOG_WARN("Failed to load function %s", name);
  }
  return fn;
}

std::mutex EnnApi::instance_mutex_;

EnnApi* EnnApi::getEnnApiInstance() {
  std::lock_guard<std::mutex> lgd(instance_mutex_);
  static EnnApi enn_api;
  if (!enn_api.getInitialize()) {
    auto status = enn_api.loadApiLib();
    if (status == Error::Ok) {
      enn_api.initialize_ = true;
    }
  }
  return &enn_api;
}

EnnApi::~EnnApi() {
  if (getInitialize()) {
    unloadApiLib();
  }
}

bool EnnApi::getInitialize() const {
  return initialize_;
}

Error EnnApi::loadApiLib() {
  const char enn_api_lib_name[] = "libenn_public_api_cpp.so";
  libenn_public_api_ = dlopen(enn_api_lib_name, RTLD_NOW | RTLD_LOCAL);
  ET_CHECK_OR_RETURN_ERROR(
      libenn_public_api_ != nullptr, Internal, "Lib load failed.")

  ENN_LOAD_API_FUNC(libenn_public_api_, EnnInitialize, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnSetPreferencePerfMode, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnGetPreferencePerfMode, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnOpenModel, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnOpenModelFromMemory, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnSetFastIpc, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnUnsetFastIpc, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnExecuteModelFastIpc, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnExecuteModel, this);
  ENN_LOAD_API_FUNC(
      libenn_public_api_, EnnExecuteModelWithSessionIdAsync, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnExecuteModelWithSessionIdWait, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnCloseModel, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnDeinitialize, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnAllocateAllBuffers, this);
  ENN_LOAD_API_FUNC(
      libenn_public_api_, EnnAllocateAllBuffersWithSessionId, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnBufferCommit, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnGetBuffersInfo, this);
  ENN_LOAD_API_FUNC(libenn_public_api_, EnnReleaseBuffers, this);

  return Error::Ok;
}

Error EnnApi::unloadApiLib() {
  if (dlclose(libenn_public_api_) != 0) {
    ENN_LOG_ERROR("Failed to close enn public api library. %s", dlerror());
    return Error::Internal;
  };
  return Error::Ok;
}

} // namespace enn
} // namespace executor
} // namespace torch
