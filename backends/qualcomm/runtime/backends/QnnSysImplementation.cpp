/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dlfcn.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnSysImplementation.h>
namespace torch {
namespace executor {
namespace qnn {
Error QnnSystemImplementation::Load() {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  void* lib_handle_ = dlopen(lib_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (lib_handle_ == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Cannot Open QNN library %s, with error: %s",
        lib_path_.c_str(),
        dlerror());
    return Error::Internal;
  }

  auto* get_providers =
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      reinterpret_cast<QnnSystemInterfaceGetProvidersFn*>(
          dlsym(lib_handle_, "QnnSystemInterface_getProviders"));
  if (get_providers == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QnnSystemImplementation::Load Cannot load symbol "
        "QnnSystemInterface_getProviders : %s",
        dlerror());
    return Error::Internal;
  }

  std::uint32_t num_providers;
  const QnnSystemInterface_t** provider_list = nullptr;
  error = get_providers(&provider_list, &num_providers);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QnnSystemInterface failed to "
        "get providers. Error %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  if (num_providers != required_num_providers_) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QnnSystemInterface Num "
        "Providers is %d instead of required %d",
        num_providers,
        required_num_providers_);
    return Error::Internal;
  }

  qnn_sys_interface_.SetQnnSystemInterface(provider_list[0]);

  return Error::Ok;
}

Error QnnSystemImplementation::Unload() {
  if (lib_handle_ == nullptr)
    return Error::Ok;

  int dlclose_error = dlclose(lib_handle_);
  if (dlclose_error != 0) {
    QNN_EXECUTORCH_LOG_WARN(
        "Failed to close QnnSystem library with error %s", dlerror());
    return Error::Internal;
  }

  lib_handle_ = nullptr;

  return Error::Ok;
}

const QnnSystemInterface& QnnSystemImplementation::GetQnnSystemInterface()
    const {
  if (!qnn_sys_interface_.IsLoaded()) {
    QNN_EXECUTORCH_LOG_WARN(
        "GetQnnSystemInterface, returning a QNN interface "
        "which is not loaded yet.");
  }
  return qnn_sys_interface_;
}
} // namespace qnn
} // namespace executor
} // namespace torch
