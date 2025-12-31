/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <memory>
#include "QnnInterface.h"
namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

struct DlCloser {
  int operator()(void* handle) {
    if (handle == nullptr)
      return 0;
    return dlclose(handle);
  }
};

Error QnnImplementation::InitBackend(
    void* const lib_handle,
    const QnnSaver_Config_t** saver_config) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  // saver_config must be set before backend initialization
  auto saver_initialize =
      loadQnnFunction<QnnSaverInitializeFn*>(lib_handle, "QnnSaver_initialize");
  if (saver_initialize != nullptr) {
    error = saver_initialize(saver_config);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "[Qnn Delegate] QnnSaver Backend Failed to "
          "saver_initialize. Error %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }
  return Error::Ok;
}

QnnImplementation::~QnnImplementation() {
  Unload();
}

const QnnInterface_t* QnnImplementation::StartBackend(
    const std::string& lib_path,
    const QnnSaver_Config_t** saver_config) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  // If the library is already loaded, return the handle.
  std::unique_ptr<void, DlCloser> lib_handle(
      dlopen(lib_path.c_str(), RTLD_NOW | RTLD_NOLOAD));
  if (!lib_handle) {
    lib_handle = std::unique_ptr<void, DlCloser>(
        dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL));
  }
  if (lib_handle == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Cannot Open QNN library %s, with error: %s",
        lib_path.c_str(),
        dlerror());
    return nullptr;
  }

  // load get_provider function
  auto get_providers = loadQnnFunction<QnnInterfaceGetProvidersFn*>(
      lib_handle.get(), "QnnInterface_getProviders");

  if (get_providers == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QnnImplementation::Load Cannot load symbol "
        "QnnInterface_getProviders : %s",
        dlerror());
    return nullptr;
  }

  // Get QnnInterface Providers
  std::uint32_t num_providers;
  const QnnInterface_t** provider_list = nullptr;
  error = get_providers(&provider_list, &num_providers);

  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Qnn Interface failed to get providers. Error %d",
        QNN_GET_ERROR_CODE(error));
    return nullptr;
  }

  if (num_providers != required_num_providers_) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Qnn Interface Num Providers is "
        "%d instead of required %d",
        num_providers,
        required_num_providers_);
    return nullptr;
  }

  // Saver backend need initialization.
  Error be_init_st = InitBackend(lib_handle.get(), saver_config);

  if (be_init_st != Error::Ok) {
    return nullptr;
  }

  // hold the lib_handle
  lib_handle_ = lib_handle.release();
  return provider_list[0];
}

Error QnnImplementation::Unload() {
  qnn_interface_.Unload();

  if (lib_handle_ == nullptr) {
    return Error::Ok;
  }

  int dlclose_error = dlclose(lib_handle_);
  if (dlclose_error != 0) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Fail to close QNN backend %s with error %s",
        lib_path_.c_str(),
        dlerror());
    return Error::Internal;
  }
  lib_handle_ = nullptr;
  return Error::Ok;
}

Error QnnImplementation::Load(const QnnSaver_Config_t** saver_config) {
  const QnnInterface_t* p_qnn_intf = StartBackend(lib_path_, saver_config);
  ET_CHECK_OR_RETURN_ERROR(
      p_qnn_intf != nullptr, Internal, "Fail to start backend");

  // Connect QnnInterface
  qnn_interface_.SetQnnInterface(p_qnn_intf);

  return Error::Ok;
}

const QnnInterface& QnnImplementation::GetQnnInterface() const {
  if (!qnn_interface_.IsLoaded()) {
    QNN_EXECUTORCH_LOG_WARN(
        "GetQnnInterface, returning a QNN interface "
        "which is not loaded yet.");
  }
  return qnn_interface_;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
