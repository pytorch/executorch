/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <dlfcn.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>

#include "QnnInterface.h"
namespace torch {
namespace executor {
namespace qnn {
template <typename Fn>
Fn loadQnnFunction(void* handle, const char* function_name) {
  return reinterpret_cast<Fn>(dlsym(handle, function_name)); // NOLINT
}

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

// instantiate static members
// NOLINTNEXTLINE(fuchsia-statically-constructed-objects)
std::unordered_map<std::string, QnnImplementation::BackendIdType>
    QnnImplementation::lib_path_to_backend_id_;
// NOLINTNEXTLINE(fuchsia-statically-constructed-objects)
std::unordered_map<QnnImplementation::BackendIdType, const QnnInterface_t*>
    QnnImplementation::loaded_backend_;
// NOLINTNEXTLINE(fuchsia-statically-constructed-objects)
std::unordered_map<QnnImplementation::BackendIdType, void*>
    QnnImplementation::loaded_lib_handle_;
// NOLINTNEXTLINE(fuchsia-statically-constructed-objects)
std::mutex QnnImplementation::be_init_mutex_;

Error QnnImplementation::StartBackend(
    const std::string& lib_path,
    const QnnSaver_Config_t** saver_config) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  void* lib_handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);

  if (lib_handle == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Cannot Open QNN library %s, with error: %s",
        lib_path.c_str(),
        dlerror());
    return Error::Internal;
  }

  // load get_provider function
  auto get_providers = loadQnnFunction<QnnInterfaceGetProvidersFn*>(
      lib_handle, "QnnInterface_getProviders");

  if (get_providers == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QnnImplementation::Load Cannot load symbol "
        "QnnInterface_getProviders : %s",
        dlerror());
    return Error::Internal;
  }

  // Get QnnInterface Providers
  std::uint32_t num_providers;
  const QnnInterface_t** provider_list = nullptr;
  error = get_providers(&provider_list, &num_providers);

  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Qnn Interface failed to get providers. Error %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  if (num_providers != required_num_providers_) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Qnn Interface Num Providers is "
        "%d instead of required %d",
        num_providers,
        required_num_providers_);
    return Error::Internal;
  }

  BackendIdType backend_id = provider_list[0]->backendId;

  // store everything
  lib_path_to_backend_id_[lib_path] = backend_id;

  // we use lib_path as the first unique key.
  // Users can get wrong like, he or she assigns
  //   library_path=libQnnHtp_1.so
  //   library_path=libQnnHtp_2.so
  // for different QnnBackend instances.
  // So we warning out here.
  if (loaded_backend_.count(backend_id) > 0) {
    QNN_EXECUTORCH_LOG_WARN(
        "lib_path %s is loaded, but backend %d "
        "already exists. Overwriting previous loaded backend...",
        lib_path.c_str(),
        backend_id);
  }
  loaded_backend_[backend_id] = provider_list[0];

  if (loaded_lib_handle_.count(backend_id) > 0) {
    QNN_EXECUTORCH_LOG_WARN("closing %pK...", loaded_lib_handle_[backend_id]);

    int dlclose_error = dlclose(loaded_lib_handle_[backend_id]);
    if (dlclose_error != 0) {
      QNN_EXECUTORCH_LOG_WARN(
          "Sadly, fail to close %pK with error %s",
          loaded_lib_handle_[backend_id],
          dlerror());
    }
  }
  loaded_lib_handle_[backend_id] = lib_handle;

  // Saver backend need initialization.
  Error be_init_st = InitBackend(loaded_lib_handle_[backend_id], saver_config);

  if (be_init_st != Error::Ok) {
    // backend init fails. clear things
    lib_path_to_backend_id_.erase(lib_path);
    loaded_backend_.erase(backend_id);

    int dlclose_error = dlclose(loaded_lib_handle_[backend_id]);
    if (dlclose_error != 0) {
      QNN_EXECUTORCH_LOG_WARN(
          "fail to close %pK after backend-init "
          "failure, with error %s",
          loaded_lib_handle_[backend_id],
          dlerror());
    }

    loaded_lib_handle_.erase(backend_id);
    return be_init_st;
  }

  return Error::Ok;
}

Error QnnImplementation::TerminateAllBackends() {
  Error ret_status = Error::Ok;

  loaded_backend_.clear();

  for (auto& it : loaded_lib_handle_) {
    int dlclose_error = dlclose(it.second);
    if (dlclose_error != 0) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Fail to close QNN backend %d with error %s", it.first, dlerror());
      ret_status = Error::Internal;
    }
  }
  loaded_lib_handle_.clear();
  lib_path_to_backend_id_.clear();

  return ret_status;
}

Error QnnImplementation::Load(const QnnSaver_Config_t** saver_config) {
  BackendIdType backend_id = QNN_BACKEND_ID_NULL;
  {
    const std::lock_guard<std::mutex> lock(be_init_mutex_);

    if (lib_path_to_backend_id_.count(lib_path_) == 0) {
      Error st = StartBackend(lib_path_, saver_config);
      ET_CHECK_OR_RETURN_ERROR(
          st == Error::Ok, Internal, "Fail to start backend");
    }

    // Get backend ID
    backend_id = lib_path_to_backend_id_[lib_path_];

    // really don't expect.
    if (loaded_backend_.count(backend_id) == 0 ||
        loaded_lib_handle_.count(backend_id) == 0) {
      QNN_EXECUTORCH_LOG_ERROR(
          "library %s is loaded but "
          "loaded backend count=%zu, "
          "loaded lib_handle count=%zu",
          lib_path_.c_str(),
          loaded_backend_.count(backend_id),
          loaded_lib_handle_.count(backend_id));
      return Error::Internal;
    }
  } // be_init_mutex_ release.

  // Connect QnnInterface
  qnn_interface_.SetQnnInterface(loaded_backend_[backend_id]);

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
} // namespace executor
} // namespace torch
