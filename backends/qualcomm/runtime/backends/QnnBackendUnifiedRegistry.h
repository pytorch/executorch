/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnDeviceCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnLogger.h>
#include <executorch/runtime/core/error.h>

#include <memory>
#include <mutex>
#include <string>

namespace executorch {
namespace backends {
namespace qnn {

// A bundle struct to hold all shared QNN backend resources
struct QnnBackendBundle {
  std::unique_ptr<QnnImplementation> implementation;
  std::unique_ptr<QnnLogger> qnn_logger_ptr;
  std::unique_ptr<QnnBackend> qnn_backend_ptr;
  std::unique_ptr<QnnDevice> qnn_device_ptr;

  // Default ctor
  QnnBackendBundle()
      : implementation(nullptr),
        qnn_logger_ptr(nullptr),
        qnn_backend_ptr(nullptr),
        qnn_device_ptr(nullptr) {}
  // Default dtor
  ~QnnBackendBundle() {
    qnn_device_ptr.reset();
    qnn_backend_ptr.reset();
    qnn_logger_ptr.reset();
    implementation.reset();
  }
};

class QnnBackendUnifiedRegistry {
  // Singleton class to manage shared QNN backend resources. It ensures that
  // only one instance of the registry exists throughout the application's
  // lifetime. The registry maintains a map of backend bundles indexed by
  // backend_type. Each bundle contains QnnImplentation, QnnLogger, QnnBackend,
  // and QnnDevice objects for a specific backend type. The registry provides
  // methods to get or create backend bundles, ensuring that resources are
  // properly managed and reused when possible. It also includes a cleanup
  // mechanism to remove expired bundles.
 public:
  static QnnBackendUnifiedRegistry& GetInstance();

  executorch::runtime::Error GetOrCreateBackendBundle(
      const QnnExecuTorchOptions* options,
      std::shared_ptr<QnnBackendBundle>& bundle);

  void CleanupExpired();

 private:
  QnnBackendUnifiedRegistry();
  ~QnnBackendUnifiedRegistry();

  // Delete copy constructor and assignment operator
  QnnBackendUnifiedRegistry(const QnnBackendUnifiedRegistry&) = delete;
  QnnBackendUnifiedRegistry& operator=(const QnnBackendUnifiedRegistry&) =
      delete;

  static constexpr const char* htp_library_name_ = "libQnnHtp.so";
  static constexpr const char* gpu_library_name_ = "libQnnGpu.so";
  static constexpr const char* dsp_library_name_ = "libQnnDsp.so";

  std::unique_ptr<const QnnSaver_Config_t*[]> GetImplementationConfig(
      const QnnExecuTorchOptions* options) {
    if (options->saver()) {
      auto outputDirCfg = std::make_unique<QnnSaver_Config_t>();
      outputDirCfg->option = QNN_SAVER_CONFIG_OPTION_OUTPUT_DIRECTORY;
      outputDirCfg->outputDirectory = options->saver_output_dir()->c_str();

      auto saverCfg = std::make_unique<const QnnSaver_Config_t*[]>(2);
      saverCfg[0] = outputDirCfg.release();
      saverCfg[1] = nullptr;

      return saverCfg;
    } else {
      return nullptr;
    }
  }

  // Stores the collection of shared resources, with backend_type being used as
  // the key.
  std::unordered_map<QnnExecuTorchBackendType, std::weak_ptr<QnnBackendBundle>>
      qnn_backend_bundles_map_;

  std::mutex mutex_; // Protects access to resources and ensures atomic
                     // creation/destruction
};

} // namespace qnn
} // namespace backends
} // namespace executorch
