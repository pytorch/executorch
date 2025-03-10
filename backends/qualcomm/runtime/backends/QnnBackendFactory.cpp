/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendFactory.h>
namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

std::unique_ptr<BackendConfigurator> QnnBackendFactory::Create(
    const QnnBackends& implementations,
    QnnLogger* logger,
    const QnnExecuTorchContextBinary& qnn_context_blob,
    const QnnExecuTorchOptions* options) {
  auto backend_configurator_ptr = std::make_unique<BackendConfigurator>();

  if (implementations.qnn_loaded_comm_backend_ptr_ != nullptr) {
    const QnnImplementation& comm_implementation =
        *(implementations.qnn_loaded_comm_backend_ptr_);
    backend_configurator_ptr->backend_params_ptr_->qnn_comm_backend_ptr_ =
        std::make_unique<QnnBackend>(comm_implementation, logger);

    backend_configurator_ptr->backend_params_ptr_->qnn_comm_device_ptr_ =
        std::make_unique<QnnDevice>(comm_implementation, logger);

    backend_configurator_ptr->backend_params_ptr_->qnn_comm_backend_cache_ptr_ =
        std::make_unique<QnnBackendCache>(
            qnn_context_blob, options->graph_name()->str());

    backend_configurator_ptr->backend_params_ptr_->qnn_comm_context_ptr_ =
        std::make_unique<QnnContext>(
            comm_implementation,
            backend_configurator_ptr->backend_params_ptr_->qnn_comm_backend_ptr_
                .get(),
            backend_configurator_ptr->backend_params_ptr_->qnn_comm_device_ptr_
                .get(),
            backend_configurator_ptr->backend_params_ptr_
                ->qnn_comm_backend_cache_ptr_.get());

    backend_configurator_ptr->backend_params_ptr_->qnn_comm_graph_ptr_ =
        std::make_unique<QnnGraph>(
            comm_implementation,
            backend_configurator_ptr->backend_params_ptr_->qnn_comm_backend_ptr_
                .get(),
            backend_configurator_ptr->backend_params_ptr_->qnn_comm_context_ptr_
                .get(),
            options->profile_level());

    if (backend_configurator_ptr->backend_params_ptr_->qnn_comm_backend_ptr_
            ->VerifyQNNSDKVersion() != Error::Ok) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Fail to verify Qnn SDK version for common backend")

      return nullptr;
    }
  }

  switch (options->backend_options()->backend_type()) {
    case QnnExecuTorchBackendType::kHtpBackend: {
      auto htp_options = options->backend_options()->htp_options();
      if (options->log_level() >= QnnExecuTorchLogLevel::kLogLevelInfo) {
        const std::string skel_library_dir =
            htp_options->skel_library_dir()->str();
        if (!skel_library_dir.empty()) {
          setenv(
              "ADSP_LIBRARY_PATH", skel_library_dir.c_str(), /*overwrite=*/1);
        }
        QNN_EXECUTORCH_LOG_INFO(
            "skel_library_dir: %s", skel_library_dir.c_str());
        QNN_EXECUTORCH_LOG_INFO(
            "htp_arch in htp_info: %s",
            EnumNameHtpArch(options->soc_info()->htp_info()->htp_arch()));
        QNN_EXECUTORCH_LOG_INFO(
            "vtcm_size_in_mb in htp_info: %d",
            options->soc_info()->htp_info()->vtcm_size_in_mb());
        QNN_EXECUTORCH_LOG_INFO(
            "performance_mode in htp_options: %s",
            EnumNameQnnExecuTorchHtpPerformanceMode(
                htp_options->performance_mode()));
        QNN_EXECUTORCH_LOG_INFO(
            "precision in htp_options: %s",
            EnumNameQnnExecuTorchHtpPrecision(htp_options->precision()));
        QNN_EXECUTORCH_LOG_INFO(
            "pd_session in htp_options: %s",
            EnumNameQnnExecuTorchHtpPdSession(htp_options->pd_session()));
        QNN_EXECUTORCH_LOG_INFO(
            "use_conv_hmx in htp_options: %d", htp_options->use_conv_hmx());
        QNN_EXECUTORCH_LOG_INFO(
            "use_fold_relu in htp_options: %d", htp_options->use_fold_relu());
      }
      const QnnImplementation& implementation =
          *(implementations.qnn_loaded_backend_ptr_);
      backend_configurator_ptr->backend_params_ptr_->qnn_backend_ptr_ =
          std::make_unique<HtpBackend>(implementation, logger);

      backend_configurator_ptr->backend_params_ptr_->qnn_device_ptr_ =
          std::make_unique<HtpDevice>(
              implementation, logger, options->soc_info(), htp_options);

      backend_configurator_ptr->backend_params_ptr_->qnn_backend_cache_ptr_ =
          std::make_unique<HtpBackendCache>(
              qnn_context_blob, options->graph_name()->str());

      backend_configurator_ptr->backend_params_ptr_
          ->qnn_context_ptr_ = std::make_unique<HtpContext>(
          implementation,
          backend_configurator_ptr->backend_params_ptr_->qnn_backend_ptr_.get(),
          backend_configurator_ptr->backend_params_ptr_->qnn_device_ptr_.get(),
          backend_configurator_ptr->backend_params_ptr_->qnn_backend_cache_ptr_
              .get(),
          htp_options);

      backend_configurator_ptr->backend_params_ptr_
          ->qnn_graph_ptr_ = std::make_unique<HtpGraph>(
          implementation,
          backend_configurator_ptr->backend_params_ptr_->qnn_backend_ptr_.get(),
          backend_configurator_ptr->backend_params_ptr_->qnn_context_ptr_.get(),
          options->profile_level(),
          options->soc_info(),
          htp_options);
      backend_configurator_ptr->backend_params_ptr_->qnn_mem_manager_ptr_ =
          std::make_unique<QnnMemManager>(
              implementation,
              backend_configurator_ptr->backend_params_ptr_->qnn_context_ptr_
                  .get());
      backend_configurator_ptr->backend_params_ptr_->backend_init_state_ =
          BackendInitializeState::INITIALIZED;
    } break;
    case QnnExecuTorchBackendType::kGpuBackend:
    case QnnExecuTorchBackendType::kDspBackend:
    case QnnExecuTorchBackendType::kUndefinedBackend:
    default:
      return nullptr;
  }

  if (backend_configurator_ptr->backend_params_ptr_->qnn_backend_ptr_
          ->VerifyQNNSDKVersion() == Error::Ok) {
    return backend_configurator_ptr;
  }

  return nullptr;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
