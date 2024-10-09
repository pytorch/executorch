/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendFactory.h>
namespace torch {
namespace executor {
namespace qnn {
std::unique_ptr<BackendConfigParameters> QnnBackendFactory::Create(
    const QnnImplementation& implementation,
    QnnLogger* logger,
    const QnnExecuTorchContextBinary& qnn_context_blob,
    const QnnExecuTorchOptions* options) {
  auto backend_params = std::make_unique<BackendConfigParameters>();

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
      backend_params->qnn_backend_ptr_ =
          std::make_unique<HtpBackend>(implementation, logger);

      backend_params->qnn_device_ptr_ = std::make_unique<HtpDevice>(
          implementation, logger, options->soc_info(), htp_options);

      backend_params->qnn_backend_cache_ptr_ =
          std::make_unique<HtpBackendCache>(qnn_context_blob);

      backend_params->qnn_context_ptr_ = std::make_unique<HtpContext>(
          implementation,
          backend_params->qnn_backend_ptr_.get(),
          backend_params->qnn_device_ptr_.get(),
          backend_params->qnn_backend_cache_ptr_.get(),
          htp_options);

      backend_params->qnn_graph_ptr_ = std::make_unique<HtpGraph>(
          implementation,
          backend_params->qnn_backend_ptr_.get(),
          backend_params->qnn_context_ptr_.get(),
          options->profile_level(),
          options->graph_name()->str(),
          options->soc_info(),
          htp_options);
      backend_params->qnn_mem_manager_ptr_ = std::make_unique<QnnMemManager>(
          implementation, backend_params->qnn_context_ptr_.get());
      backend_params->backend_init_state_ = BackendInitializeState::INITIALIZED;
    } break;
    case QnnExecuTorchBackendType::kGpuBackend:
    case QnnExecuTorchBackendType::kDspBackend:
    case QnnExecuTorchBackendType::kUndefinedBackend:
    default:
      return nullptr;
  }

  if (backend_params->qnn_backend_ptr_->VerifyQNNSDKVersion(
          options->backend_options()->backend_type()) == Error::Ok) {
    return backend_params;
  }

  return nullptr;
}
} // namespace qnn
} // namespace executor
} // namespace torch
