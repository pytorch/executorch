/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendFactory.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnDlcManager.h>
namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

std::unique_ptr<BackendConfigParameters> QnnBackendFactory::Create(
    QnnImplementation* implementation_ptr,
    QnnBackend* qnn_backend_ptr,
    QnnDevice* qnn_device_ptr,
    const QnnExecuTorchContextBinary& qnn_context_blob,
    const QnnExecuTorchOptions* options,
    QnnDlcManager* qnn_dlc_manager) {
  auto backend_params = std::make_unique<BackendConfigParameters>();

  switch (options->backend_options()->backend_type()) {
    case QnnExecuTorchBackendType::kHtpBackend: {
      auto htp_options = options->backend_options()->htp_options();
      if (get_option(options->log_level()) >=
          QnnExecuTorchLogLevel::kLogLevelInfo) {
        QNN_EXECUTORCH_LOG_INFO(
            "htp_arch in htp_info: %s",
            EnumNameHtpArch(options->soc_info()->htp_info()->htp_arch()));
        QNN_EXECUTORCH_LOG_INFO(
            "vtcm_size_in_mb in htp_info: %d",
            options->soc_info()->htp_info()->vtcm_size_in_mb());
        QNN_EXECUTORCH_LOG_INFO(
            "performance_mode in htp_options: %s",
            EnumNameQnnExecuTorchHtpPerformanceMode(
                get_option(htp_options->performance_mode())));
        QNN_EXECUTORCH_LOG_INFO(
            "precision in htp_options: %s",
            EnumNameQnnExecuTorchHtpPrecision(htp_options->precision()));
        QNN_EXECUTORCH_LOG_INFO(
            "pd_session in htp_options: %s",
            EnumNameQnnExecuTorchHtpPdSession(htp_options->pd_session()));
        QNN_EXECUTORCH_LOG_INFO(
            "use_conv_hmx in htp_options: %d", htp_options->use_conv_hmx());
        QNN_EXECUTORCH_LOG_INFO(
            "use_dlbc in htp_options: %d", htp_options->use_dlbc());
        QNN_EXECUTORCH_LOG_INFO(
            "use_fold_relu in htp_options: %d", htp_options->use_fold_relu());
        QNN_EXECUTORCH_LOG_INFO(
            "use_multi_contexts in htp_options: %d",
            htp_options->use_multi_contexts());
        QNN_EXECUTORCH_LOG_INFO(
            "use_weight_sharing in htp_options: %d",
            htp_options->use_weight_sharing());
      }
      backend_params->qnn_backend_cache_ptr_ =
          std::make_unique<HtpBackendCache>(qnn_context_blob);

      backend_params->qnn_context_ptr_ = std::make_unique<HtpContext>(
          implementation_ptr,
          qnn_backend_ptr,
          qnn_device_ptr,
          backend_params->qnn_backend_cache_ptr_.get(),
          htp_options,
          qnn_dlc_manager);

      backend_params->qnn_graph_ptr_ = std::make_unique<HtpGraph>(
          implementation_ptr,
          qnn_backend_ptr,
          backend_params->qnn_context_ptr_.get(),
          get_option(options->profile_level()),
          options->soc_info(),
          htp_options);
    } break;
    case QnnExecuTorchBackendType::kGpuBackend: {
      auto gpu_options = options->backend_options()->gpu_options();
      if (options->log_level() >= QnnExecuTorchLogLevel::kLogLevelInfo) {
        QNN_EXECUTORCH_LOG_INFO(
            "performance_mode in gpu_options: %s",
            EnumNameQnnExecuTorchGpuPerformanceMode(
                gpu_options->performance_mode()));
        QNN_EXECUTORCH_LOG_INFO(
            "precision in gpu_options: %s",
            EnumNameQnnExecuTorchGpuPrecision(gpu_options->precision()));
        QNN_EXECUTORCH_LOG_INFO(
            "use_memory_optimizations in gpu_options: %d",
            gpu_options->use_memory_optimizations());
        QNN_EXECUTORCH_LOG_INFO(
            "use_node_optimizations in gpu_options: %d",
            gpu_options->use_node_optimizations());
        QNN_EXECUTORCH_LOG_INFO(
            "use_queue_recording in gpu_options: %d",
            gpu_options->use_queue_recording());
        QNN_EXECUTORCH_LOG_INFO(
            "use_weight_sharing in gpu_options: %d",
            gpu_options->use_weight_sharing());
      }

      backend_params->qnn_backend_cache_ptr_ =
          std::make_unique<QnnBackendCache>(qnn_context_blob);

      backend_params->qnn_context_ptr_ = std::make_unique<GpuContext>(
          implementation_ptr,
          qnn_backend_ptr,
          qnn_device_ptr,
          backend_params->qnn_backend_cache_ptr_.get(),
          qnn_dlc_manager,
          gpu_options);

      backend_params->qnn_graph_ptr_ = std::make_unique<GpuGraph>(
          implementation_ptr,
          qnn_backend_ptr,
          backend_params->qnn_context_ptr_.get(),
          options->profile_level(),
          gpu_options);
    } break;
    case QnnExecuTorchBackendType::kDspBackend:
    case QnnExecuTorchBackendType::kUndefinedBackend:
    default:
      return nullptr;
  }

  backend_params->qnn_mem_manager_ptr_ = std::make_unique<QnnMemManager>(
      implementation_ptr,
      backend_params->qnn_context_ptr_.get(),
      options->log_level());

  backend_params->backend_init_state_ = BackendInitializeState::INITIALIZED;
  return backend_params;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
