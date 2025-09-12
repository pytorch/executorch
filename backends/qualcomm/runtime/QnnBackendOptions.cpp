/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorchBackend.h>

namespace executorch {
namespace backends {
namespace qnn {

using namespace qnn_delegate;

template <typename T>
T get_option(T aot_option) {
  executorch::runtime::Error status;
  executorch::runtime::BackendOption backend_option;

  if constexpr (std::is_same_v<T, QnnExecuTorchLogLevel>) {
    std::strncpy(
        backend_option.key,
        QNN_RUNTIME_LOG_LEVEL,
        runtime::kMaxOptionKeyLength);
    backend_option.key[runtime::kMaxOptionKeyLength - 1] = '\0';
    backend_option.value = -1;
  } else if constexpr (std::is_same_v<T, QnnExecuTorchHtpPerformanceMode>) {
    std::strncpy(
        backend_option.key,
        QNN_RUNTIME_HTP_PERFORMANCE_MODE,
        runtime::kMaxOptionKeyLength);
    backend_option.key[runtime::kMaxOptionKeyLength - 1] = '\0';
    backend_option.value = -1;
  } else if constexpr (std::is_same_v<T, QnnExecuTorchProfileLevel>) {
    std::strncpy(
        backend_option.key,
        QNN_RUNTIME_PROFILE_LEVEL,
        runtime::kMaxOptionKeyLength);
    backend_option.key[runtime::kMaxOptionKeyLength - 1] = '\0';
    backend_option.value = -1;
  }

  // This will call get_option under runtime backend interface
  status = get_option(QNN_BACKEND, backend_option);

  if (status != executorch::runtime::Error::Ok) {
    return aot_option;
  } else {
    return static_cast<T>(std::get<int>(backend_option.value));
  }
}

// Explicit instantiations
template QnnExecuTorchLogLevel get_option<QnnExecuTorchLogLevel>(
    QnnExecuTorchLogLevel);
template QnnExecuTorchHtpPerformanceMode get_option<
    QnnExecuTorchHtpPerformanceMode>(QnnExecuTorchHtpPerformanceMode);
template QnnExecuTorchProfileLevel get_option<QnnExecuTorchProfileLevel>(
    QnnExecuTorchProfileLevel);

} // namespace qnn
} // namespace backends
} // namespace executorch
