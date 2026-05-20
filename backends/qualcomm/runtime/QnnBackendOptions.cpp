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
T get_option(T aot_option, const char* aot_key) {
  executorch::runtime::Error status;
  executorch::runtime::BackendOption backend_option;
  std::strncpy(backend_option.key, aot_key, runtime::kMaxOptionKeyLength);
  backend_option.key[runtime::kMaxOptionKeyLength - 1] = '\0';
  backend_option.value = -1;

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
    QnnExecuTorchLogLevel,
    const char*);
template QnnExecuTorchHtpPerformanceMode
get_option<QnnExecuTorchHtpPerformanceMode>(
    QnnExecuTorchHtpPerformanceMode,
    const char*);
template int32_t get_option<int32_t>(int32_t, const char*);
template QnnExecuTorchLpaiClientPerf get_option<QnnExecuTorchLpaiClientPerf>(
    QnnExecuTorchLpaiClientPerf,
    const char*);
template QnnExecuTorchLpaiCoreAffinity get_option<
    QnnExecuTorchLpaiCoreAffinity>(QnnExecuTorchLpaiCoreAffinity, const char*);
template uint32_t get_option<uint32_t>(uint32_t, const char*);
template QnnExecuTorchProfileLevel get_option<QnnExecuTorchProfileLevel>(
    QnnExecuTorchProfileLevel,
    const char*);

} // namespace qnn
} // namespace backends
} // namespace executorch
