/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnDlcManager.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>

namespace executorch {
namespace backends {
namespace qnn {

QnnDlcManager::QnnDlcManager(
    const QnnExecuTorchContextBinary& qnn_context_blob,
    const QnnExecuTorchOptions* options)
    : qnn_context_blob_(qnn_context_blob), options_(options) {
  if (options_ == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Fail to create QnnDlcManager, options is nullptr");
  }
}

Error QnnDlcManager::LoadQnnIrLibrary() {
  return Error::Ok;
}

Error QnnDlcManager::Create() {
  return Error::Ok;
}

Error QnnDlcManager::Configure(const std::vector<std::string>& graph_names) {
  return Error::Ok;
}

Error QnnDlcManager::SetUpDlcEnvironment(
    const Qnn_Version_t& coreApiVersion,
    const std::vector<std::string>& graph_names) {
  return Error::Ok;
}

void QnnDlcManager::Destroy() {}

} // namespace qnn
} // namespace backends
} // namespace executorch
