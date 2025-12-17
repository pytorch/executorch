/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fstream>

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/ir/IrContext.h>
namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

Error IrContext::GetContextBinary(
    QnnExecuTorchContextBinary& qnn_executorch_context_binary) {
  // read Dlc and write to buffer
  std::string dlc_name = GetGraphNames()[0] + ".dlc";
  std::ifstream dlc_file(dlc_name, std::ios::binary | std::ios::ate);
  if (dlc_file.is_open()) {
    std::streamsize size = dlc_file.tellg();
    dlc_file.seekg(0, std::ios::beg);

    buffer_ = std::vector<char>(size);
    dlc_file.read(buffer_.data(), size);
    dlc_file.close();
    qnn_executorch_context_binary.buffer = buffer_.data();
    qnn_executorch_context_binary.nbytes = size;
    return Error::Ok;
  } else {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to open dlc file %s for building QnnExecuTorchContextBinary",
        dlc_name.c_str());
  }
  return Error::Internal;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
