/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/lpai/LpaiGraph.h>

namespace executorch {
namespace backends {
namespace qnn {

Error LpaiGraph::AfterConfigure(const std::string& graph_name) {
  // LPAI does not support online prepare and require graph to be finalized
  // again
  Qnn_ErrorHandle_t error = GraphFinalize(graph_name);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to finalize Qnn Graph with error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  return Error::Ok;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
