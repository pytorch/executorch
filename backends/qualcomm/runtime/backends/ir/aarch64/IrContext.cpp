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
  return Error::Ok;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
