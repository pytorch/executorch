/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>

namespace executorch {
namespace backends {
namespace qnn {
class IrContext : public QnnContext {
 public:
  using QnnContext::QnnContext;

  executorch::runtime::Error GetContextBinary(
      QnnExecuTorchContextBinary& qnn_executorch_context_binary) override;

 private:
  std::vector<char> buffer_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
