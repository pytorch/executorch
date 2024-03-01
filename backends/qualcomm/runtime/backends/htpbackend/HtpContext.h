/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
namespace torch {
namespace executor {
namespace qnn {
class HtpContext : public QnnContext {
 public:
  HtpContext(
      const QnnImplementation& implementation,
      QnnBackend* backend,
      QnnDevice* device,
      const QnnExecuTorchContextBinary& qnn_context_blob,
      const QnnExecuTorchHtpBackendOptions* htp_options)
      : QnnContext(implementation, backend, device, qnn_context_blob) {}
  ~HtpContext() {}

 protected:
  Error MakeConfig(std::vector<const QnnContext_Config_t*>& config) override {
    return Error::Ok;
  }
};
} // namespace qnn
} // namespace executor
} // namespace torch
