/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EXECUTORCH_QNN_EXECUTORCH_BACKENDS_HTP_BACKEND_HTP_BACKEND_H_
#define EXECUTORCH_QNN_EXECUTORCH_BACKENDS_HTP_BACKEND_HTP_BACKEND_H_
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
namespace torch {
namespace executor {
namespace qnn {
class HtpBackend : public QnnBackend {
 public:
  HtpBackend(const QnnImplementation& implementation, QnnLogger* logger)
      : QnnBackend(implementation, logger) {}
  ~HtpBackend() {}

 protected:
  Error MakeConfig(std::vector<const QnnBackend_Config_t*>& config) override {
    return Error::Ok;
  }
};
}  // namespace qnn
}  // namespace executor
}  // namespace torch
#endif  // EXECUTORCH_QNN_EXECUTORCH_BACKENDS_HTP_BACKEND_HTP_BACKEND_H_
