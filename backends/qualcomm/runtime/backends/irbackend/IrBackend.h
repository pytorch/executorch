/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 23)
#include "IR/QnnIrCommon.h"
#endif
#include "QnnTypes.h"

namespace executorch {
namespace backends {
namespace qnn {
class IrBackend : public QnnBackend {
 public:
  IrBackend(const QnnImplementation& implementation, QnnLogger* logger)
      : QnnBackend(implementation, logger) {}
  ~IrBackend() {}

  Qnn_Version_t GetExpectedBackendVersion() const override {
    Qnn_Version_t backend_version;
#if (QNN_API_VERSION_MAJOR >= 2 && QNN_API_VERSION_MINOR >= 23)
    backend_version.major = QNN_IR_API_VERSION_MAJOR;
    backend_version.minor = QNN_IR_API_VERSION_MINOR;
    backend_version.patch = QNN_IR_API_VERSION_PATCH;
#else
    backend_version = QNN_VERSION_INIT;
#endif
    return backend_version;
  }
};
} // namespace qnn
} // namespace backends
} // namespace executorch
