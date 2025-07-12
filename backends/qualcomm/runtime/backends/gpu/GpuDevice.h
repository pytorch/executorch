/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/backends/QnnDeviceCommon.h>

namespace executorch {
namespace backends {
namespace qnn {

class GpuDevice : public QnnDevice {
 public:
  GpuDevice(const QnnImplementation& implementation, QnnLogger* logger)
      : QnnDevice(implementation, logger){};

  // GPU backend does not support device creation
  executorch::runtime::Error Configure() override {
    return executorch::runtime::Error::Ok;
  }
};

} // namespace qnn
} // namespace backends
} // namespace executorch
