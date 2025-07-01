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

using executorch::runtime::Error;

class LpaiDevice : public QnnDevice {
 public:
  LpaiDevice(QnnImplementation* implementation, QnnLogger* logger)
      : QnnDevice(implementation, logger){};

  executorch::runtime::Error Configure() override;

 private:
  std::vector<QnnDevice_Config_t> device_config_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
