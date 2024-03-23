/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnLogger.h>
#include <executorch/backends/qualcomm/schema_generated.h>

#include <vector>

#include "QnnDevice.h"
namespace torch {
namespace executor {
namespace qnn {
class QnnDevice {
 public:
  explicit QnnDevice(const QnnImplementation& implementation, QnnLogger* logger)
      : implementation_(implementation), handle_(nullptr), logger_(logger) {}

  virtual ~QnnDevice();

  Qnn_DeviceHandle_t GetHandle() {
    return handle_;
  }

  Error Configure();

 protected:
  virtual Error MakeConfig(std::vector<const QnnDevice_Config_t*>& config) {
    return Error::Ok;
  };

  virtual Error AfterCreateDevice() {
    return Error::Ok;
  };
  const QnnImplementation& implementation_;

 private:
  Qnn_DeviceHandle_t handle_;
  QnnLogger* logger_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
