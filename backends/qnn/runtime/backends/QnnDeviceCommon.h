/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_DEVICE_COMMON_H_
#define EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_DEVICE_COMMON_H_

#include <executorch/backends/qnn/runtime/Logging.h>
#include <executorch/backends/qnn/runtime/QnnExecuTorch.h>
#include <executorch/backends/qnn/runtime/backends/QnnImplementation.h>
#include <executorch/backends/qnn/runtime/backends/QnnLogger.h>

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

  Qnn_DeviceHandle_t GetHandle() { return handle_; }

  Error Configure();

 protected:
  virtual Error MakeConfig(std::vector<const QnnDevice_Config_t*>& config) {
    return Error::Ok;
  };

  virtual Error AfterCreateDevice() { return Error::Ok; };
  const QnnImplementation& implementation_;

 private:
  Qnn_DeviceHandle_t handle_;
  QnnLogger* logger_;
};
}  // namespace qnn
}  // namespace executor
}  // namespace torch
#endif  // EXECUTORCH_QNN_EXECUTORCH_BACKENDS_QNN_DEVICE_COMMON_H_
