/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EXECUTORCH_QNN_EXECUTORCH_LOGGING_H_
#define EXECUTORCH_QNN_EXECUTORCH_LOGGING_H_
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/runtime/core/error.h>
namespace torch {
namespace executor {
namespace qnn {
void Log(QnnExecuTorchLogLevel log_level, const char* format, ...);

#define QNN_EXECUTORCH_LOG(log_level, format, ...) \
  do {                                             \
    Log(log_level, format, ##__VA_ARGS__);         \
  } while (false);
}  // namespace qnn
}  // namespace executor
}  // namespace torch
#endif  // EXECUTORCH_QNN_EXECUTORCH_LOGGING_H_
