/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <executorch/runtime/core/event_tracer_hooks_delegate.h>
#include "QnnProfile.h"
namespace torch {
namespace executor {
namespace qnn {

class QnnProfile {
 public:
  explicit QnnProfile(
      const QnnImplementation& implementation,
      QnnBackend* backend,
      const QnnExecuTorchProfileLevel& profile_level);
  ~QnnProfile();
  Qnn_ErrorHandle_t ProfileData(EventTracer* event_tracer);

  Qnn_ProfileHandle_t GetHandle() {
    return handle_;
  }

 private:
  Qnn_ProfileHandle_t handle_;
  const QnnImplementation& implementation_;
  QnnBackend* backend_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
