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
namespace executorch {
namespace backends {
namespace qnn {

class QnnProfile {
 public:
  explicit QnnProfile(
      QnnImplementation* implementation,
      QnnBackend* backend,
      const QnnExecuTorchProfileLevel& profile_level);
  ~QnnProfile();
  Qnn_ErrorHandle_t ProfileData(executorch::runtime::EventTracer* event_tracer);

  Qnn_ProfileHandle_t GetHandle() {
    return handle_;
  }

 private:
  Qnn_ProfileHandle_t handle_;
  QnnImplementation* implementation_;
  QnnBackend* backend_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
