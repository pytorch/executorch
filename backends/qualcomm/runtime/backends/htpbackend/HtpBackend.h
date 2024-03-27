/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include "HTP/QnnHtpProfile.h"
namespace torch {
namespace executor {
namespace qnn {
class HtpBackend : public QnnBackend {
 public:
  HtpBackend(const QnnImplementation& implementation, QnnLogger* logger)
      : QnnBackend(implementation, logger) {}
  ~HtpBackend() {}

  bool IsProfileEventTypeParentOfNodeTime(
      QnnProfile_EventType_t event_type) override {
    return (
        event_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_CYCLE);
  }

 protected:
  Error MakeConfig(std::vector<const QnnBackend_Config_t*>& config) override {
    return Error::Ok;
  }
};
} // namespace qnn
} // namespace executor
} // namespace torch
