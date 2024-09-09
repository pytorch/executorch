/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include "HTP/QnnHtpCommon.h"
#include "HTP/QnnHtpProfile.h"
#include "QnnTypes.h"
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

  Qnn_Version_t GetExpectedBackendVersion() const override {
    Qnn_Version_t backend_version;
    backend_version.major = QNN_HTP_API_VERSION_MAJOR;
    backend_version.minor = QNN_HTP_API_VERSION_MINOR;
    backend_version.patch = QNN_HTP_API_VERSION_PATCH;
    return backend_version;
  }

 protected:
  Error MakeConfig(std::vector<const QnnBackend_Config_t*>& config) override {
    return Error::Ok;
  }
};
} // namespace qnn
} // namespace executor
} // namespace torch
