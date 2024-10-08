/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/QnnProfiler.h>

namespace torch {
namespace executor {
namespace qnn {

QnnProfile::QnnProfile(
    const QnnImplementation& implementation,
    QnnBackend* backend,
    const QnnExecuTorchProfileLevel& profile_level)
    : handle_(nullptr), implementation_(implementation), backend_(backend) {
  if (profile_level != QnnExecuTorchProfileLevel::kProfileOff) {
    const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
    Qnn_ErrorHandle_t error = qnn_interface.qnn_profile_create(
        backend_->GetHandle(), static_cast<int>(profile_level), &handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_WARN(
          "Failed to create profile_handle for backend "
          " %u, error=%d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));

      // ignore error and continue to create backend handle...
      handle_ = nullptr;
    }
  }
}

Qnn_ErrorHandle_t QnnProfile::ProfileData(EventTracer* event_tracer) {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  const QnnProfile_EventId_t* events_ptr = nullptr;
  const QnnProfile_EventId_t* sub_events_ptr = nullptr;
  std::uint32_t num_events = 0;
  std::uint32_t num_sub_events = 0;
  Qnn_ErrorHandle_t error =
      qnn_interface.qnn_profile_get_events(handle_, &events_ptr, &num_events);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "ProfileData failed to get events: %d", QNN_GET_ERROR_CODE(error));
    return error;
  }
  QnnProfile_EventData_t event_data;
  for (std::uint32_t i = 0; i < num_events; ++i) {
    error =
        qnn_interface.qnn_profile_get_event_data(events_ptr[i], &event_data);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "ProfileData failed to get event data "
          "for event %d: %d",
          i,
          QNN_GET_ERROR_CODE(error));
      return error;
    }
    // Check an event's sub events only if it relates to graph execution time
    // (and its sub events are the individual op executions):
    if (backend_->IsProfileEventTypeParentOfNodeTime(event_data.type)) {
      error = qnn_interface.qnn_profile_get_sub_events(
          events_ptr[i], &sub_events_ptr, &num_sub_events);
      if (error != QNN_SUCCESS) {
        QNN_EXECUTORCH_LOG_ERROR(
            "ProfileData failed to get sub events "
            "for event %d: %d",
            i,
            QNN_GET_ERROR_CODE(error));
        return error;
      }
      QnnProfile_EventData_t sub_event_data;
      for (std::uint32_t j = 0; j < num_sub_events; ++j) {
        error = qnn_interface.qnn_profile_get_event_data(
            sub_events_ptr[j], &sub_event_data);
        if (error != QNN_SUCCESS) {
          QNN_EXECUTORCH_LOG_ERROR(
              "ProfileData failed to get sub "
              "event data for sub event %d of event %d: %d",
              j,
              i,
              QNN_GET_ERROR_CODE(error));
          return error;
        }
        if (sub_event_data.type == QNN_PROFILE_EVENTTYPE_NODE &&
            (sub_event_data.unit == QNN_PROFILE_EVENTUNIT_MICROSEC ||
             sub_event_data.unit == QNN_PROFILE_EVENTUNIT_CYCLES)) {
          torch::executor::event_tracer_log_profiling_delegate(
              event_tracer,
              sub_event_data.identifier,
              /*delegate_debug_id=*/
              static_cast<torch::executor::DebugHandle>(-1),
              0,
              sub_event_data.value);
        }
      }
    }
  }
  return error;
}

QnnProfile::~QnnProfile() {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  if (handle_ != nullptr) {
    Qnn_ErrorHandle_t error = qnn_interface.qnn_profile_free(handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to free QNN profile_handle. Backend "
          "ID %u, error %d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
    }
    handle_ = nullptr;
  }
}
} // namespace qnn
} // namespace executor
} // namespace torch
