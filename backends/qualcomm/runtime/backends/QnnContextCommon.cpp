/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnDlcManager.h>

namespace executorch {
namespace backends {
namespace qnn {

std::mutex QnnContext::htp_context_mutex_;
int QnnContext::htp_context_count_{0};

void QnnContext::WriteHeapProfile() {
  executorch::runtime::BackendOption backend_option;
  std::string heap_profiling_path;
  if (get_runtime_option(QNN_RUNTIME_HEAP_PROFILING_PATH, backend_option) ==
      Error::Ok) {
    auto* arr = std::get_if<std::array<char, runtime::kMaxOptionValueLength>>(
        &backend_option.value);
    if (arr) {
      heap_profiling_path = arr->data();
    }
  }
  Qnn_ErrorHandle_t error_profile =
      qnn_profiler_->ProfileDataToFile(heap_profiling_path);
  if (error_profile != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to profile. Cannot get profile from handle. Error %d",
        QNN_GET_ERROR_CODE(error_profile));
  }
}

QnnContext::~QnnContext() {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  if (handle_ != nullptr) {
    QNN_EXECUTORCH_LOG_INFO("Destroy Qnn context");

    bool do_heap_profile = false;
    {
      std::lock_guard<std::mutex> lock(htp_context_mutex_);
      if (is_htp_backend_ && htp_context_count_ > 0 && need_to_profile_) {
        --htp_context_count_;
        do_heap_profile = (htp_context_count_ == 0);
      }
    }
    error = qnn_interface.qnn_context_free(
        handle_, do_heap_profile ? qnn_profiler_->GetHandle() : nullptr);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to free QNN "
          "context_handle_. Backend "
          "ID %u, error %d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
    } else if (do_heap_profile) {
      WriteHeapProfile();
    }
    handle_ = nullptr;
  }
}

Error QnnContext::Configure() {
  // create qnn context
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  std::vector<const QnnContext_Config_t*> temp_context_config;
  ET_CHECK_OR_RETURN_ERROR(
      MakeConfig(temp_context_config) == Error::Ok,
      Internal,
      "Fail to make context config.");

  if (cache_->GetCacheState() == QnnBackendCache::DESERIALIZE) {
    const QnnExecuTorchContextBinary& qnn_context_blob =
        cache_->GetQnnContextBlob();
    /*
    Total DSP heap usage can be measured in two conditions, first context
    creation and last context free. By the QNN documentation, we need to insert
    profileHandle in qnn_context_create_from_binary when creating first context
    and closing last context.

    Limitations are two:
    1.Only supported on Android and QNX platforms.
    2.By enabling this feature initialization and cleanup time might be
    impacted.
    */

    bool do_heap_profile = false;
    {
      std::lock_guard<std::mutex> lock(htp_context_mutex_);
      do_heap_profile =
          is_htp_backend_ && (htp_context_count_ == 0) && need_to_profile_;
      if (is_htp_backend_) {
        ++htp_context_count_;
      }
    }

    error = qnn_interface.qnn_context_create_from_binary(
        backend_->GetHandle(),
        device_->GetHandle(),
        (temp_context_config.empty() ? nullptr : temp_context_config.data()),
        static_cast<uint8_t*>(qnn_context_blob.buffer),
        qnn_context_blob.nbytes,
        &handle_,
        do_heap_profile ? qnn_profiler_->GetHandle() : nullptr);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Can't create context from "
          "binary. Error %d.",
          QNN_GET_ERROR_CODE(error));
      // Rollback the count since context creation failed
      {
        std::lock_guard<std::mutex> lock(htp_context_mutex_);
        if (is_htp_backend_ && htp_context_count_ > 0) {
          --htp_context_count_;
        }
      }
      return Error::Internal;
    } else if (do_heap_profile) {
      WriteHeapProfile();
    }
  } else if (
      cache_->GetCacheState() == QnnBackendCache::SERIALIZE ||
      cache_->GetCacheState() == QnnBackendCache::ONLINE_PREPARE ||
      cache_->GetCacheState() == QnnBackendCache::MULTI_GRAPH) {
    error = qnn_interface.qnn_context_create(
        backend_->GetHandle(),
        device_->GetHandle(),
        temp_context_config.empty() ? nullptr : temp_context_config.data(),
        &handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to create QNN context for Backend "
          "ID %u, error=%d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  } else {
    QNN_EXECUTORCH_LOG_ERROR("QNN context cache is invalid.");
    return Error::Internal;
  }
  if (AfterConfigure() != Error::Ok) {
    return Error::Internal;
  }
  if (cache_->GetCacheState() == QnnBackendCache::ONLINE_PREPARE) {
    // Register graphs from DLC during online prepare for HTP/GPU/DSP backends
    return qnn_dlc_manager_->RegisterGraphsFromDLC(
        implementation_, backend_, this, cache_);
  }
  return Error::Ok;
}

Error QnnContext::GetContextBinary(
    QnnExecuTorchContextBinary& qnn_executorch_context_binary) {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ContextBinarySize_t binary_size = 0;
  Qnn_ContextBinarySize_t bytes_written = 0;
  Qnn_ErrorHandle_t error =
      qnn_interface.qnn_context_get_binary_size(handle_, &binary_size);
  if (error == QNN_SUCCESS) {
    // create our own protocol here
    qnn_context_custom_protocol_ = QnnContextCustomProtocol(binary_size);
    qnn_context_custom_protocol_.BuildContextCustomBuffer();
    auto [context_buffer_ptr, context_buffer_size] =
        qnn_context_custom_protocol_.GetCustomProtocolBuffer();
    error = qnn_interface.qnn_context_get_binary(
        handle_,
        static_cast<uint8_t*>(context_buffer_ptr) +
            qnn_context_custom_protocol_.GetContextBinaryOffset(),
        binary_size,
        &bytes_written);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Can't get graph binary to be saved to "
          "cache. Error %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    } else {
      if (binary_size < bytes_written) {
        QNN_EXECUTORCH_LOG_ERROR(
            "Illegal written buffer size [%d] bytes. Cannot "
            "exceed allocated memory of [%d] bytes",
            bytes_written,
            binary_size);
        return Error::Internal;
      }

      qnn_executorch_context_binary.buffer = context_buffer_ptr;
      qnn_executorch_context_binary.nbytes = context_buffer_size;
    }
  } else {
    QNN_EXECUTORCH_LOG_ERROR(
        "Can't determine the size of "
        "graph binary to be saved to cache. Error %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  return Error::Ok;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
