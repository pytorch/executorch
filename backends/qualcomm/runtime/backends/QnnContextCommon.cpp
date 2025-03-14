/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dlfcn.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>

namespace executorch {
namespace backends {
namespace qnn {

template <typename Fn>
Fn loadQnnFunction(void* handle, const char* function_name) {
  return reinterpret_cast<Fn>(dlsym(handle, function_name));
}

QnnContext::~QnnContext() {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (handle_ != nullptr) {
    QNN_EXECUTORCH_LOG_INFO("Destroy Qnn context");
    error = qnn_interface.qnn_context_free(handle_, /*profile=*/nullptr);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to free QNN "
          "context_handle_. Backend "
          "ID %u, error %d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
    }
    handle_ = nullptr;
  }
}

Error QnnContext::RegisterGraphsFromDLC() {
  const QnnExecuTorchContextBinary& qnn_context_blob =
      cache_->GetQnnContextBlob();

  int fd = memfd_create("tmp.dlc", 0);
  if (fd == -1) {
    perror("memfd_create fail");
    return Error::Internal;
  }

  if (ftruncate(fd, qnn_context_blob.nbytes) == -1) {
    perror("ftruncate fail");
    close(fd);
    return Error::Internal;
  }

  void* addr = mmap(
      NULL, qnn_context_blob.nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    perror("mmap");
    close(fd);
    return Error::Internal;
  }

  memcpy(addr, qnn_context_blob.buffer, qnn_context_blob.nbytes);

  char dlc_path[256];
  snprintf(dlc_path, sizeof(dlc_path), "/proc/self/fd/%d", fd);

  const QNN_INTERFACE_VER_TYPE& interfaceVer =
      implementation_.GetQnnInterface().GetInterfaceVer();

  // Get compose graph dlc
  void* lib_handle = dlopen(dlc_lib_, RTLD_NOW | RTLD_GLOBAL);
  if (lib_handle == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Cannot Open lib %s, with error: %s", dlc_lib_, dlerror());
    return Error::Internal;
  }

  QnnModel_composeGraphsFromDlc composeGraphsFromDlc =
      loadQnnFunction<QnnModel_composeGraphsFromDlc>(
          lib_handle, "QnnModel_composeGraphsFromDlc");

  if (composeGraphsFromDlc == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Cannot load symbol "
        "QnnModel_composeGraphsFromDlc : %s",
        dlerror());
    return Error::Internal;
  }

  if (composeGraphsFromDlc(
          /*backendHandle=*/backend_->GetHandle(),
          /*interface=*/interfaceVer,
          /*contextHandle=*/GetHandle(),
          /*graphsConfigInfo=*/nullptr,
          /*dlcPath=*/dlc_path,
          /*numGraphsConfigInfo=*/0,
          /*graphsInfo=*/&p_graph_info_,
          /*numGraphsInfo=*/&graph_info_num_,
          /*debug=*/false,
          /*logCallback=*/nullptr,
          /*maxLogLevel=*/QNN_LOG_LEVEL_VERBOSE) !=
      qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR) {
    QNN_EXECUTORCH_LOG_ERROR("Failed to open Dlc");
    return Error::Internal;
  }
  close(fd);

  for (uint32_t i = 0; i < graph_info_num_; ++i) {
    auto& graphInfo = (*p_graph_info_)[i];
    cache_->SetGraphNames(graphInfo.graphName);
  }

  return Error::Ok;
}

Error QnnContext::Configure() {
  // create qnn context
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  std::vector<const QnnContext_Config_t*> temp_context_config;
  ET_CHECK_OR_RETURN_ERROR(
      MakeConfig(temp_context_config) == Error::Ok,
      Internal,
      "Fail to make context config.");

  if (cache_->GetCacheState() == QnnBackendCache::DESERIALIZE) {
    const QnnExecuTorchContextBinary& qnn_context_blob =
        cache_->GetQnnContextBlob();

    error = qnn_interface.qnn_context_create_from_binary(
        backend_->GetHandle(),
        device_->GetHandle(),
        temp_context_config.empty() ? nullptr : temp_context_config.data(),
        static_cast<uint8_t*>(qnn_context_blob.buffer),
        qnn_context_blob.nbytes,
        &handle_,
        /*profile=*/nullptr);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Can't create context from "
          "binary. Error %d.",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
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
    return RegisterGraphsFromDLC();
  }
  return Error::Ok;
}

Error QnnContext::GetContextBinaryFromDLC(
    QnnExecuTorchContextBinary& qnn_executorch_context_binary) {
  // qnn_context_get_binary is not supported on IrBackend
  // read DLC and write to buffer
  auto dlc_name = GetGraphNames()[0] + ".dlc";
  std::ifstream dlc_file(dlc_name, std::ios::binary | std::ios::ate);
  if (dlc_file.is_open()) {
    std::streamsize size = dlc_file.tellg();
    dlc_file.seekg(0, std::ios::beg);

    auto buffer = std::make_shared<std::vector<char>>(size);
    dlc_file.read(buffer->data(), size);
    dlc_file.close();
    qnn_executorch_context_binary.buffer = buffer->data();
    qnn_executorch_context_binary.nbytes = size;
    static std::shared_ptr<std::vector<char>> static_buffer = buffer;
    return Error::Ok;
  } else {
    QNN_EXECUTORCH_LOG_ERROR(
        "Unable to open dlc file %s for building QnnExecuTorchContextBinary",
        dlc_name.c_str());
  }
  return Error::Internal;
}
// std::vector<char> buffer(size);
Error QnnContext::GetContextBinary(
    QnnExecuTorchContextBinary& qnn_executorch_context_binary) {
  const QnnInterface& qnn_interface = implementation_.GetQnnInterface();
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
