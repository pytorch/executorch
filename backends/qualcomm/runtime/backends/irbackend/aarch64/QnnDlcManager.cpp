/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnDlcManager.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>

namespace executorch {
namespace backends {
namespace qnn {

QnnDlcManager::QnnDlcManager(
    const QnnExecuTorchContextBinary& qnn_context_blob,
    const QnnExecuTorchOptions* options)
    : qnn_loaded_backend_(""),
      qnn_context_blob_(qnn_context_blob),
      options_(options) {
  if (options_ == nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Fail to create QnnDlcManager, options is nullptr");
  }
}

Error QnnDlcManager::LoadQnnIrLibrary() {
  return Error::Ok;
}

Error QnnDlcManager::Create() {
  return Error::Ok;
}

Error QnnDlcManager::Configure() {
  return Error::Ok;
}

Error QnnDlcManager::SetUpDlcEnvironment(const Qnn_Version_t& coreApiVersion) {
  return Error::Ok;
}

Error QnnDlcManager::RegisterGraphsFromDLC(
    const QnnImplementation& implementation,
    QnnBackend* backend,
    QnnContext* context,
    QnnBackendCache* cache) {
  void* lib_handle = dlopen(dlc_lib_, RTLD_NOW | RTLD_LOCAL);
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

  // memfd_create on android api level 30 and above
  int fd = -1;
#ifdef __ANDROID__
#if __ANDROID_API__ >= 30
  fd = memfd_create("tmp.dlc", 0);
#endif
#endif
  if (fd == -1) {
    QNN_EXECUTORCH_LOG_ERROR("memfd_create fail");
    return Error::Internal;
  }

  if (ftruncate(fd, qnn_context_blob_.nbytes) == -1) {
    QNN_EXECUTORCH_LOG_ERROR("ftruncate fail");
    close(fd);
    return Error::Internal;
  }

  void* addr = mmap(
      NULL,
      qnn_context_blob_.nbytes,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      fd,
      0);
  if (addr == MAP_FAILED) {
    QNN_EXECUTORCH_LOG_ERROR("mmap");
    close(fd);
    return Error::Internal;
  }

  memcpy(addr, qnn_context_blob_.buffer, qnn_context_blob_.nbytes);

  char dlc_path[256];
  snprintf(dlc_path, sizeof(dlc_path), "/proc/self/fd/%d", fd);

  const QNN_INTERFACE_VER_TYPE& interfaceVer =
      implementation.GetQnnInterface().GetInterfaceVer();

  if (composeGraphsFromDlc(
          /*backendHandle=*/backend->GetHandle(),
          /*interface=*/interfaceVer,
          /*contextHandle=*/context->GetHandle(),
          /*graphsConfigInfo=*/nullptr,
          /*dlcPath=*/dlc_path,
          /*numGraphsConfigInfo=*/0,
          /*graphsInfo=*/&qnn_dlc_graph_info_,
          /*numGraphsInfo=*/&qnn_dlc_graph_info_num_,
          /*debug=*/false,
          /*logCallback=*/nullptr,
          /*maxLogLevel=*/QNN_LOG_LEVEL_VERBOSE) !=
      qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR) {
    QNN_EXECUTORCH_LOG_ERROR("Failed to open Dlc");
    return Error::Internal;
  }
  munmap(addr, qnn_context_blob_.nbytes);
  close(fd);
  dlclose(lib_handle);

  for (uint32_t i = 0; i < qnn_dlc_graph_info_num_; ++i) {
    auto& graphInfo = (*qnn_dlc_graph_info_)[i];
    cache->SetGraphNames(graphInfo.graphName);
  }

  return Error::Ok;
}

void QnnDlcManager::ResetBackendParams() {}
void QnnDlcManager::ResetLogger() {}
void QnnDlcManager::TerminateAllBackends() {}

} // namespace qnn
} // namespace backends
} // namespace executorch
