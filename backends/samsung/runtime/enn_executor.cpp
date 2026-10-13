/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#include <executorch/backends/samsung/runtime/enn_executor.h>
#include <executorch/backends/samsung/runtime/enn_shared_memory_manager.h>
#include <executorch/backends/samsung/runtime/logging.h>
#include <executorch/backends/samsung/runtime/profile.hpp>

#include <inttypes.h>
#include <cstring>
#include <vector>

namespace torch {
namespace executor {
namespace enn {

Error EnnExecutor::initialize(const char* binary_buf_addr, size_t buf_size) {
  return initialize(binary_buf_addr, buf_size, nullptr, 0, false);
}

Error EnnExecutor::initialize(
    const char* binary_buf_addr,
    size_t buf_size,
    const char* weight_buf_addr,
    size_t weight_size,
    bool use_weight_fd) {
  EXYNOS_ATRACE_FUNCTION_LINE();
  auto sm_instance = executorch::backends::enn::shared_memory_manager::
      SharedMemoryManager::getInstance();
  const EnnApi* enn_api_inst = EnnApi::getEnnApiInstance();
  ET_CHECK_OR_RETURN_ERROR(
      enn_api_inst->isInitialized(), Internal, "Enn initialize failed.");
  // EnnInitialize() resets the calling thread's preference on every call, so
  // re-apply it right before opening the model.
  enn_api_inst->applyPerfConfigFromSoc();
  ET_CHECK_OR_RETURN_ERROR(
      buf_size <= UINT32_MAX && weight_size <= UINT32_MAX,
      InvalidArgument,
      "Model (%zu) or weight (%zu) buffer exceeds the 4GiB ENN limit",
      buf_size,
      weight_size);
  EnnReturn ret = ENN_RET_SUCCESS;

  ET_LOG(Info, "Start to open model %p, %zu", binary_buf_addr, buf_size);

  EnnBufferPtr pte_buffer = nullptr;
  if (sm_instance->query(&pte_buffer, binary_buf_addr, buf_size)) {
    int32_t fd;
    if (pte_buffer->va == binary_buf_addr &&
        !enn_api_inst->EnnGetFileDescriptorFromEnnBuffer(pte_buffer, &fd)) {
      if (use_weight_fd) {
        ET_CHECK_OR_RETURN_ERROR(
            enn_api_inst->EnnOpenModelFromFdWithWeight != nullptr,
            NotSupported,
            "This ENN runtime cannot open models with external weights.");
        EnnBufferPtr ptd_buffer = nullptr;
        ET_CHECK_OR_RETURN_ERROR(
            sm_instance->query(&ptd_buffer, weight_buf_addr, weight_size),
            Internal,
            "PTD EnnBuffer not found.");
        ET_LOG(
            Info,
            "Open FD & weights. fd: %d, model size %zu, offset: %u, weight buffer: %p",
            fd,
            buf_size,
            pte_buffer->offset,
            ptd_buffer->va);
        ret = enn_api_inst->EnnOpenModelFromFdWithWeight(
            fd, buf_size, pte_buffer->offset, &ptd_buffer, 1, &model_id_);
        ET_CHECK_OR_RETURN_ERROR(
            ret == ENN_RET_SUCCESS,
            Internal,
            "Open model with weights failed.");
        ET_LOG(Info, "Opened model from file descriptor with weights");
        sm_instance->free(pte_buffer->va);
      } else {
        ret = enn_api_inst->EnnOpenModelFromFd(fd, &model_id_);
        if (ret == ENN_RET_SUCCESS) {
          ET_LOG(Info, "Opened model from file descriptor, so fd is closed");
          sm_instance->free(pte_buffer->va);
        }
      }
    }
  }
  ET_CHECK_OR_RETURN_ERROR(
      !use_weight_fd || model_id_,
      Internal,
      "Failed to open the model with external weights from file descriptor.");
  if (!model_id_) {
    ET_LOG(Info, "Open model from memory");
    ret = enn_api_inst->EnnOpenModelFromMemory(
        binary_buf_addr, buf_size, &model_id_);
  }
  ET_CHECK_OR_RETURN_ERROR(
      ret == ENN_RET_SUCCESS,
      Internal,
      "Failed to load Enn model from buffer %d",
      (int)ret);
  ET_LOG(Info, "Open successfully.");

  NumberOfBuffersInfo buffers_info;
  ret = enn_api_inst->EnnAllocateAllBuffersWithSessionId(
      model_id_, &alloc_buffer_, &buffers_info, 0, true);
  ET_CHECK_OR_RETURN_ERROR(
      ret == ENN_RET_SUCCESS,
      Internal,
      "Failed to allocate buffers for model_id = 0x%" PRIX64,
      model_id_);
  num_of_inputs_ = buffers_info.n_in_buf;
  num_of_outputs_ = buffers_info.n_out_buf;

  return Error::Ok;
}

Error EnnExecutor::eval(
    const std::vector<DataBuffer>& inputs,
    const std::vector<DataBuffer>& outputs) {
  EXYNOS_ATRACE_FUNCTION_LINE();
  const EnnApi* enn_api_inst = EnnApi::getEnnApiInstance();
  ET_CHECK_OR_RETURN_ERROR(
      inputs.size() == getInputSize(),
      InvalidArgument,
      "Invalid number of inputs, expect %" PRIu32 " while get %ld",
      getInputSize(),
      inputs.size());
  ET_CHECK_OR_RETURN_ERROR(
      outputs.size() == getOutputSize(),
      InvalidArgument,
      "Invalid number of outputs, expected %" PRIu32 " while get %ld",
      getOutputSize(),
      outputs.size());

  EXYNOS_ATRACE_BEGIN("ExynosExecutor: memcpy buffer");
  int relative_input_index = 0;
  for (const auto& input : inputs) {
    EnnBufferPtr* input_buffer_ptr = alloc_buffer_;
    EnnBuffer& enn_buffer = *input_buffer_ptr[relative_input_index];
    memcpy(enn_buffer.va, input.buf_ptr_, input.size_);
    relative_input_index++;
  }
  EXYNOS_ATRACE_END();

  ENN_LOG_DEBUG("Starting model execution");
  auto ret = enn_api_inst->EnnExecuteModel(model_id_);
  if (ret != ENN_RET_SUCCESS) {
    ENN_LOG_ERROR(
        "EnnExecuteModel failed, ret=%d (model_id=0x%" PRIX64 ")",
        ret,
        model_id_);
    return Error::Internal;
  }

  EnnBufferPtr* output_buffer_ptr = alloc_buffer_ + getInputSize();
  int relative_output_index = 0;
  for (const auto& output : outputs) {
    EnnBuffer& enn_buffer = *output_buffer_ptr[relative_output_index];
    memcpy(output.buf_ptr_, enn_buffer.va, output.size_);
    relative_output_index++;
  }
  return Error::Ok;
}

EnnExecutor::~EnnExecutor() {
  EXYNOS_ATRACE_FUNCTION_LINE();
  const EnnApi* enn_api_inst = EnnApi::getEnnApiInstance();
  NumberOfBuffersInfo buffers_info;
  if (enn_api_inst->EnnGetBuffersInfo(model_id_, &buffers_info) ==
      ENN_RET_SUCCESS) {
    const int32_t num_of_buffers =
        buffers_info.n_in_buf + buffers_info.n_out_buf;
    enn_api_inst->EnnReleaseBuffers(alloc_buffer_, num_of_buffers);
  }
  enn_api_inst->EnnCloseModel(model_id_);

  ET_LOG(
      Info,
      "EnnExecutor destroyed: model 0x%" PRIX64 " closed, resources released",
      model_id_);
}

} // namespace enn
} // namespace executor
} // namespace torch
