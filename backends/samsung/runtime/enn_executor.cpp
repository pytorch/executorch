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

#include <android/log.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <vector>

namespace torch {
namespace executor {
namespace enn {

uint32_t get_size_from_fd(int fd) {
  if (fd < 0) {
    ET_LOG(Error, "get_size_from_fd(), invalid fd(%d)\n", fd);
    return 0;
  } else {
    off_t file_size = lseek(fd, 0, SEEK_END);
    if (file_size < 0) {
      return 0;
    } else {
      return static_cast<uint32_t>(file_size);
    }
  }
}

Error EnnExecutor::initialize(const char* binary_buf_addr, size_t buf_size) {
  EXYNOS_ATRACE_FUNCTION_LINE();
  auto _sm_instance = executorch::backends::enn::shared_memory_manager::
      SharedMemoryManager::getInstance();
  const EnnApi* enn_api_inst = EnnApi::getEnnApiInstance();
  EnnReturn ret;

  ET_LOG(Info, "Start to open model %p, %ld", binary_buf_addr, buf_size);

  EnnBufferPtr _out;
  if (_sm_instance->query(&_out, binary_buf_addr, buf_size)) {
    int fd;
    if (_out->va == binary_buf_addr &&
        !enn_api_inst->EnnGetFileDescriptorFromEnnBuffer(_out, &fd)) {
      ret = enn_api_inst->EnnOpenModelFromFd(fd, &model_id_);
      ET_LOG(Info, "Opened Model From File Descriptor");
      if (ret == ENN_RET_SUCCESS) {
        ET_LOG(Info, "Buffer Loading finished with fd, so fd would be closed");
        _sm_instance->free(_out->va);
      }
    }
  }
  if (!model_id_) {
    ET_LOG(Info, "Opened Model From Memory");
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

  ENN_LOG_DEBUG("Start to execute model.");
  auto ret = enn_api_inst->EnnExecuteModel(model_id_);
  if (ret != ENN_RET_SUCCESS) {
    ENN_LOG_ERROR("EnnExecuteModel Failed");
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
}

} // namespace enn
} // namespace executor
} // namespace torch
