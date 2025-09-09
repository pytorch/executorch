/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */
#include <executorch/backends/samsung/runtime/enn_executor.h>
#include <executorch/backends/samsung/runtime/logging.h>
#include <inttypes.h>

#include <fstream>
#include <string>
#include <vector>

namespace torch {
namespace executor {
namespace enn {

Error EnnExecutor::initialize(const char* binary_buf_addr, size_t buf_size) {
  const EnnApi* enn_api_inst = EnnApi::getEnnApiInstance();
  auto ret = enn_api_inst->EnnInitialize();
  ET_CHECK_OR_RETURN_ERROR(
      ret == ENN_RET_SUCCESS, Internal, "Enn initialize failed.");

  ET_LOG(Info, "Start to open model %p, %ld", binary_buf_addr, buf_size);
  ret = enn_api_inst->EnnOpenModelFromMemory(
      binary_buf_addr, buf_size, &model_id_);

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

  int relative_input_index = 0;
  for (const auto& input : inputs) {
    EnnBufferPtr* input_buffer_ptr = alloc_buffer_;
    EnnBuffer& enn_buffer = *input_buffer_ptr[relative_input_index];
    memcpy(enn_buffer.va, input.buf_ptr_, input.size_);
    relative_input_index++;
  }

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
