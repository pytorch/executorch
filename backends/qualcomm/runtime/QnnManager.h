/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/qualcomm/aot/wrappers/OpWrapper.h>
#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendFactory.h>
#include <executorch/backends/qualcomm/schema_generated.h>
#include <executorch/runtime/core/error.h>

#include <memory>
#include <unordered_map>

namespace torch {
namespace executor {
namespace qnn {
class QnnManager {
 public:
  // Construct QnnManager
  explicit QnnManager(
      const QnnExecuTorchOptions* options,
      const QnnExecuTorchContextBinary& qnn_executorch_context_binary);

  ~QnnManager();
  Error Init();
  Error AllocateTensor();
  Error AllocateTensor(
      std::vector<std::shared_ptr<TensorWrapper>>& inputs,
      std::vector<std::shared_ptr<TensorWrapper>>& outputs);

  Error Execute(
      const std::vector<Qnn_Tensor_t>& input_tensor_structs,
      std::vector<Qnn_Tensor_t>& output_tensor_structs,
      EventTracer* event_tracer);

  Error ProfileExecuteData(EventTracer* event_tracer);

  void Destroy();

  bool IsAvailable() {
    return true;
  }

  bool IsOnlinePrepare() {
    return options_->online_prepare();
  }

  bool IsTensorDump() {
    return options_->dump_intermediate_outputs();
  }

  bool IsNodeSupportedByBackend(
      std::vector<std::shared_ptr<OpWrapper>>& op_wrappers);

  Error Compile(
      std::vector<std::shared_ptr<OpWrapper>>& op_wrappers,
      QnnExecuTorchContextBinary& qnn_executorch_context_binary);

  Error RegisterMem(
      void* data_ptr,
      const std::shared_ptr<TensorWrapper>& tensor_wrapper);

  // Pre-register custom memory handle from the SharedBuffer before execution
  Error PreRegisterMem();

  uint64_t GetSpillFillBufferSize() {
    auto* htp_backend_cache_ptr = static_cast<HtpBackendCache*>(
        backend_params_ptr_->qnn_backend_cache_ptr_.get());
    return htp_backend_cache_ptr->GetSpillFillBufferSize();
  }

  std::vector<std::shared_ptr<TensorWrapper>> GetGraphInputs() {
    return input_tensors_;
  }
  std::vector<std::shared_ptr<TensorWrapper>> GetGraphOutputs() {
    return output_tensors_;
  }

 private:
  Error LoadQnnLibrary();

  static constexpr const char* htp_library_name_ = "libQnnHtp.so";
  static constexpr const char* gpu_library_name_ = "libQnnGpu.so";
  static constexpr const char* dsp_library_name_ = "libQnnDsp.so";

  QnnExecuTorchContextBinary qnn_context_blob_;
  std::unique_ptr<BackendConfigParameters> backend_params_ptr_;
  QnnImplementation qnn_loaded_backend_;
  std::unique_ptr<QnnLogger> logger_;
  const QnnExecuTorchOptions* options_;
  std::vector<std::shared_ptr<TensorWrapper>> input_tensors_;
  std::vector<std::shared_ptr<TensorWrapper>> output_tensors_;
  Error RegisterIonMem(
      void* data_ptr,
      const std::shared_ptr<TensorWrapper>& tensor_wrapper);
  Error RegisterCustomMem(
      void* data_ptr,
      void* custom_mem_base,
      const std::shared_ptr<TensorWrapper>& tensor_wrapper);
  std::unordered_map<Qnn_DataType_t, ScalarType> qnn_dtype_to_scalar_type_ = {
      {Qnn_DataType_t::QNN_DATATYPE_INT_32, ScalarType::Int},
      {Qnn_DataType_t::QNN_DATATYPE_FLOAT_32, ScalarType::Float},
      {Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8, ScalarType::Char},
      {Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_16, ScalarType::Short},
      {Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_8, ScalarType::Byte},
      {Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_16, ScalarType::Bits16},
  };
};
} // namespace qnn
} // namespace executor
} // namespace torch
