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
      std::vector<Qnn_Tensor_t>& output_tensor_structs);

  Error ProfileExecuteData(EventTracer* event_tracer);

  void Destroy();

  bool IsAvailable() {
    return true;
  }

  bool IsOnlinePrepare() {
    return options_->online_prepare();
  }

  bool IsTensorDump() {
    return options_->tensor_dump_output_path()->size() > 0;
  }

  bool IsNodeSupportedByBackend(
      std::vector<std::shared_ptr<OpWrapper>>& op_wrappers);

  Error Compile(
      std::vector<std::shared_ptr<OpWrapper>>& op_wrappers,
      QnnExecuTorchContextBinary& qnn_executorch_context_binary);

  Error RegisterMem(
      void* data_ptr,
      const std::shared_ptr<TensorWrapper>& tensor_wrapper);

  std::vector<std::shared_ptr<TensorWrapper>> GetGraphInputs() {
    return input_tensors_;
  }
  std::vector<std::shared_ptr<TensorWrapper>> GetGraphOutputs() {
    return output_tensors_;
  }

 private:
  Error LoadQnnLibrary();

#ifdef _WIN32
  static constexpr const char* htp_library_name_ = "QnnHtp.dll";
  static constexpr const char* gpu_library_name_ = "QnnGpu.dll";
  static constexpr const char* dsp_library_name_ = "QnnDsp.dll";
#else
  static constexpr const char* htp_library_name_ = "libQnnHtp.so";
  static constexpr const char* gpu_library_name_ = "libQnnGpu.so";
  static constexpr const char* dsp_library_name_ = "libQnnDsp.so";
#endif

  QnnExecuTorchContextBinary qnn_context_blob_;
  std::unique_ptr<BackendConfigParameters> backend_params_ptr_;
  QnnImplementation qnn_loaded_backend_;
  std::unique_ptr<QnnLogger> logger_;
  const QnnExecuTorchOptions* options_;
  std::vector<std::shared_ptr<TensorWrapper>> input_tensors_;
  std::vector<std::shared_ptr<TensorWrapper>> output_tensors_;
};
} // namespace qnn
} // namespace executor
} // namespace torch
