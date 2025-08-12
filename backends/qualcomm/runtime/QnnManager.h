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
#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendFactory.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnDlcManager.h>
#include <executorch/runtime/core/error.h>

#include <memory>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace qnn {
class QnnManager {
 public:
  // Construct QnnManager
  explicit QnnManager(
      const QnnExecuTorchOptions* options,
      const QnnExecuTorchContextBinary& qnn_executorch_context_binary);

  ~QnnManager();
  executorch::runtime::Error Init();
  executorch::runtime::Error AllocateTensor(const std::string& graph_name);
  executorch::runtime::Error AllocateTensor(
      const std::string& graph_name,
      std::vector<std::shared_ptr<TensorWrapper>>& inputs,
      std::vector<std::shared_ptr<TensorWrapper>>& outputs);

  executorch::runtime::Error Execute(
      const std::string& graph_name,
      const std::vector<Qnn_Tensor_t>& input_tensor_structs,
      std::vector<Qnn_Tensor_t>& output_tensor_structs,
      executorch::runtime::EventTracer* event_tracer);

  executorch::runtime::Error ProfileExecuteData(
      const std::string& graph_name,
      executorch::runtime::EventTracer* event_tracer);

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

  executorch::runtime::Error GetContextBinary(
      QnnExecuTorchContextBinary& qnn_executorch_context_binary);

  executorch::runtime::Error CompileDlc();
  executorch::runtime::Error Compile(
      const std::string& graph_name,
      std::vector<std::shared_ptr<OpWrapper>>& op_wrappers);

  executorch::runtime::Error RegisterMem(
      void* data_ptr,
      const std::shared_ptr<TensorWrapper>& tensor_wrapper);

  // Pre-register custom memory handle from the SharedBuffer before execution
  executorch::runtime::Error PreRegisterMem();

  uint64_t GetSpillFillBufferSize() {
    auto* htp_backend_cache_ptr = static_cast<HtpBackendCache*>(
        backend_params_ptr_->qnn_backend_cache_ptr_.get());
    return htp_backend_cache_ptr->GetSpillFillBufferSize();
  }

  std::vector<std::shared_ptr<TensorWrapper>> GetGraphInputs(
      const std::string& graph_name) {
    return !input_tensors_.count(graph_name)
        ? std::vector<std::shared_ptr<TensorWrapper>>()
        : input_tensors_[graph_name];
  }

  std::vector<std::shared_ptr<TensorWrapper>> GetGraphOutputs(
      const std::string& graph_name) {
    return !output_tensors_.count(graph_name)
        ? std::vector<std::shared_ptr<TensorWrapper>>()
        : output_tensors_[graph_name];
  }

  std::vector<std::string> GetGraphNames() {
    return backend_params_ptr_->qnn_context_ptr_->GetGraphNames();
  }

  std::string GetBinarySignature();

 private:
  std::unique_ptr<const QnnSaver_Config_t*[]> GetImplementationConfig() {
    if (options_->saver()) {
      auto outputDirCfg = std::make_unique<QnnSaver_Config_t>();
      outputDirCfg->option = QNN_SAVER_CONFIG_OPTION_OUTPUT_DIRECTORY;
      outputDirCfg->outputDirectory = options_->saver_output_dir()->c_str();

      auto saverCfg = std::make_unique<const QnnSaver_Config_t*[]>(2);
      saverCfg[0] = outputDirCfg.release();
      saverCfg[1] = nullptr;

      return saverCfg;
    } else {
      return nullptr;
    }
  }

  executorch::runtime::Error LoadQnnLibrary();

  static constexpr const char* htp_library_name_ = "libQnnHtp.so";
  static constexpr const char* gpu_library_name_ = "libQnnGpu.so";
  static constexpr const char* dsp_library_name_ = "libQnnDsp.so";

  QnnExecuTorchContextBinary qnn_context_blob_;
  std::unique_ptr<BackendConfigParameters> backend_params_ptr_;
  QnnImplementation qnn_loaded_backend_;
  std::unique_ptr<QnnLogger> logger_;
  const QnnExecuTorchOptions* options_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<TensorWrapper>>>
      input_tensors_;
  std::unordered_map<std::string, std::vector<std::shared_ptr<TensorWrapper>>>
      output_tensors_;
  executorch::runtime::Error RegisterIonMem(
      void* data_ptr,
      const std::shared_ptr<TensorWrapper>& tensor_wrapper);
  executorch::runtime::Error RegisterCustomMem(
      void* data_ptr,
      void* custom_mem_base,
      const std::shared_ptr<TensorWrapper>& tensor_wrapper);
  std::unordered_map<Qnn_DataType_t, executorch::aten::ScalarType>
      qnn_dtype_to_scalar_type_ = {
          {Qnn_DataType_t::QNN_DATATYPE_INT_32,
           executorch::aten::ScalarType::Int},
          {Qnn_DataType_t::QNN_DATATYPE_FLOAT_32,
           executorch::aten::ScalarType::Float},
          {Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8,
           executorch::aten::ScalarType::Char},
          {Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_16,
           executorch::aten::ScalarType::Short},
          {Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_8,
           executorch::aten::ScalarType::Byte},
          {Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_16,
           executorch::aten::ScalarType::UInt16},
  };

  // Manager for handling DLC (Deep Learning Container)
  std::shared_ptr<QnnDlcManager> qnn_dlc_manager_;
};
} // namespace qnn
} // namespace backends
} // namespace executorch
