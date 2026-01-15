/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>
#include <executorch/backends/qualcomm/runtime/Utils.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnCustomProtocol.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <executorch/extension/tensor/tensor.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace qnn {

using executorch::runtime::Error;

bool CompareExportedInput(
    const std::shared_ptr<TensorWrapper>& a,
    const std::shared_ptr<TensorWrapper>& b) {
  // Using the order of the nodes as external_id in AOT
  // to extract the right arg from *args at runtime
  int numA = std::stoi(a->GetName().substr(a->GetName().find('_') + 1));
  int numB = std::stoi(b->GetName().substr(b->GetName().find('_') + 1));
  return numA < numB;
}

int ExtractMutableBufferNumber(const std::string& name) {
  std::string prefix = "mutbuf_";
  size_t startPos = name.find(prefix);
  if (startPos != std::string::npos) {
    startPos += prefix.length();
    return std::stoi(name.substr(startPos));
  }
  return -1;
}

QnnManager::~QnnManager() {
  Destroy();
}

QnnManager::QnnManager(
    const QnnExecuTorchOptions* options,
    const QnnExecuTorchContextBinary& qnn_executorch_context_binary)
    : qnn_context_blob_(qnn_executorch_context_binary), options_(options) {
  QnnExecuTorchBackendType backend_type =
      options->backend_options()->backend_type();

  if (get_option(options_->log_level()) >=
      QnnExecuTorchLogLevel::kLogLevelInfo) {
    QNN_EXECUTORCH_LOG_INFO(
        "soc_model in soc_info: %s",
        EnumNameQcomChipset(options_->soc_info()->soc_model()));
    QNN_EXECUTORCH_LOG_INFO(
        "backend_type: %s", EnumNameQnnExecuTorchBackendType(backend_type));
    QNN_EXECUTORCH_LOG_INFO(
        "library_path: %s", options->library_path()->str().c_str());
    QNN_EXECUTORCH_LOG_INFO("dump intermediate outputs: %s", IsTensorDump());
    QNN_EXECUTORCH_LOG_INFO(
        "log_level: %s",
        EnumNameQnnExecuTorchLogLevel(get_option(options_->log_level())));
    QNN_EXECUTORCH_LOG_INFO(
        "profile_level: %s",
        EnumNameQnnExecuTorchProfileLevel(
            get_option(options_->profile_level())));
    QNN_EXECUTORCH_LOG_INFO(
        "the size of qnn context binary: %d",
        qnn_executorch_context_binary.nbytes);
    QNN_EXECUTORCH_LOG_INFO(
        "Is on-device graph construction: %d", options->online_prepare());
    QNN_EXECUTORCH_LOG_INFO(
        "Enable shared buffer: %d", options->shared_buffer());
    QNN_EXECUTORCH_LOG_INFO(
        "The number of op packages: %d",
        options_->op_package_options()->op_package_infos()->size());
  }

  backend_params_ptr_ = std::make_unique<BackendConfigParameters>();
  backend_bundle_ptr_ = std::make_shared<QnnBackendBundle>();

  qnn_dlc_manager_ =
      std::make_shared<QnnDlcManager>(qnn_context_blob_, options_);
}

Error QnnManager::RegisterMem(
    void* data_ptr,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();
  // Not enable shared buffer
  if (!options_->shared_buffer()) {
    return Error::Internal;
  }

  if (backend_params_ptr_->qnn_mem_manager_ptr_ == nullptr) {
    QNN_EXECUTORCH_LOG_WARN(
        "Backend %s doesn't supported shared buffer.",
        EnumNameQnnExecuTorchBackendType(
            options_->backend_options()->backend_type()));
    return Error::Internal;
  }

  void* custom_mem_base = shared_buffer_manager.GetCustomMemBase(data_ptr);
  if (custom_mem_base != nullptr) {
    return RegisterCustomMem(data_ptr, custom_mem_base, tensor_wrapper);
  }
  return RegisterIonMem(data_ptr, tensor_wrapper);
}

Error QnnManager::RegisterIonMem(
    void* data_ptr,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();
  if (!shared_buffer_manager.IsAllocated(data_ptr)) {
    // It means two scenarios here:
    // 1. the input and output partitioned graph
    // 2. Actually, user doesn't allocate shared buffer with
    // QnnExecuTorchAllocCustomMem API
    return Error::Internal;
  } else if (backend_params_ptr_->qnn_mem_manager_ptr_->IsRegistered(
                 tensor_wrapper->GetMemHandle(), data_ptr)) {
    if (get_option(options_->log_level()) >=
        QnnExecuTorchLogLevel::kLogLevelInfo)
      QNN_EXECUTORCH_LOG_INFO(
          "Tensor name %s has been registered shared memory.",
          tensor_wrapper->GetName().c_str());
    return Error::Ok;
  }

  int32_t mem_fd = shared_buffer_manager.MemToFd(data_ptr);
  if (mem_fd == -1) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor name %s is failed to get file descriptor.",
        tensor_wrapper->GetName().c_str());
    return Error::Internal;
  }
  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_->qnn_mem_manager_ptr_->RegisterIonMem(
          tensor_wrapper, mem_fd, data_ptr) == Error::Ok,
      Internal,
      "Fail to register to shared memory.");

  return Error::Ok;
}

Error QnnManager::RegisterCustomMem(
    void* data_ptr,
    void* custom_mem_base,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  if (backend_params_ptr_->qnn_mem_manager_ptr_->IsRegistered(
          tensor_wrapper->GetMemHandle(), data_ptr)) {
    if (get_option(options_->log_level()) >=
        QnnExecuTorchLogLevel::kLogLevelInfo)
      QNN_EXECUTORCH_LOG_INFO(
          "Tensor name %s has been registered shared memory.",
          tensor_wrapper->GetName().c_str());
    return Error::Ok;
  }

  CustomMemTensorInfo info{
      custom_mem_base,
      data_ptr,
      static_cast<size_t>(
          static_cast<char*>(data_ptr) - static_cast<char*>(custom_mem_base)),
      tensor_wrapper->GetBytes(),
      tensor_wrapper->GetDims(),
      tensor_wrapper->GetRank(),
      qnn_dtype_to_scalar_type_[tensor_wrapper->GetDataType()]};

  Qnn_MemHandle_t pre_registered_handle =
      backend_params_ptr_->qnn_mem_manager_ptr_->GetPreRegisteredHandle(info);
  // If this memory block has already been registered, we can use it directly.
  // This applies when running llama in lookahead mode with the same AR-N model
  // handling both the prompt processor and the token generator.
  if (pre_registered_handle != nullptr) {
    if (get_option(options_->log_level()) >=
        QnnExecuTorchLogLevel::kLogLevelInfo) {
      QNN_EXECUTORCH_LOG_INFO(
          "Tensor name %s found a pre-registered memHandle.",
          tensor_wrapper->GetName().c_str());
    }
    return backend_params_ptr_->qnn_mem_manager_ptr_->SetMemHandle(
        tensor_wrapper, data_ptr, pre_registered_handle);
  }

  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();

  size_t tensor_offset = info.pos;
  size_t total_custom_mem_size =
      shared_buffer_manager.GetAllocatedSize(custom_mem_base);

  int32_t mem_fd = shared_buffer_manager.MemToFd(custom_mem_base);
  // Note: If obtaining the file descriptor fails, it may be due to memory not
  // being released with QnnExecuTorchFreeCustomMem. In this situation, we could
  // consider adding a map to monitor it.
  if (mem_fd == -1) {
    QNN_EXECUTORCH_LOG_WARN(
        "Tensor name %s failed to get file descriptor.",
        tensor_wrapper->GetName().c_str());
    return Error::Internal;
  }

  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_->qnn_mem_manager_ptr_->RegisterCustomMem(
          tensor_wrapper,
          mem_fd,
          data_ptr,
          total_custom_mem_size,
          tensor_offset,
          info) == Error::Ok,
      Internal,
      "Fail to register to shared memory.");

  return Error::Ok;
}

Error QnnManager::InitBackend() {
  // Get or create the shared backend bundle
  Error err = QnnBackendUnifiedRegistry::GetInstance().GetOrCreateBackendBundle(
      options_, backend_bundle_ptr_);
  ET_CHECK_OR_RETURN_ERROR(
      err == Error::Ok,
      Internal,
      "Fail to get or create shared Qnn backend bundle. Error code: %d",
      static_cast<int>(err));
  return Error::Ok;
}

Error QnnManager::InitContext(
    std::optional<std::vector<std::string>> graph_names) {
  if (backend_params_ptr_->backend_init_state_ ==
      BackendInitializeState::UNINITIALIZED) {
    QNN_EXECUTORCH_LOG_INFO(
        "Initialize Qnn backend "
        "parameters for Qnn executorch backend type %d",
        options_->backend_options()->backend_type());
    backend_params_ptr_ = QnnBackendFactory().Create(
        backend_bundle_ptr_->implementation.get(),
        backend_bundle_ptr_->qnn_backend_ptr.get(),
        backend_bundle_ptr_->qnn_device_ptr.get(),
        qnn_context_blob_,
        options_,
        qnn_dlc_manager_.get());
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_ != nullptr,
        Internal,
        "Failed to load Qnn backend.");
    // Note: For online_prepare or deserialization, the graph name will be
    // obtained from the binary.
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_backend_cache_ptr_->Configure(
            graph_names.value_or(std::vector<std::string>{})) == Error::Ok,
        Internal,
        "Fail to configure Qnn backend cache");
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_context_ptr_->Configure() == Error::Ok,
        Internal,
        "Fail to configure Qnn context");
    for (const std::string& graph_name :
         backend_params_ptr_->qnn_context_ptr_->GetGraphNames()) {
      ET_CHECK_OR_RETURN_ERROR(
          backend_params_ptr_->qnn_graph_ptr_->Configure(graph_name) ==
              Error::Ok,
          Internal,
          "Fail to configure Qnn graph");
    }

    backend_params_ptr_->backend_init_state_ =
        BackendInitializeState::INITIALIZED;
  }

  if (IsOnlinePrepare()) {
    // Check whether the QNN version supports the DLC format.
    Qnn_ApiVersion_t qnn_version = {QNN_VERSION_INIT};
    backend_bundle_ptr_->implementation->GetQnnInterface()
        .qnn_backend_get_api_version(&qnn_version);

    ET_CHECK_OR_RETURN_ERROR(
        qnn_dlc_manager_->SetUpDlcEnvironment(
            qnn_version.coreApiVersion,
            graph_names.value_or(std::vector<std::string>{})) == Error::Ok,
        Internal,
        "Fail to setup Dlc environment");
  }
  return Error::Ok;
}

Error QnnManager::AllocateTensor(const std::string& graph_name) {
  std::vector<Qnn_Tensor_t> input_tensors =
      backend_params_ptr_->qnn_context_ptr_->GetGraphInputs(graph_name);
  std::vector<Qnn_Tensor_t> output_tensors =
      backend_params_ptr_->qnn_context_ptr_->GetGraphOutputs(graph_name);

  // Mapping memory address for the input and output of mutable buffer
  std::unordered_map<int, const void*> mutable_buffer_id_to_memory_map;

  for (auto& tensor : input_tensors) {
    std::shared_ptr<TensorWrapper> tensor_wrapper = CreateTensorWrapper(tensor);
    tensor_wrapper->UpdateQnnTensorMeta(tensor);

    int mutable_buffer_id =
        ExtractMutableBufferNumber(tensor_wrapper->GetName());
    if (mutable_buffer_id != -1) {
      // Delegate maintains the memory for mutable buffer
      tensor_wrapper->AllocateDataBuffer();
      mutable_buffer_id_to_memory_map[mutable_buffer_id] =
          tensor_wrapper->GetStaticTensorData();
    }
    input_tensors_[graph_name].emplace_back(std::move(tensor_wrapper));
  }
  if (!options_->is_from_context_binary()) {
    std::sort(
        input_tensors_[graph_name].begin(),
        input_tensors_[graph_name].end(),
        CompareExportedInput);
  }
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    std::shared_ptr<TensorWrapper> tensor_wrapper =
        CreateTensorWrapper(output_tensors[i]);
    tensor_wrapper->UpdateQnnTensorMeta(output_tensors[i]);
    const std::string& tensor_name = tensor_wrapper->GetName();
    // this is required by identifying shared buffer mechanism
    // info might be missed if context binary came from qnn_converter
    if (options_->is_from_context_binary() &&
        tensor_name.find("output_") == std::string::npos) {
      tensor_wrapper->SetName("output_" + tensor_name);
    }
    if (IsTensorDump()) {
      tensor_wrapper->AllocateDataBuffer();
    }
    int mutable_buffer_id =
        ExtractMutableBufferNumber(tensor_wrapper->GetName());
    if (mutable_buffer_id != -1 &&
        mutable_buffer_id_to_memory_map.find(mutable_buffer_id) !=
            mutable_buffer_id_to_memory_map.end()) {
      // Fill the same memory for I/O of mutable buffer
      tensor_wrapper->FillDataBuffer(
          mutable_buffer_id_to_memory_map[mutable_buffer_id],
          false /* copy_data */);
    }
    output_tensors_[graph_name].emplace_back(std::move(tensor_wrapper));
  }
  return Error::Ok;
}

Error QnnManager::AllocateTensor(
    const std::string& graph_name,
    std::vector<std::shared_ptr<TensorWrapper>>& inputs,
    std::vector<std::shared_ptr<TensorWrapper>>& outputs) {
  input_tensors_[graph_name] = std::move(inputs);
  // TODO: suuport per-tensor dump in online prepare mode
  //       should be achievable with some pre-process
  if (!options_->is_from_context_binary()) {
    std::sort(
        input_tensors_[graph_name].begin(),
        input_tensors_[graph_name].end(),
        CompareExportedInput);
  }
  output_tensors_[graph_name] = std::move(outputs);
  return Error::Ok;
}

Error QnnManager::Execute(
    const std::string& graph_name,
    const std::vector<Qnn_Tensor_t>& input_tensor_structs,
    std::vector<Qnn_Tensor_t>& output_tensor_structs,
    executorch::runtime::EventTracer* event_tracer) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  error = backend_params_ptr_->qnn_graph_ptr_->GraphExecute(
      graph_name, input_tensor_structs, output_tensor_structs);

  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "qnn_graph_execute failed. Error %d", QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  if (IsTensorDump()) {
    // TODO: Need to handle the graph which is partitioned.
    // Maybe we could use graph name.

    // Parsing out the debug handle id
    std::regex re("_debugID_(\\d+)");
    std::smatch match;
    uint32_t debug_handle_id;
    for (std::size_t out_idx = 0; out_idx < output_tensor_structs.size();
         ++out_idx) {
      const Qnn_Tensor_t& output_tensor = output_tensor_structs[out_idx];
      std::vector<executorch::aten::SizesType> sizes(
          QNN_TENSOR_VER_PTR(output_tensor)->dimensions,
          QNN_TENSOR_VER_PTR(output_tensor)->dimensions +
              QNN_TENSOR_VER_PTR(output_tensor)->rank);

      auto dump_tensor = executorch::extension::from_blob(
          QNN_TENSOR_VER_PTR(output_tensor)->clientBuf.data,
          sizes,
          qnn_dtype_to_scalar_type_[QNN_TENSOR_VER_PTR(output_tensor)
                                        ->dataType]);

      std::string qnn_tensor_name =
          std::string(QNN_TENSOR_VER_PTR(output_tensor)->name);
      if (std::regex_search(qnn_tensor_name, match, re)) {
        debug_handle_id = static_cast<uint32_t>(std::stoul(match[1].str()));

        QNN_EXECUTORCH_LOG_INFO(
            "Found the debug_handle id %d from qnn_tensor_name: %s",
            debug_handle_id,
            QNN_TENSOR_VER_PTR(output_tensor)->name);
        executorch::runtime::event_tracer_log_output_delegate<
            executorch::aten::Tensor>(
            event_tracer,
            /*name*/
            nullptr,
            /*delegate_debug_id=*/
            static_cast<executorch::runtime::DebugHandle>(debug_handle_id),
            *dump_tensor);
      } else {
        QNN_EXECUTORCH_LOG_INFO(
            "Unable to find the debug_handle id from qnn_tensor_name: %s. Use qnn_tensor_name as key instead.",
            QNN_TENSOR_VER_PTR(output_tensor)->name);
        executorch::runtime::event_tracer_log_output_delegate<
            executorch::aten::Tensor>(
            event_tracer,
            /*name*/
            QNN_TENSOR_VER_PTR(output_tensor)->name,
            /*delegate_debug_id=*/
            static_cast<executorch::runtime::DebugHandle>(-1),
            *dump_tensor);
      }
    }
  }

  return Error::Ok;
}

Error QnnManager::ProfileExecuteData(
    const std::string& graph_name,
    executorch::runtime::EventTracer* event_tracer) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (get_option(options_->profile_level()) !=
      QnnExecuTorchProfileLevel::kProfileOff) {
    error = backend_params_ptr_->qnn_graph_ptr_->ProfileExecuteData(
        graph_name, event_tracer);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          " Failed to profile. Error %d", QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }
  return Error::Ok;
}

void QnnManager::Destroy() {
  backend_params_ptr_.reset(new BackendConfigParameters());
  backend_bundle_ptr_.reset(new QnnBackendBundle());
  qnn_dlc_manager_->Destroy();
}

void QnnManager::DestroyContext() {
  backend_params_ptr_.reset(new BackendConfigParameters());
  qnn_dlc_manager_->Destroy();
}

bool QnnManager::IsNodeSupportedByBackend(
    std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
    for (const auto& param : op_wrapper->GetParams()) {
      // unused?
      // auto* p_tensor_param = dynamic_cast<TensorParamWrapper*>(param.get());
      if (param->PopulateQnnParam() != Error::Ok) {
        QNN_EXECUTORCH_LOG_WARN(
            "Qnn Backend op validation failed "
            "with PopulateQnnParam: %d",
            QNN_GET_ERROR_CODE(error));
        return false;
      }
    }

    error = backend_bundle_ptr_->qnn_backend_ptr->BackendValidateOpConfig(
        op_wrapper->GetOpConfig());
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_WARN(
          "Qnn Backend op validation failed with error: %d",
          QNN_GET_ERROR_CODE(error));

      return false;
    }
  }
  return true;
}

Error QnnManager::GetContextBinary(
    QnnExecuTorchContextBinary& qnn_executorch_context_binary) {
  if (IsOnlinePrepare() &&
      qnn_dlc_manager_->backend_params_ptr_->qnn_context_ptr_.get() !=
          nullptr) {
    ET_CHECK_OR_RETURN_ERROR(
        qnn_dlc_manager_->backend_params_ptr_->qnn_context_ptr_
                ->GetContextBinary(qnn_executorch_context_binary) == Error::Ok,
        Internal,
        "Fail to get context binary.");
  }

  else {
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_context_ptr_->GetContextBinary(
            qnn_executorch_context_binary) == Error::Ok,
        Internal,
        "Fail to get context binary.");
  }
  return Error::Ok;
}

Error QnnManager::CompileDlc() {
  Qnn_ErrorHandle_t error;
  auto qnn_dlc_graph_info = qnn_dlc_manager_->GetQnnDlcGraphInfoPtr();
  uint32_t qnn_dlc_graph_info_num = qnn_dlc_manager_->GetQnnDlcGraphInfoNum();
  for (uint32_t i = 0; i < qnn_dlc_graph_info_num; ++i) {
    auto& graphInfo = (*qnn_dlc_graph_info)[i];
    backend_params_ptr_->qnn_graph_ptr_->SetGraphHandle(
        graphInfo.graphName, graphInfo.graph);
    error =
        backend_params_ptr_->qnn_graph_ptr_->GraphFinalize(graphInfo.graphName);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to finalize Qnn Graph with error: %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }

    std::vector<std::shared_ptr<TensorWrapper>> graph_inputs, graph_outputs,
        tensors;

    // Mapping memory address for the input and output of mutable buffer
    std::unordered_map<int, const void*> mutable_buffer_id_to_memory_map;
    for (int i = 0; i < graphInfo.numInputTensors; ++i) {
      auto tw = CreateTensorWrapper(graphInfo.inputTensors[i]);
      tw->UpdateQnnTensorMeta(graphInfo.inputTensors[i]);

      int mutable_buffer_id = ExtractMutableBufferNumber(tw->GetName());
      if (mutable_buffer_id != -1) {
        // Delegate maintains the memory for mutable buffer
        tw->AllocateDataBuffer();
        mutable_buffer_id_to_memory_map[mutable_buffer_id] =
            tw->GetStaticTensorData();
      }
      graph_inputs.push_back(tw);
    }
    for (int i = 0; i < graphInfo.numOutputTensors; ++i) {
      auto tw = CreateTensorWrapper(graphInfo.outputTensors[i]);
      tw->UpdateQnnTensorMeta(graphInfo.outputTensors[i]);
      int mutable_buffer_id = ExtractMutableBufferNumber(tw->GetName());
      if (mutable_buffer_id != -1 &&
          mutable_buffer_id_to_memory_map.find(mutable_buffer_id) !=
              mutable_buffer_id_to_memory_map.end()) {
        // Fill the same memory for I/O of mutable buffer
        tw->FillDataBuffer(
            mutable_buffer_id_to_memory_map[mutable_buffer_id],
            false /* copy_data */);
      }
      graph_outputs.push_back(tw);
    }

    ET_CHECK_OR_RETURN_ERROR(
        AllocateTensor(graphInfo.graphName, graph_inputs, graph_outputs) ==
            Error::Ok,
        Internal,
        "Fail to allocate tensor for Dlc with graph_name: %s",
        graphInfo.graphName);
  }

  return Error::Ok;
}

Error QnnManager::Compile(
    const std::string& graph_name,
    std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  QnnGraph* qnn_graph_ptr = backend_params_ptr_->qnn_graph_ptr_.get();

  if (IsOnlinePrepare() &&
      qnn_dlc_manager_->backend_params_ptr_->qnn_graph_ptr_.get() != nullptr) {
    qnn_graph_ptr = qnn_dlc_manager_->backend_params_ptr_->qnn_graph_ptr_.get();
  }
  for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
    for (const auto& tensor_wrapper : op_wrapper->GetInputTensors()) {
      ET_CHECK_OR_RETURN_ERROR(
          qnn_graph_ptr->EnsureTensorInQnnGraph(graph_name, tensor_wrapper) ==
              Error::Ok,
          Internal,
          "Tensor name %s isn't added to Qnn Graph",
          tensor_wrapper->GetName().c_str());
    }
    for (const auto& tensor_wrapper : op_wrapper->GetOutputTensors()) {
      ET_CHECK_OR_RETURN_ERROR(
          qnn_graph_ptr->EnsureTensorInQnnGraph(graph_name, tensor_wrapper) ==
              Error::Ok,
          Internal,
          "Tensor name %s isn't added to Qnn Graph",
          tensor_wrapper->GetName().c_str());
    }
    for (const auto& param : op_wrapper->GetParams()) {
      auto* p_tensor_param = dynamic_cast<TensorParamWrapper*>(param.get());
      if (p_tensor_param != nullptr) {
        ET_CHECK_OR_RETURN_ERROR(
            qnn_graph_ptr->EnsureTensorInQnnGraph(
                graph_name, p_tensor_param->GetTensorWrapper()) == Error::Ok,
            Internal,
            "Param tensor name %s isn't added to Qnn Graph",
            p_tensor_param->GetName().c_str());
      }
      ET_CHECK_OR_RETURN_ERROR(
          param->PopulateQnnParam() == Error::Ok,
          Internal,
          "Fail to configure Qnn backend");
    }

    error = qnn_graph_ptr->GraphAddNode(graph_name, op_wrapper->GetOpConfig());
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to add node to Qnn Graph with error: %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }
  error = qnn_graph_ptr->GraphFinalize(graph_name);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to finalize Qnn Graph with error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  return Error::Ok;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
void* QnnExecuTorchAllocCustomMem(size_t bytes, size_t alignment) {
  void* buffer_ptr =
      executorch::backends::qnn::SharedBuffer::GetSharedBufferManager()
          .AllocMem(bytes, alignment);
  return buffer_ptr;
}

void QnnExecuTorchFreeCustomMem(void* buffer_ptr) {
  executorch::backends::qnn::SharedBuffer::GetSharedBufferManager().FreeMem(
      buffer_ptr);
}

void QnnExecuTorchAddCustomMemTensorAddr(void* tensor_addr, void* custom_mem) {
  executorch::backends::qnn::SharedBuffer::GetSharedBufferManager()
      .AddCusomMemTensorAddr(tensor_addr, custom_mem);
}
