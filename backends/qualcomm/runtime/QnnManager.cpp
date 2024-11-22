/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/aot/ir/qcir_utils.h>
#include <executorch/backends/qualcomm/qc_binary_info_generated.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/SharedBuffer.h>
#include <executorch/backends/qualcomm/runtime/Utils.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnBackendCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnImplementation.h>
#include <executorch/extension/tensor/tensor.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

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

QnnManager::~QnnManager() {
  backend_params_ptr_.reset(new BackendConfigParameters());
  logger_.reset();
  qnn_loaded_backend_.TerminateAllBackends();
}

QnnManager::QnnManager(
    const QnnExecuTorchOptions* options,
    const QnnExecuTorchContextBinary& qnn_executorch_context_binary)
    : qnn_context_blob_(qnn_executorch_context_binary),
      qnn_loaded_backend_(""),
      // options' life cycle is decided by compiler specs which is
      // kept by executorch runtime framework
      // please pay attention to any potential seg fault
      options_(options) {
  QnnExecuTorchBackendType backend_type =
      options->backend_options()->backend_type();
  std::string library_path = options->library_path()->str();

  if (options->log_level() >= QnnExecuTorchLogLevel::kLogLevelInfo) {
    QNN_EXECUTORCH_LOG_INFO(
        "soc_model in soc_info: %s",
        EnumNameQcomChipset(options_->soc_info()->soc_model()));
    QNN_EXECUTORCH_LOG_INFO(
        "backend_type: %s", EnumNameQnnExecuTorchBackendType(backend_type));
    QNN_EXECUTORCH_LOG_INFO("graph_name: %s", options_->graph_name()->c_str());
    QNN_EXECUTORCH_LOG_INFO("library_path: %s", library_path.c_str());
    QNN_EXECUTORCH_LOG_INFO("dump intermediate outputs: %s", IsTensorDump());
    QNN_EXECUTORCH_LOG_INFO(
        "log_level: %s", EnumNameQnnExecuTorchLogLevel(options_->log_level()));
    QNN_EXECUTORCH_LOG_INFO(
        "profile_level: %s",
        EnumNameQnnExecuTorchProfileLevel(options_->profile_level()));
    QNN_EXECUTORCH_LOG_INFO(
        "the size of qnn context binary: %d",
        qnn_executorch_context_binary.nbytes);
    QNN_EXECUTORCH_LOG_INFO(
        "Is on-device graph construction: %d", options->online_prepare());
    QNN_EXECUTORCH_LOG_INFO(
        "Enable shared buffer: %d", options->shared_buffer());
  }

  if (library_path.empty()) {
    switch (backend_type) {
      case QnnExecuTorchBackendType::kHtpBackend:
        library_path = htp_library_name_;
        break;
      case QnnExecuTorchBackendType::kDspBackend:
        library_path = dsp_library_name_;
        break;
      case QnnExecuTorchBackendType::kGpuBackend:
        library_path = gpu_library_name_;
        break;
      default:
        QNN_EXECUTORCH_LOG_ERROR("Unknown backend type: %d", backend_type);
        break;
    }
  }
  qnn_loaded_backend_ = QnnImplementation(library_path);
  backend_params_ptr_ = std::make_unique<BackendConfigParameters>();
}

Error QnnManager::LoadQnnLibrary() {
  Error ret = qnn_loaded_backend_.Load(nullptr);
  return ret;
}

Error QnnManager::PreRegisterMem() {
  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();
  for (const auto info : shared_buffer_manager.GetCustomMemTensorInfoSet()) {
    void* unaligned_custom_mem_base =
        shared_buffer_manager.GetUnAlignedAddr(info.custom_mem);

    size_t tensor_offset = (static_cast<char*>(info.custom_mem) -
                            static_cast<char*>(unaligned_custom_mem_base)) +
        info.pos;
    size_t total_custom_mem_size =
        shared_buffer_manager.GetAllocatedSize(info.custom_mem);

    int32_t mem_fd = shared_buffer_manager.MemToFd(unaligned_custom_mem_base);
    if (mem_fd == -1) {
      QNN_EXECUTORCH_LOG_WARN(
          "PreRegisterMem failed to get file descriptor.",
          "custom_mem: %p",
          "tensor_addr: %p",
          "pos: %uz",
          "tensor_bytes: %uz",
          "shape: %p",
          "rank: %zu",
          "qnn_dtype: %X",
          info.custom_mem,
          info.tensor_addr,
          info.pos,
          info.tensor_bytes,
          info.shape,
          info.rank,
          info.dtype);
      return Error::Internal;
    }

    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_mem_manager_ptr_->PreRegisterCustomMemHandle(
            mem_fd,
            unaligned_custom_mem_base,
            total_custom_mem_size,
            tensor_offset,
            info) == Error::Ok,
        Internal,
        "Fail to register to shared memory.");
  }
  return Error::Ok;
}

Error QnnManager::RegisterMem(
    void* data_ptr,
    const std::shared_ptr<TensorWrapper>& tensor_wrapper) {
  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();
  // Not enable shared buffer
  if (!options_->shared_buffer())
    return Error::Internal;

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
    if (options_->log_level() >= QnnExecuTorchLogLevel::kLogLevelInfo)
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
    if (options_->log_level() >= QnnExecuTorchLogLevel::kLogLevelInfo)
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
  if (pre_registered_handle != nullptr) {
    if (options_->log_level() >= QnnExecuTorchLogLevel::kLogLevelInfo) {
      QNN_EXECUTORCH_LOG_INFO(
          "Tensor name %s found a pre-registered memHandle.",
          tensor_wrapper->GetName().c_str());
    }
    return backend_params_ptr_->qnn_mem_manager_ptr_->SetMemHandle(
        tensor_wrapper, data_ptr, pre_registered_handle);
  }

  SharedBuffer& shared_buffer_manager = SharedBuffer::GetSharedBufferManager();
  void* unaligned_custom_mem_base =
      shared_buffer_manager.GetUnAlignedAddr(custom_mem_base);

  size_t tensor_offset = static_cast<char*>(custom_mem_base) -
      static_cast<char*>(unaligned_custom_mem_base) + info.pos;
  size_t total_custom_mem_size =
      shared_buffer_manager.GetAllocatedSize(custom_mem_base);

  int32_t mem_fd = shared_buffer_manager.MemToFd(unaligned_custom_mem_base);
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
          unaligned_custom_mem_base,
          total_custom_mem_size,
          tensor_offset) == Error::Ok,
      Internal,
      "Fail to register to shared memory.");

  return Error::Ok;
}

Error QnnManager::Init() {
  ET_CHECK_OR_RETURN_ERROR(
      LoadQnnLibrary() == Error::Ok, Internal, "Fail to load Qnn library");
  logger_ = std::make_unique<QnnLogger>(
      qnn_loaded_backend_, LoggingCallback, options_->log_level());
  if (backend_params_ptr_->backend_init_state_ ==
      BackendInitializeState::UNINITIALIZED) {
    QNN_EXECUTORCH_LOG_INFO(
        "Initialize Qnn backend "
        "parameters for Qnn executorch backend type %d",
        options_->backend_options()->backend_type());
    backend_params_ptr_ = QnnBackendFactory().Create(
        qnn_loaded_backend_, logger_.get(), qnn_context_blob_, options_);
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_ != nullptr, Internal, "Failed to load Qnn backend.")
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_backend_cache_ptr_->Configure() == Error::Ok,
        Internal,
        "Fail to configure Qnn backend cache");
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_backend_ptr_->Configure() == Error::Ok,
        Internal,
        "Fail to configure Qnn backend");
    ET_CHECK_OR_RETURN_ERROR(
        backend_params_ptr_->qnn_device_ptr_->Configure() == Error::Ok,
        Internal,
        "Fail to configure Qnn device");
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

#if defined(__aarch64__)
  ET_CHECK_OR_RETURN_ERROR(
      PreRegisterMem() == Error::Ok,
      Internal,
      "Fail to pre register custom memory handle");
#endif
  return Error::Ok;
}

Error QnnManager::AllocateTensor(const std::string& graph_name) {
  std::vector<Qnn_Tensor_t> input_tensors =
      backend_params_ptr_->qnn_context_ptr_->GetGraphInputs(graph_name);
  std::vector<Qnn_Tensor_t> output_tensors =
      backend_params_ptr_->qnn_context_ptr_->GetGraphOutputs(graph_name);

  for (auto& tensor : input_tensors) {
    std::shared_ptr<TensorWrapper> tensor_wrapper = CreateTensorWrapper(tensor);
    tensor_wrapper->UpdateQnnTensorMeta(tensor);
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
    for (std::size_t out_idx = 0; out_idx < output_tensor_structs.size();
         ++out_idx) {
      const Qnn_Tensor_t& output_tensor = output_tensor_structs[out_idx];
      std::vector<executorch::aten::SizesType> sizes(
          QNN_VER_PTR(output_tensor)->dimensions,
          QNN_VER_PTR(output_tensor)->dimensions +
              QNN_VER_PTR(output_tensor)->rank);

      auto dump_tensor = executorch::extension::from_blob(
          QNN_VER_PTR(output_tensor)->clientBuf.data,
          sizes,
          qnn_dtype_to_scalar_type_[QNN_VER_PTR(output_tensor)->dataType]);

      executorch::runtime::event_tracer_log_output_delegate<
          executorch::aten::Tensor>(
          event_tracer,
          QNN_VER_PTR(output_tensor)->name,
          /*delegate_debug_id=*/
          static_cast<executorch::runtime::DebugHandle>(-1),
          *dump_tensor);
    }
  }

  return Error::Ok;
}

Error QnnManager::ProfileExecuteData(
    const std::string& graph_name,
    executorch::runtime::EventTracer* event_tracer) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (options_->profile_level() != QnnExecuTorchProfileLevel::kProfileOff) {
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
  QNN_EXECUTORCH_LOG_INFO("Destroy Qnn backend parameters");
  backend_params_ptr_.reset(new BackendConfigParameters());
  logger_.reset();

  qnn_loaded_backend_.TerminateAllBackends();
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

    error = backend_params_ptr_->qnn_backend_ptr_->BackendValidateOpConfig(
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
  ET_CHECK_OR_RETURN_ERROR(
      backend_params_ptr_->qnn_context_ptr_->GetContextBinary(
          qnn_executorch_context_binary) == Error::Ok,
      Internal,
      "Fail to get context binary.");

  return Error::Ok;
}

Error QnnManager::CompileQcir() {
  flatbuffers::Verifier verifier_binary_info(
      static_cast<const uint8_t* const>(qnn_context_blob_.buffer),
      qnn_context_blob_.nbytes);
  if (!qnn_delegate::VerifyBinaryInfoBuffer(verifier_binary_info)) {
    QNN_EXECUTORCH_LOG_ERROR("Fail to verify binary info");
    return Error::Internal;
  }

  auto binary_info = qnn_delegate::GetBinaryInfo(qnn_context_blob_.buffer);
  flatbuffers::Verifier verifier_qcir(
      binary_info->data()->data(), binary_info->data()->size());
  if (!qcir::VerifyContextBuffer(verifier_qcir)) {
    QNN_EXECUTORCH_LOG_ERROR("Fail to verify qcir format");
    return Error::Internal;
  }

  auto context = qcir::GetContext(binary_info->data()->data());
  for (const auto& graph : *context->graphs()) {
    // qcir tensors to TensorWrapper
    std::vector<std::shared_ptr<TensorWrapper>> graph_inputs, graph_outputs,
        tensors;
    for (const auto& tensor : *graph->tensors()) {
      tensors.emplace_back(CreateTensorWrapper(ToTensor(tensor)));
      if (tensor->type() == qcir::TensorType::WRITE) {
        graph_inputs.push_back(tensors.back());
      } else if (tensor->type() == qcir::TensorType::READ) {
        graph_outputs.push_back(tensors.back());
      }
    }
    std::vector<std::shared_ptr<OpWrapper>> op_wrappers;
    // qcir graph node to OpWrapper
    for (const auto& node : *graph->nodes()) {
      std::shared_ptr<OpWrapper> op = std::make_shared<OpWrapper>(
          node->name()->str(),
          node->package_name()->str(),
          node->type_name()->str());

      // qcir input tensors to OpWrapper input tensors
      std::vector<std::shared_ptr<TensorWrapper>> inputs;
      for (uint32_t index : *node->inputs()) {
        inputs.push_back(tensors[index]);
      }
      op->AddInputTensors(inputs);

      // qcir output tensors to OpWrapper output tensors
      std::vector<std::shared_ptr<TensorWrapper>> outputs;
      for (uint32_t index : *node->outputs()) {
        outputs.push_back(tensors[index]);
      }
      op->AddOutputTensors(outputs);

      // qcir operator param to OpWrapper param
      for (uint32_t index : *node->params()) {
        const auto& tensor = graph->tensors()->Get(index);
        std::string name = tensor->name()->str();
        Qnn_DataType_t dtype = ToDataType(tensor->dtype());
        if (tensor->shape()->size() != 0) {
          // add tensor param
          op->AddTensorParam(
              name,
              dtype,
              tensor->shape()->size(),
              tensor->shape()->data(),
              tensor->data()->data());
        } else {
          // add scalar param
          switch (dtype) {
            case Qnn_DataType_t::QNN_DATATYPE_INT_32:
              op->AddScalarParam(
                  name,
                  dtype,
                  *reinterpret_cast<const int32_t*>(tensor->data()->Data()));
              break;
            case Qnn_DataType_t::QNN_DATATYPE_INT_16:
              op->AddScalarParam(
                  name,
                  dtype,
                  *reinterpret_cast<const int16_t*>(tensor->data()->Data()));
              break;
            case Qnn_DataType_t::QNN_DATATYPE_INT_8:
              op->AddScalarParam(
                  name, dtype, static_cast<int8_t>(*tensor->data()->Data()));
              break;
            case Qnn_DataType_t::QNN_DATATYPE_UINT_32:
              op->AddScalarParam(
                  name,
                  dtype,
                  *reinterpret_cast<const uint32_t*>(tensor->data()->Data()));
              break;
            case Qnn_DataType_t::QNN_DATATYPE_UINT_16:
              op->AddScalarParam(
                  name,
                  dtype,
                  *reinterpret_cast<const uint16_t*>(tensor->data()->Data()));
              break;
            case Qnn_DataType_t::QNN_DATATYPE_UINT_8:
              op->AddScalarParam(name, dtype, *tensor->data()->Data());
              break;
            case Qnn_DataType_t::QNN_DATATYPE_FLOAT_32:
            case Qnn_DataType_t::QNN_DATATYPE_FLOAT_16:
              op->AddScalarParam(
                  name,
                  dtype,
                  *reinterpret_cast<const float*>(tensor->data()->Data()));
              break;
            case Qnn_DataType_t::QNN_DATATYPE_BOOL_8:
              op->AddScalarParam(name, dtype, *tensor->data()->Data());
              break;
            default:
              QNN_EXECUTORCH_LOG_ERROR(
                  "Invalid scalar type: %s", tensor->name()->c_str());
              break;
          }
        }
      }
      op_wrappers.push_back(std::move(op));
    }

    ET_CHECK_OR_RETURN_ERROR(
        Compile(graph->name()->str(), op_wrappers) == Error::Ok,
        Internal,
        "Fail to compile graph from qcir with graph_name: %s",
        graph->name()->str().c_str());

    ET_CHECK_OR_RETURN_ERROR(
        AllocateTensor(graph->name()->str(), graph_inputs, graph_outputs) ==
            Error::Ok,
        Internal,
        "Fail to allocate tensor for qcir with graph_name: %s",
        graph->name()->str().c_str());
  }

  return Error::Ok;
}

Error QnnManager::Compile(
    const std::string& graph_name,
    std::vector<std::shared_ptr<OpWrapper>>& op_wrappers) {
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  for (std::shared_ptr<OpWrapper>& op_wrapper : op_wrappers) {
    for (const auto& tensor_wrapper : op_wrapper->GetInputTensors()) {
      ET_CHECK_OR_RETURN_ERROR(
          backend_params_ptr_->qnn_graph_ptr_->EnsureTensorInQnnGraph(
              graph_name, tensor_wrapper) == Error::Ok,
          Internal,
          "Tensor name %s isn't added to Qnn Graph",
          tensor_wrapper->GetName().c_str());
    }

    for (const auto& tensor_wrapper : op_wrapper->GetOutputTensors()) {
      ET_CHECK_OR_RETURN_ERROR(
          backend_params_ptr_->qnn_graph_ptr_->EnsureTensorInQnnGraph(
              graph_name, tensor_wrapper) == Error::Ok,
          Internal,
          "Tensor name %s isn't added to Qnn Graph",
          tensor_wrapper->GetName().c_str());
    }

    for (const auto& param : op_wrapper->GetParams()) {
      auto* p_tensor_param = dynamic_cast<TensorParamWrapper*>(param.get());
      if (p_tensor_param != nullptr) {
        ET_CHECK_OR_RETURN_ERROR(
            backend_params_ptr_->qnn_graph_ptr_->EnsureTensorInQnnGraph(
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

    error = backend_params_ptr_->qnn_graph_ptr_->GraphAddNode(
        graph_name, op_wrapper->GetOpConfig());
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to add node to Qnn Graph with error: %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  }

  error = backend_params_ptr_->qnn_graph_ptr_->GraphFinalize(graph_name);
  if (error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "Failed to finalize Qnn Graph with error: %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }

  return Error::Ok;
}

std::string QnnManager::GetBinarySignature() {
  flatbuffers::Verifier verifier(
      static_cast<const uint8_t* const>(qnn_context_blob_.buffer),
      qnn_context_blob_.nbytes);
  return VerifyBinaryInfoBuffer(verifier)
      ? GetBinaryInfo(qnn_context_blob_.buffer)->signature()->str()
      : "";
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

void QnnExecuTorchAddCustomMemTensorInfo(const CustomMemTensorInfo& info) {
  executorch::backends::qnn::SharedBuffer::GetSharedBufferManager()
      .AddCusomMemTensorInfo(info);
}
