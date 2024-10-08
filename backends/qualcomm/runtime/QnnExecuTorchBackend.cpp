/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/aot/ir/qcir_utils.h>
#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorchBackend.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/schema_generated.h>
namespace torch {
namespace executor {
// ========== Public method implementations =========================
using namespace qnn;
using namespace qnn_delegate;
constexpr const char* QNN_COMPILE_SPEC = "qnn_compile_spec";
Result<DelegateHandle*> QnnExecuTorchBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  // covert SizedBuffer to qnn ExecuTorch option
  QnnExecuTorchContextBinary qnn_context_blob;
  const qnn_delegate::QnnExecuTorchOptions* qnn_executorch_options = nullptr;

  qnn_context_blob.buffer = const_cast<void*>(processed->data());
  qnn_context_blob.nbytes = processed->size();

  // convert CompileSpec to qnn ExecuTorch option
  for (auto& compile_spec : compile_specs) {
    if (std::strcmp(compile_spec.key, QNN_COMPILE_SPEC) == 0)
      qnn_executorch_options =
          GetQnnExecuTorchOptions(compile_spec.value.buffer);
    else
      QNN_EXECUTORCH_LOG_WARN("unknown argument: %s", compile_spec.key);
  }
  // Create QnnManager
  MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
  QnnManager* qnn_manager =
      ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(runtime_allocator, QnnManager);

  // NOTE: Since we use placement new and since this type is not trivially
  // destructible, we must call the destructor manually in destroy().
  new (qnn_manager) QnnManager(qnn_executorch_options, qnn_context_blob);

  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->Init() == Error::Ok,
      Internal,
      "Fail to initialize Qnn Manager");

  if (qnn_manager->IsOnlinePrepare()) {
    auto graph = qcir::GetGraph(qnn_context_blob.buffer);
    // qcir tensors to TensorWrapper
    std::vector<std::shared_ptr<TensorWrapper>> tensors, graph_inputs,
        graph_outputs;
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

    QnnExecuTorchContextBinary context_binary;
    ET_CHECK_OR_RETURN_ERROR(
        qnn_manager->Compile(op_wrappers, context_binary) == Error::Ok,
        Internal,
        "Fail to compile graph in online prepare stage");

    ET_CHECK_OR_RETURN_ERROR(
        qnn_manager->AllocateTensor(graph_inputs, graph_outputs) == Error::Ok,
        Internal,
        "Fail to allocate tensor in online prepare stage");
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        qnn_manager->AllocateTensor() == Error::Ok,
        Internal,
        "Fail to allocate tensor");
  }
  return qnn_manager;
}

Error QnnExecuTorchBackend::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle,
    EValue** args) const {
  QnnManager* qnn_manager = static_cast<QnnManager*>(handle);

  std::vector<std::shared_ptr<TensorWrapper>> input_tensors =
      qnn_manager->GetGraphInputs();
  std::vector<std::shared_ptr<TensorWrapper>> output_tensors =
      qnn_manager->GetGraphOutputs();
  std::vector<Qnn_Tensor_t> input_tensor_structs;
  std::vector<Qnn_Tensor_t> output_tensor_structs;

  input_tensor_structs.reserve(input_tensors.size());
  for (int i = 0; i < input_tensors.size(); ++i) {
    if (qnn_manager->RegisterMem(
            args[i]->toTensor().mutable_data_ptr(), input_tensors[i]) !=
        Error::Ok) {
      // update data ptr only should be fine
      input_tensors[i]->FillDataBuffer(
          args[i]->toTensor().const_data_ptr(), false /* copy_data */);
    }
    input_tensor_structs.push_back(input_tensors[i]->CloneTensorStruct());
  }

  int output_index = input_tensors.size();
  for (const auto& output_tensor : output_tensors) {
    // pos=0 limits the search to the prefix
    if (output_tensor->GetName().rfind("output_", 0) == 0) {
      void* mutable_data_ptr =
          args[output_index]->toTensor().mutable_data_ptr();
      if (qnn_manager->RegisterMem(mutable_data_ptr, output_tensor) !=
          Error::Ok) {
        output_tensor->FillDataBuffer(mutable_data_ptr, false /* copy_data */);
      }
      output_index++;
    }
    output_tensor_structs.push_back(output_tensor->CloneTensorStruct());
  }

  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->Execute(
          input_tensor_structs,
          output_tensor_structs,
          context.event_tracer()) == Error::Ok,
      Internal,
      "Fail to execute graph");
  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->ProfileExecuteData(context.event_tracer()) == Error::Ok,
      Internal,
      "Fail to profile graph");

  return Error::Ok;
}

void QnnExecuTorchBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr) {
    QnnManager* qnn_manager = static_cast<QnnManager*>(handle);
    qnn_manager->Destroy();
  }
}

bool QnnExecuTorchBackend::is_available() const {
  return true;
}

namespace {
auto cls = QnnExecuTorchBackend();
Backend backend{"QnnBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace
} // namespace executor
} // namespace torch
