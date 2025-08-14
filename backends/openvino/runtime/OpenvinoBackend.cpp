/*  Copyright (c) Intel Corporation
 *
 *  Licensed under the BSD License (the "License"); you may not use this file
 *  except in compliance with the License. See the license file found in the
 *  LICENSE file in the root directory of this source tree.
 */

#include <cstring>
#include <iostream>
#include <memory>

#include <openvino/openvino.hpp>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include "OpenvinoBackend.h"

namespace executorch {
namespace backends {
namespace openvino {

OpenvinoBackend::OpenvinoBackend() {}

bool OpenvinoBackend::is_available() const {
  try {
    // Create an OpenVINO Core object to verify runtime availability
    ov::Core core;

    // Check if at least one device is available
    auto devices = core.get_available_devices();
    if (!devices.empty()) {
      return true; // OpenVINO is available
    }
  } catch (const std::exception& e) {
    // Log the exception if OpenVINO runtime is not available
    ET_LOG(Error, "OpenVINO is not available: %s", e.what());
  } catch (...) {
    // Handle any unexpected errors
    ET_LOG(
        Error, "OpenVINO availability check failed due to an unknown error.");
  }

  return false; // OpenVINO is not available
}

exr::Result<exr::DelegateHandle*> OpenvinoBackend::init(
    exr::BackendInitContext& context,
    exr::FreeableBuffer* processed,
    exr::ArrayRef<exr::CompileSpec> compile_specs) const {
  ET_LOG(Info, "OpenvinoBackend::init %p", processed->data());

  ov::Core core;
  const char* data_ptr = static_cast<const char*>(processed->data());
  size_t data_size = processed->size();

  // Copy data to a string or vector
  std::string data_string(data_ptr, data_size);

  // Wrap the data in a stream
  std::istringstream compiled_stream(data_string);

  auto device = "CPU";
  // Get the device value, if provided in compile sepcs
  for (auto& compile_spec : compile_specs) {
    if (std::strcmp(compile_spec.key, "device") == 0)
      device = static_cast<char*>(compile_spec.value.buffer);
  }

  // Import the model
  auto compiled_model = core.import_model(compiled_stream, device);

  // The processed data can be freed since the model is compiled
  processed->Free();

  // Allocate an infer request
  std::shared_ptr<ov::InferRequest> infer_request =
      std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());

  // Allocate execution handle
  exr::MemoryAllocator* allocator = context.get_runtime_allocator();
  ExecutionHandle* handle = allocator->allocateInstance<ExecutionHandle>();
  new (handle) ExecutionHandle;
  handle->compiled_model = std::make_shared<ov::CompiledModel>(compiled_model);
  handle->infer_request = infer_request;

  return handle;
}

exr::Error OpenvinoBackend::execute(
    exr::BackendExecutionContext& context,
    exr::DelegateHandle* input_handle,
    exr::Span<exr::EValue*> args) const {
  ExecutionHandle* execution_handle = (ExecutionHandle*)input_handle;

  auto infer_request = execution_handle->infer_request;

  size_t num_inputs = infer_request->get_compiled_model().inputs().size();
  size_t num_outputs = infer_request->get_compiled_model().outputs().size();

  // Set inputs
  for (size_t i = 0; i < num_inputs; i++) {
    auto input_tensor = args[i]->toTensor();
    ov::Shape input_shape(
        input_tensor.sizes().begin(), input_tensor.sizes().end());

    // Convert input tensor to OpenVINO tensor
    ov::element::Type ov_type =
        convert_to_openvino_type(input_tensor.scalar_type());
    ov::Tensor ov_input_tensor(
        ov_type, input_shape, input_tensor.mutable_data_ptr());

    infer_request->set_input_tensor(i, ov_input_tensor);
  }

  // Set outputs
  for (size_t i = 0; i < num_outputs; i++) {
    auto output_tensor = args[num_inputs + i]->toTensor();
    ov::Shape output_shape(
        output_tensor.sizes().begin(), output_tensor.sizes().end());

    // Convert input tensor to OpenVINO tensor
    ov::element::Type ov_type =
        convert_to_openvino_type(output_tensor.scalar_type());
    ov::Tensor ov_output_tensor(
        ov_type, output_shape, output_tensor.mutable_data_ptr());

    infer_request->set_output_tensor(i, ov_output_tensor);
  }

  // Execute the inference
  infer_request->infer();

  return exr::Error::Ok;
}

void OpenvinoBackend::destroy(exr::DelegateHandle* handle) const {
  if (!handle) {
    ET_LOG(Info, "Attempted to destroy a null handle.");
    return;
  }

  // Cast the handle to the appropriate type
  ExecutionHandle* execution_handle = static_cast<ExecutionHandle*>(handle);

  // Clean up resources
  if (execution_handle->infer_request) {
    execution_handle->infer_request.reset(); // Release the infer request
    ET_LOG(Info, "Infer request successfully destroyed.");
  }

  if (execution_handle->compiled_model) {
    execution_handle->compiled_model.reset(); // Release the compiled model
    ET_LOG(Info, "Compiled model successfully destroyed.");
  }

  ET_LOG(Info, "Delegate handle destroyed successfully.");
}

ov::element::Type OpenvinoBackend::convert_to_openvino_type(
    exa::ScalarType scalar_type) const {
  switch (scalar_type) {
    case exa::ScalarType::Float:
      return ov::element::f32;
    case exa::ScalarType::Int:
      return ov::element::i32;
    case exa::ScalarType::Char:
      return ov::element::i8;
    case exa::ScalarType::Long:
      return ov::element::i64;
    case exa::ScalarType::Bool:
      return ov::element::boolean;
    default:
      throw std::runtime_error("Unsupported scalar type");
  }
}

} // namespace openvino
} // namespace backends
} // namespace executorch

namespace {
auto backend = executorch::backends::openvino::OpenvinoBackend();
executorch::runtime::Backend backend_id{"OpenvinoBackend", &backend};
static auto registered = executorch::runtime::register_backend(backend_id);
} // namespace
