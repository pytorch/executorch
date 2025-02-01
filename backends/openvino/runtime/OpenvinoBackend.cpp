/*
 * Copyright (c) Intel Corporation
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include <cstring>
#include <memory>
#include <iostream>

#include <openvino/openvino.hpp>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include "OpenvinoBackend.hpp"

using namespace std;
using executorch::aten::ScalarType;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;

namespace executorch {
namespace backends {
namespace openvino {

OpenvinoBackend::OpenvinoBackend() {
    if (!is_available()) {
        //ET_LOG(Error, "OpenVINO runtime is not available. Initialization failed.");
        throw std::runtime_error("OpenVINO runtime not available");
    }

    //ET_LOG(Info, "OpenVINO runtime successfully verified and initialized.");
}

bool OpenvinoBackend::is_available() const {
    try {
        // Create an OpenVINO Core object to verify runtime availability
        ov::Core core;

        // Check if at least one device is available
        auto devices = core.get_available_devices();
        if (!devices.empty()) {
            return true;  // OpenVINO is available
        }
    } catch (const std::exception& e) {
        // Log the exception if OpenVINO runtime is not available
        ET_LOG(Error, "OpenVINO is not available: %s", e.what());
    } catch (...) {
        // Handle any unexpected errors
        ET_LOG(Error, "OpenVINO availability check failed due to an unknown error.");
    }

    return false;  // OpenVINO is not available
}

Result<DelegateHandle*> OpenvinoBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {

    ET_LOG(Info, "OpenvinoBackend::init %p", processed->data());

    ov::Core core;
    const char* data_ptr = static_cast<const char*>(processed->data());
    size_t data_size = processed->size();

    // Copy data to a string or vector
    std::string data_string(data_ptr, data_size);

    // Wrap the data in a stream
    std::istringstream compiled_stream(data_string);

    // Import the model
    auto compiled_model = core.import_model(compiled_stream, "CPU");

    // Allocate an infer request
    std::shared_ptr<ov::InferRequest> infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());

    // Allocate execution handle
    MemoryAllocator* allocator = context.get_runtime_allocator();
    ExecutionHandle* handle = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(allocator, ExecutionHandle);
    handle->compiled_model = std::make_shared<ov::CompiledModel>(compiled_model);
    handle->infer_request = infer_request;

    return handle;
}

Error OpenvinoBackend::execute(
    BackendExecutionContext& context,
    DelegateHandle* input_handle,
    EValue** args) const {

    ExecutionHandle* execution_handle = (ExecutionHandle*)input_handle;

    auto infer_request = execution_handle->infer_request;

    size_t num_inputs = infer_request->get_compiled_model().inputs().size();
    size_t num_outputs = infer_request->get_compiled_model().outputs().size();

    // Set inputs
    for (size_t i = 0; i < num_inputs; i++) {
      auto input_tensor = args[i]->toTensor();
      ov::Shape input_shape(input_tensor.sizes().begin(), input_tensor.sizes().end());

      // Convert input tensor to OpenVINO tensor
      ov::element::Type ov_type = convert_to_openvino_type(input_tensor.scalar_type());
      ov::Tensor ov_input_tensor(ov_type, input_shape, input_tensor.mutable_data_ptr());

      infer_request->set_input_tensor(i, ov_input_tensor);
    }

    // Set outputs
    for (size_t i = 0; i < num_outputs; i++) {
      auto output_tensor = args[num_inputs+i]->toTensor();
      ov::Shape output_shape(output_tensor.sizes().begin(), output_tensor.sizes().end());

      // Convert input tensor to OpenVINO tensor
      ov::element::Type ov_type = convert_to_openvino_type(output_tensor.scalar_type());
      ov::Tensor ov_output_tensor(ov_type, output_shape, output_tensor.mutable_data_ptr());

      infer_request->set_output_tensor(i, ov_output_tensor);
    }

    // Execute the inference
    infer_request->infer();

    return Error::Ok;
}

void OpenvinoBackend::destroy(DelegateHandle* handle) const {
    if (!handle) {
        ET_LOG(Info, "Attempted to destroy a null handle.");
        return;
    }

    // Cast the handle to the appropriate type
    ExecutionHandle* execution_handle = static_cast<ExecutionHandle*>(handle);

    // Clean up resources
    if (execution_handle->infer_request) {
        execution_handle->infer_request.reset();  // Release the infer request
        ET_LOG(Info, "Infer request successfully destroyed.");
    }

    if (execution_handle->compiled_model) {
        execution_handle->compiled_model.reset();  // Release the compiled model
        ET_LOG(Info, "Compiled model successfully destroyed.");
    }

    ET_LOG(Info, "Delegate handle destroyed successfully.");
}

ov::element::Type OpenvinoBackend::convert_to_openvino_type(ScalarType scalar_type) const {
    switch (scalar_type) {
      case ScalarType::Float:
        return ov::element::f32;
      case ScalarType::Int:
        return ov::element::i32;
      case ScalarType::Char:
        return ov::element::i8;
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


