#include <cstring>
#include <memory>
#include <iostream>

#include <openvino/openvino.hpp>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

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

typedef struct {
    std::shared_ptr<ov::CompiledModel> compiled_model;
    std::shared_ptr<ov::InferRequest> infer_request;
} ExecutionHandle;

class OpenvinoBackend final : public ::executorch::runtime::BackendInterface {
 public:
  OpenvinoBackend() {std::cout << "In OV Backend constructor" << std::endl;}

  ~OpenvinoBackend() = default;

  virtual bool is_available() const override {
    // Check if OpenVINO runtime is available
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    ET_LOG(Info, "OpenvinoBackend::init %p", processed->data());

    ov::Core core;

    const char* data_ptr = static_cast<const char*>(processed->data());
    size_t data_size = processed->size();

    // Copy data to a string or vector
    std::string data_string(data_ptr, data_size);

    // Wrap the data in a stream
    std::istringstream compiled_stream(data_string);

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

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* input_handle,
      EValue** args) const override {
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

  void destroy(DelegateHandle* handle) const override {
    return;
  }

 private:
  ov::element::Type convert_to_openvino_type(ScalarType scalar_type) const {
    // Convert ExecuteTorch scalar types to OpenVINO element types
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
};

} // namespace openvino
} // namespace backends
} // namespace executorch

namespace {
auto backend = executorch::backends::openvino::OpenvinoBackend();
executorch::runtime::Backend backend_id{"OpenvinoBackend", &backend};
static auto registered = executorch::runtime::register_backend(backend_id);
} // namespace


