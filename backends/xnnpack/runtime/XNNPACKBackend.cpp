/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNCompiler.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/profiler.h>
#include <memory>

#pragma clang diagnostic ignored "-Wglobal-constructors"

namespace torch {
namespace executor {

class XnnpackBackend final : public PyTorchBackendInterface {
 public:
  ~XnnpackBackend() = default;

  bool is_available() const override {
    return xnn_status_success == xnn_initialize(/*allocator=*/nullptr);
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    auto executor = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        context.get_runtime_allocator(), xnnpack::delegate::XNNExecutor);

    // Executor has been allocated but not constructed, ensure that runtime_ is
    // nullptr by constructing it in place here. NOTE: Since we use placement
    // new and since this type is not trivially destructible, we must call the
    // destructor manually in destroy().
    new (executor) xnnpack::delegate::XNNExecutor;

    Error err = xnnpack::delegate::XNNCompiler::compileModel(
        processed->data(),
        processed->size(),
        executor,
        context.get_runtime_allocator());
    if (err != Error::Ok) {
      ET_LOG(Error, "XNNCompiler::compleModel failed: 0x%x", (unsigned int)err);
    }

    // Free the flatbuffer
    processed->Free();

    return executor;
  }

  Error execute(
      __ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      EValue** args) const override {
    auto executor = static_cast<xnnpack::delegate::XNNExecutor*>(handle);

    // TODO merge these two in a single struct?
    std::vector<Tensor*> input_pointers;
    std::vector<Tensor*> output_pointers;
    std::vector<xnnpack::delegate::XNNShape> input_shapes;
    std::vector<xnnpack::delegate::XNNShape> output_shapes;

    ET_CHECK_OR_RETURN_ERROR(
        executor->get_args_size() ==
            executor->getNumInputs() + executor->getNumOutputs(),
        Internal,
        "External id and expected delegate args mismatch");

    // Intialize XNNShapes for both inputs and outputs.
    // That will allow us to gradually build up shape inference support for
    // xnnpack ops without breaking existing models when we always try to resize
    // delegate output tensor(s).
    input_shapes.resize(executor->getNumInputs());
    output_shapes.resize(executor->getNumOutputs());

    for (int i = 0; i < executor->get_args_size(); i++) {
      int index = executor->get_arg_index(i);

      if (!args[index]->isTensor()) {
        ET_LOG(Error, "Expected argument to be a tensor");
      }

      Tensor* tensor = &args[index]->toTensor();
      size_t num_dims = tensor->dim();
      struct xnnpack::delegate::XNNShape* shape = nullptr;

      if (i < executor->getNumInputs()) {
        input_pointers.push_back(tensor);
        shape = &input_shapes[i];
      } else {
        output_pointers.push_back(tensor);
        shape = &output_shapes[i - executor->getNumInputs()];
      }

      shape->num_dims = num_dims;
      for (int d = 0; d < num_dims; ++d) {
        shape->dim[d] = tensor->size(d);
      }
    }

    Error err = executor->set_inputs(
        input_pointers, output_pointers, input_shapes, output_shapes);

    if (err != Error::Ok) {
      return err;
    }

    err = executor->forward();

#ifdef ENABLE_XNNPACK_PROFILING
    executor->log_op_timings(); // Log the op execution time.
#endif

    // Resize output tensors - should be a no-op for static models
    for (int i = 0; i < executor->getNumOutputs(); i++) {
      err = executor->resizeOutput(output_pointers[i], &output_shapes[i]);
      if (err != Error::Ok) {
        return err;
      }
    }

    for (int i = executor->getNumInputs();
         i < executor->getNumInputs() + executor->getNumOutputs();
         i++) {
      if (args[i]->isTensor()) {
        exec_aten::Tensor output_tensor = args[i]->toTensor();
        if (output_tensor.scalar_type() == ScalarType::Long) {
          // Output datatype is int64. However, XNNPACK doesn't support
          // int64. This means that the data was put into this tensor
          // by XNNPACK as int32 and needs to be copied to int64 form
          int64_t* data_64 = output_tensor.mutable_data_ptr<int64_t>();
          const int32_t* data_32 = output_tensor.const_data_ptr<int32_t>();
          for (int j = output_tensor.numel() - 1; j >= 0; j--) {
            data_64[j] = data_32[j];
          }
        }
      }
    }

    return err;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      auto executor = static_cast<xnnpack::delegate::XNNExecutor*>(handle);
#ifdef ENABLE_XNNPACK_PROFILING
      executor->print_avg_op_timings();
#endif
      // XNNExecutor is not trivially destructible. Since this was constructed
      // manually in init(), we must destroy it manually here.
      executor->~XNNExecutor();
    }
  }
};

namespace {
auto cls = XnnpackBackend();
Backend backend{"XnnpackBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace executor
} // namespace torch
