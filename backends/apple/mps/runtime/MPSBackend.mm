//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#import <Foundation/Foundation.h>

#include "MPSCompiler.h"
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/profiler.h>
#include <cstdio>
#include <cstdlib> /* strtol */
#include <memory>
#include <string>
#include <iostream>

namespace torch {
namespace executor {

class MPSBackend final : public ::executorch::runtime::BackendInterface {
 public:
  ~MPSBackend() = default;

  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    auto executor = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        context.get_runtime_allocator(), mps::delegate::MPSExecutor);
    // NOTE: Since we use placement new and since this type is not trivially
    // destructible, we must call the destructor manually in destroy().
    new (executor) mps::delegate::MPSExecutor;
    Error err = mps::delegate::MPSCompiler::compileModel(
        processed->data(),
        processed->size(),
        executor,
        context.get_runtime_allocator(),
        compile_specs);
    ET_CHECK_OR_RETURN_ERROR(
      err == Error::Ok,
      Internal,
      "Failed to initialize the MPS delegate");

    // Free the flatbuffer.
    processed->Free();

    return executor;
  }

  // Function that actually executes the model in the backend.
  Error execute(
    ET_UNUSED BackendExecutionContext& context,
    DelegateHandle* handle,
    EValue** args) const override {
    auto executor = static_cast<mps::delegate::MPSExecutor*>(handle);
    std::vector<const Tensor*> input_pointers;
    std::vector<const Tensor*> output_pointers;

    Error err = Error::Ok;

    int i = 0;
    int total_placeholders = executor->getNumInputs() + executor->getNumOutputs();
    while ((input_pointers.size() != executor->getNumInputs()    ||
           output_pointers.size() != executor->getNumOutputs())  &&
           (i < total_placeholders)) {
      ET_CHECK_OR_RETURN_ERROR(
        args[i] != nullptr,
        Internal,
        "Nullptr tensor received during graph execution");

      if (args[i]->isTensor()) {
        if (input_pointers.size() < executor->getNumInputs()) {
          input_pointers.push_back(&args[i]->toTensor());
        } else {
          output_pointers.push_back(&args[i]->toTensor());
        }
      } else if (args[i]->isTensorList()) {
        const exec_aten::ArrayRef<exec_aten::Tensor>& tensorList = args[i]->toTensorList();
        for (auto& tensor_ : tensorList) {
          if (input_pointers.size() < executor->getNumInputs()) {
            input_pointers.push_back(&tensor_);
          } else {
            output_pointers.push_back(&tensor_);
          }
        }
      } else {
        ET_CHECK_OR_RETURN_ERROR(
          false,
          Internal,
          "Unhandled tag during execution of the graph");
      }
      i++;
    }

    err = executor->set_inputs_outputs(input_pointers, output_pointers);
    if (err != Error::Ok) {
      return err;
    }

    err = executor->forward(output_pointers);
    return err;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      auto executor = static_cast<mps::delegate::MPSExecutor*>(handle);
      // manually in init(), we must destroy it manually here.
      executor->~MPSExecutor();
    }
  }
};

namespace {
auto cls = MPSBackend();
Backend backend{"MPSBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace executor
} // namespace torch
