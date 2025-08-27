/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/exir/backend/test/demos/rpc/ExecutorBackend.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib> /* strtol */
#include <memory>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>

using ::executorch::aten::Tensor;
using ::executorch::extension::BufferDataLoader;
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::Backend;
using ::executorch::runtime::BackendExecutionContext;
using ::executorch::runtime::BackendInitContext;
using ::executorch::runtime::CompileSpec;
using ::executorch::runtime::DelegateHandle;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::FreeableBuffer;
using ::executorch::runtime::HierarchicalAllocator;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::MemoryManager;
using ::executorch::runtime::Method;
using ::executorch::runtime::MethodMeta;
using ::executorch::runtime::NamedDataMap;
using ::executorch::runtime::Program;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;
using ::executorch::runtime::Tag;
using ::executorch::runtime::internal::copy_tensor_data;

namespace example {

/**
 * ExecutorBackend is a backend to execute an executorch program via delegate.
 * In preprocess, the preprocesed bytes (delegate blob) is an executorch
 * program. In ExecutorBackend, an executor backend is constructed in init and
 * execute in execute. This backend can serve for 2 purposes
 *
 * 1. Serve as an RPC call to execute partial program on a different backend,
 * for example, host executor in cpu and client executor in dsp.
 * 2. Making incremental changes like experiment different different compiler
 * front-end before having the actual backend ready.
 */

class ExecutorBackend final : public ::executorch::runtime::BackendInterface {
 public:
  ~ExecutorBackend() = default;

  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ET_UNUSED ArrayRef<CompileSpec> compile_specs) const override {
    // `processed` contains an executorch program. Wrap it in a DataLoader that
    // will return the data directly without copying it.
    MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
    auto loader = runtime_allocator->allocateInstance<BufferDataLoader>();
    if (loader == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    new (loader) BufferDataLoader(processed->data(), processed->size());
    // Can't free `processed` because the program will point into that memory.

    // Try loading the program.
    Result<Program> program_result = Program::load(loader);
    if (!program_result.ok()) {
      return program_result.error();
    }

    // Move the Program off the stack.
    auto client_program = runtime_allocator->allocateInstance<Program>();
    if (client_program == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    new (client_program) Program(std::move(program_result.get()));

    Result<MethodMeta> method_meta = client_program->method_meta("forward");
    if (!method_meta.ok()) {
      ET_LOG(Error, "error constructing method meta");
      return method_meta.error();
    }

    // Building all different allocators for the client executor
    auto num_memory_planned_buffers = method_meta->num_memory_planned_buffers();

    Span<uint8_t>* memory_planned_buffers =
        runtime_allocator->allocateList<Span<uint8_t>>(
            num_memory_planned_buffers);
    if (memory_planned_buffers == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
      size_t buffer_size = static_cast<size_t>(
          method_meta->memory_planned_buffer_size(id).get());
      uint8_t* buffer_i = runtime_allocator->allocateList<uint8_t>(buffer_size);
      if (buffer_i == nullptr) {
        return Error::MemoryAllocationFailed;
      }

      memory_planned_buffers[id] = {buffer_i, buffer_size};
    }

    auto client_planned_memory =
        runtime_allocator->allocateInstance<HierarchicalAllocator>();
    if (client_planned_memory == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    new (client_planned_memory) HierarchicalAllocator(
        {memory_planned_buffers, num_memory_planned_buffers});

    // Allocate some memory from runtime allocator for the client executor, in
    // real case, like if it's an executor in dsp, it should allocate memory
    // dedicated to this specific hardware
    auto client_method_allocator =
        runtime_allocator->allocateInstance<MemoryAllocator>();
    if (client_method_allocator == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    const size_t kClientRuntimeMemorySize = 4 * 1024U;
    auto runtime_pool = runtime_allocator->allocate(kClientRuntimeMemorySize);
    if (runtime_pool == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    new (client_method_allocator) MemoryAllocator(
        kClientRuntimeMemorySize, static_cast<uint8_t*>(runtime_pool));

    auto client_memory_manager =
        runtime_allocator->allocateInstance<MemoryManager>();
    if (client_memory_manager == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    new (client_memory_manager)
        MemoryManager(client_method_allocator, client_planned_memory);

    const NamedDataMap* named_data_map = context.get_named_data_map();
    // Construct the client Method
    Result<Method> method_res = client_program->load_method(
        "forward",
        client_memory_manager,
        /*event_tracer=*/nullptr,
        named_data_map);
    if (!method_res.ok()) {
      ET_LOG(
          Error,
          "Failed to load client method: 0x%x",
          (unsigned int)method_res.error());
      return method_res.error();
    }

    auto client_method = runtime_allocator->allocateInstance<Method>();
    if (client_method == nullptr) {
      return Error::MemoryAllocationFailed;
    }

    new (client_method) Method(std::move(method_res.get()));

    // Return the client method so it will be passed to `execute()` as
    // `handle`.
    return client_method;
  }

  Error execute(
      ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    Method* client_method = static_cast<Method*>(handle);
    auto num_inputs = client_method->inputs_size();
    Error status = Error::Ok;

    // Receive client executor input
    for (size_t input_idx = 0; input_idx < num_inputs; input_idx++) {
      status = client_method->set_input(*args[input_idx], input_idx);
    }
    // Execute client executor
    status = client_method->execute();

    auto output_sizes = client_method->outputs_size();
    // Send the client executor output, we'd need to copy the data instead of
    // assigning the Evalue pointer
    for (int i = 0; i < output_sizes; i++) {
      EValue output = client_method->get_output(i);
      if (output.tag == Tag::Tensor) {
        Tensor t_src = output.toTensor();
        Tensor t_dst = args[num_inputs + i]->toTensor();
        status = copy_tensor_data(t_dst, t_src);
      }
    }

    return status;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      Method* client_executor = static_cast<Method*>(handle);
      client_executor->~Method();
    }
  }
};

Error register_executor_backend() {
  static auto cls = ExecutorBackend();
  static Backend backend{"ExecutorBackend", &cls};
  static auto success_with_compiler = register_backend(backend);
  return success_with_compiler;
}

} // namespace example
