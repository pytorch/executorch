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
#include <executorch/runtime/backend/backend_registry.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/util/util.h>

namespace torch {
namespace executor {
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

class ExecutorBackend final : public PyTorchBackendInterface {
 public:
  ~ExecutorBackend() = default;

  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      __ET_UNUSED ArrayRef<CompileSpec> compile_specs) const override {
    // `processed` contains an executorch program. Wrap it in a DataLoader that
    // will return the data directly without copying it.
    MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
    auto loader = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        runtime_allocator, util::BufferDataLoader);
    new (loader) util::BufferDataLoader(processed->data(), processed->size());
    // Can't free `processed` because the program will point into that memory.

    // Try loading the program.
    Result<Program> program_result = Program::Load(loader);
    if (!program_result.ok()) {
      return program_result.error();
    }

    // Move the Program off the stack.
    auto client_program =
        ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(runtime_allocator, Program);
    new (client_program) Program(std::move(program_result.get()));

    Result<MethodMeta> method_meta = client_program->method_meta("forward");
    if (!method_meta.ok()) {
      ET_LOG(Error, "error constructing method meta");
      return method_meta.error();
    }

    // Building all different allocators for the client executor
    auto client_const_allocator = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        runtime_allocator, MemoryAllocator);
    new (client_const_allocator) MemoryAllocator(0, nullptr);

    auto num_buffers = method_meta->num_non_const_buffers();
    size_t num_non_const_buffers = num_buffers - 1;

    uint8_t** non_const_buffers = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
        runtime_allocator, uint8_t*, num_non_const_buffers);
    MemoryAllocator* non_const_allocators = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
        runtime_allocator, MemoryAllocator, num_non_const_buffers);

    for (size_t id = 1; id < num_buffers; ++id) {
      auto buffer_size = method_meta->non_const_buffer_size(id);
      uint8_t* buffer_i = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
          runtime_allocator, uint8_t, buffer_size.get());
      non_const_buffers[id - 1] = buffer_i;
      new (&non_const_allocators[id - 1])
          MemoryAllocator(static_cast<uint32_t>(buffer_size.get()), buffer_i);
    }

    auto client_non_const_allocator = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        runtime_allocator, HierarchicalAllocator);
    new (client_non_const_allocator)
        HierarchicalAllocator(num_non_const_buffers, non_const_allocators);

    // Allocate some memory from runtime allocator for the client executor, in
    // real case, like if it's an executor in dsp, it should allocate memory
    // dedicated to this specific hardware
    auto client_runtime_allocator = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        runtime_allocator, MemoryAllocator);
    const size_t kClientRuntimeMemorySize = 4 * 1024U;
    auto runtime_pool = ET_ALLOCATE_OR_RETURN_ERROR(
        runtime_allocator, kClientRuntimeMemorySize);
    new (client_runtime_allocator) MemoryAllocator(
        kClientRuntimeMemorySize, static_cast<uint8_t*>(runtime_pool));

    auto client_temp_allocator = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        runtime_allocator, MemoryAllocator);
    new (client_temp_allocator) MemoryAllocator(0, nullptr);

    auto client_memory_manager =
        ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(runtime_allocator, MemoryManager);
    new (client_memory_manager) MemoryManager(
        client_const_allocator,
        client_non_const_allocator,
        client_runtime_allocator,
        client_temp_allocator);

    // Construct the client Method
    Result<Method> method_res =
        client_program->load_method("forward", client_memory_manager);
    if (!method_res.ok()) {
      ET_LOG(
          Error,
          "Failed to load client method: 0x%x",
          (unsigned int)method_res.error());
      return method_res.error();
    }

    auto client_method =
        ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(runtime_allocator, Method);
    new (client_method) Method(std::move(method_res.get()));

    // Return the client method so it will be passed to `execute()` as
    // `handle`.
    return client_method;
  }

  Error execute(
      __ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      EValue** args) const override {
    Method* client_method = static_cast<Method*>(handle);
    auto num_inputs = client_method->inputs_size();
    Error status = Error::Ok;

    // Receive client executor input
    for (size_t input_idx = 0; input_idx < num_inputs; input_idx++) {
      status = client_method->set_input(*args[input_idx], input_idx);
    }
    // Execute client executor
    status = client_method->execute();

    // Send the client executor output
    status = client_method->get_outputs(
        args[num_inputs], client_method->outputs_size());

    return status;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      Method* client_executor = static_cast<Method*>(handle);
      client_executor->~Method();
    }
  }
};

Error registerExecutorBackend() {
  static auto cls = ExecutorBackend();
  static Backend backend{"ExecutorBackend", &cls};
  static auto success_with_compiler = register_backend(backend);
  return success_with_compiler;
}

} // namespace executor
} // namespace torch
