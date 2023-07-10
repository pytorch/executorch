#include <executorch/backends/test/demos/rpc/ExecutorBackend.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib> /* strtol */
#include <memory>

#include <executorch/core/Constants.h>
#include <executorch/core/Error.h>
#include <executorch/core/values/Evalue.h>
#include <executorch/executor/Executor.h>
#include <executorch/runtime/backend/backend_registry.h>
#include <executorch/util/embedded_data_loader.h>
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
      FreeableBuffer* processed,
      __ET_UNUSED ArrayRef<CompileSpec> compile_specs,
      MemoryAllocator* runtime_allocator) const override {
    // `processed` contains an executorch program. Wrap it in a DataLoader that
    // will return the data directly without copying it.
    auto loader = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        runtime_allocator, util::EmbeddedDataLoader);
    new (loader) util::EmbeddedDataLoader(processed->data(), processed->size());
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

    // Building all different allocators for the client executor
    auto client_const_allocator = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        runtime_allocator, MemoryAllocator);
    new (client_const_allocator) MemoryAllocator(0, nullptr);

    size_t num_non_const_buffers = client_program->num_non_const_buffers() - 1;

    uint8_t** non_const_buffers = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
        runtime_allocator, uint8_t*, num_non_const_buffers);
    MemoryAllocator* non_const_allocators = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
        runtime_allocator, MemoryAllocator, num_non_const_buffers);

    for (size_t id = 1; id < client_program->num_non_const_buffers(); ++id) {
      const size_t buffer_size = client_program->get_non_const_buffer_size(id);
      uint8_t* buffer_i = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
          runtime_allocator, uint8_t, buffer_size);
      non_const_buffers[id - 1] = buffer_i;
      new (&non_const_allocators[id - 1])
          MemoryAllocator(static_cast<uint32_t>(buffer_size), buffer_i);
    }

    auto client_non_const_allocator = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        runtime_allocator, HierarchicalAllocator);
    new (client_non_const_allocator) HierarchicalAllocator(
        client_program->num_non_const_buffers() - 1, non_const_allocators);

    // Allocate some memory from runtime allocator for the client executor, in
    // real case, like if it's an executor in dsp, it should allocate memory
    // dedicated to this specific hardware
    auto client_runtime_allocator = ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(
        runtime_allocator, MemoryAllocator);
    const size_t kClientRuntimeMemorySize = 4 * kKB;
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

    // Construct the client executor
    auto client_executor =
        ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(runtime_allocator, Executor);
    new (client_executor) Executor(client_program, client_memory_manager);

    // Initialize the client executor
    Error err = client_executor->init_execution_plan("forward");
    if (err != Error::Ok) {
      ET_LOG(Error, "Failed to init client executor: 0x%x", (unsigned int)err);
      return err;
    }

    // Return the client executor so it will be passed to `execute()` as
    // `handle`.
    return client_executor;
  }

  Error execute(DelegateHandle* handle, EValue** args) const override {
    Executor* client_executor = static_cast<Executor*>(handle);
    auto& plan = client_executor->execution_plan();
    auto plan_inputs_size = plan.inputs_size();
    Error status = Error::Ok;

    // Receive client executor input
    for (size_t input_idx = 0; input_idx < plan_inputs_size; input_idx++) {
      status = plan.set_input(*args[input_idx], input_idx);
    }
    // Execute client executor
    status = plan.execute();

    // Send the client executor output
    status = plan.get_outputs(args[plan_inputs_size], plan.outputs_size());

    return status;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      Executor* client_executor = static_cast<Executor*>(handle);
      client_executor->~Executor();
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
