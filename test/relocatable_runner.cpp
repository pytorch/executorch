/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/read_file.h>
#include <executorch/util/util.h>

#include <gflags/gflags.h>

using namespace torch::executor;

/**
 * @file
 *
 * In some hardware environments, the same model may run on different cores for
 * different inference requests. The same core may also see a power-cycle (i.e.,
 * power down and then back up) in between two inference requests.
 *
 * For ExecuTorch to work efficiently in these environments, we want to
 * initialize the Method once once for the model and avoid re-initializing it
 * for every inference. This can be achieved by restricting the runtime contexts
 * (torch::executor::Program and torch::executor::Method) to live in a
 * pre-allocated, shared, and persistent memory.
 *
 * This tool demonstrates that the memory can be managed this way.
 */

static uint8_t method_allocator_pool[2 * 1024U * 1024U]; // 4 MB

#define MAX_INPUTS_PER_MODEL 16
#define MAX_OUTPUTS_PER_MODEL 8

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");

// These functions represent the work done on a worker core.
namespace worker {

Program* load_program(
    const void* file_data,
    size_t file_data_len,
    MemoryAllocator& allocator) {
  // Wrap the data in a DataLoader. The Program will take a pointer to it, so it
  // must live for at least as long as the Program instance.
  auto loader = allocator.allocateInstance<util::BufferDataLoader>();
  ET_CHECK(loader != nullptr);
  new (loader) util::BufferDataLoader(file_data, file_data_len);

  // Load the program.
  Result<Program> program_result = Program::load(loader);
  ET_CHECK(program_result.ok());

  // Move the Program into worker memory.
  auto program = allocator.allocateInstance<Program>();
  ET_CHECK(program != nullptr);
  new (program) Program(std::move(program_result.get()));

  return program;
}

MemoryManager* create_memory_manager(
    MethodMeta* method_meta,
    MemoryAllocator& worker_allocator) {
  // Create the runtime allocator.
  auto* method_allocator = worker_allocator.allocateInstance<MemoryAllocator>();
  ET_CHECK(method_allocator != nullptr);
  new (method_allocator)
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool);

  // Create the memory planned buffers.
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  Span<uint8_t>* memory_planned_buffers =
      worker_allocator.allocateList<Span<uint8_t>>(num_memory_planned_buffers);
  ET_CHECK(memory_planned_buffers != nullptr);
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    const size_t buffer_size =
        method_meta->memory_planned_buffer_size(id).get();
    ET_LOG(
        Info, "Setting up planned buffer id %zu, size %zu.", id, buffer_size);
    void* buffer = worker_allocator.allocate(buffer_size);
    ET_CHECK(buffer != nullptr);
    memory_planned_buffers[id] = {(uint8_t*)buffer, buffer_size};
    ET_LOG(
        Info,
        "Created memory_planned_buffers with size %zu and addr %p",
        buffer_size,
        buffer);
  }
  auto* planned_memory =
      worker_allocator.allocateInstance<HierarchicalAllocator>();
  ET_CHECK(planned_memory != nullptr);
  new (planned_memory) HierarchicalAllocator(
      {memory_planned_buffers, num_memory_planned_buffers});

  // The constant allocator is not currently used, but must be provided.
  auto* const_allocator = worker_allocator.allocateInstance<MemoryAllocator>();
  ET_CHECK(const_allocator != nullptr);
  new (const_allocator) MemoryAllocator(0, nullptr);

  // Assemble all of the allocators into the MemoryManager that the Method
  // will use.
  auto* memory_manager = worker_allocator.allocateInstance<MemoryManager>();
  ET_CHECK(memory_manager != nullptr);
  new (memory_manager) MemoryManager(method_allocator, planned_memory);

  return memory_manager;
}

Method* init_method(
    Program* program,
    const char* method_name,
    MemoryAllocator& worker_allocator,
    std::vector<size_t>& input_sizes,
    std::vector<size_t>& output_sizes) {
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK(method_meta.ok());

  MemoryManager* memory_manager =
      create_memory_manager(&method_meta.get(), worker_allocator);

  //
  // Create and load a method from the program, using the provided
  // allocators. The Method is what actually runs the model. It is
  // mutable, so should only be used by a single thread at at time, but it can
  // be reused.
  //

  auto* method = worker_allocator.allocateInstance<Method>();
  ET_CHECK(method != nullptr);
  auto method_res = program->load_method(method_name, memory_manager);
  ET_CHECK_MSG(
      method_res.error() == Error::Ok,
      "loading method('%s') failed with status 0x%" PRIx32,
      method_name,
      method_res.error());
  new (method) Method(std::move(method_res.get()));

  ET_LOG(Info, "Model method '%s' initialized.", method_name);

  // Gather the byte size of each input/output tensor.
  const size_t input_size = method->inputs_size();
  for (size_t i = 0; i < input_size; i++) {
    if (!method->get_input(i).isTensor()) {
      ET_LOG(Info, "input %zu is not a tensor, skipping", i);
      continue;
    }
    const auto& t = method->get_input(i).toTensor();
    input_sizes.push_back(t.nbytes());
  }

  const size_t output_size = method->outputs_size();
  for (size_t i = 0; i < output_size; i++) {
    const auto& t = method->get_output(i).toTensor();
    output_sizes.push_back(t.nbytes());
  }

  return method;
}

void inference_loop(
    Method* method,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers) {
  ET_LOG(
      Info,
      "Assigning input pointers, receiving %lu inputs",
      input_buffers.size());

  // Prepare the inputs.
  {
    size_t bufi = 0;
    for (size_t i = 0; i < method->inputs_size(); i++) {
      if (!method->get_input(i).isTensor()) {
        ET_LOG(Info, "input %zu is not a tensor, skipping", i);
        continue;
      }
      const auto& t = method->get_input(i).toTensor();
      ET_CHECK_MSG(
          bufi < input_buffers.size(), "Not enough input buffers for model");
      t.set_data(input_buffers[bufi++]);
    }
  }
  ET_LOG(Info, "Inputs prepared.");

  // Prepare the outputs.
  {
    size_t bufi = 0;
    for (size_t i = 0; i < method->outputs_size(); i++) {
      if (!method->get_output(i).isTensor()) {
        ET_LOG(Info, "output %zu is not a tensor, skipping", i);
        continue;
      }
      const auto& t = method->get_output(i).toTensor();
      ET_CHECK_MSG(
          bufi < output_buffers.size(), "Not enough output buffers for model");
      t.set_data(output_buffers[bufi++]);
    }
  }
  ET_LOG(Info, "Outputs prepared.");

  // Run the model.
  Error status = method->execute();
  ET_CHECK_MSG(
      status == Error::Ok,
      "method->execute() failed with status 0x%" PRIx32,
      status);
  ET_LOG(Info, "Model executed successfully.");
}

} // namespace worker

/*
 * This is an example of how ExecuTorch stack should run on multiple
 * processors setup where there is a control core for memory
 * management and a worker core that runs the actual inference.
 */

int main(int argc, char** argv) {
  torch::executor::runtime_init();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  /*
   * Step 1: The model gets loaded from file to memory on the control core
   */
  std::shared_ptr<char> file_data;
  size_t file_size;
  Error err = torch::executor::util::read_file_content(
      FLAGS_model_path.c_str(), &file_data, &file_size);
  ET_CHECK_MSG(err == Error::Ok, "read_file_content failed: %d", int(err));

  /*
   * Step 2: Prepare the memory space required for worker core
   */
  // The actual allocation size can be backend/model specific and smaller
  constexpr size_t kWorkerBufferSize = 1 * 1024U * 1024U; // 1 MB
  auto worker_buffer = std::make_unique<uint8_t[]>(kWorkerBufferSize);
  MemoryAllocator worker_allocator(kWorkerBufferSize, worker_buffer.get());

  /*
   * Step 3: The worker core sets up the corresponding data structures for the
   * program
   */
  Program* program =
      worker::load_program(file_data.get(), file_size, worker_allocator);
  ET_LOG(
      Info,
      "Loaded %s and constructed program at %p",
      FLAGS_model_path.c_str(),
      program);
  ET_CHECK(program != nullptr);

  /*
   * Step 4: The worker core sets up the Method. Here we let the control
   * core read out the I/O info from the Method. This can also be done on
   * the control core from the program flatbuffer, though there is no
   * direct API at the moment.
   */

  // Get the method name to execute.
  const char* method_name = nullptr;
  {
    // Use the first method in the program.
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Using method %s", method_name);

  std::vector<size_t> input_sizes;
  std::vector<size_t> output_sizes;

  Method* method = worker::init_method(
      program, method_name, worker_allocator, input_sizes, output_sizes);

  ET_LOG(
      Info,
      "Number of inputs is %lu and number of outputs is %lu",
      input_sizes.size(),
      output_sizes.size());

  /*
   * Step 5: The control core or the applicaton code prepares the I/O
   */

  // Allocate and initialize input/output tensor buffers for the inference
  std::vector<void*> input_buffers;
  for (size_t buffer_size : input_sizes) {
    void* buffer = malloc(buffer_size);
    memset(static_cast<char*>(buffer), 0, buffer_size);
    input_buffers.push_back(buffer);
  }
  ET_LOG(Info, "Allocated the inputs");

  std::vector<void*> output_buffers;
  for (size_t buffer_size : output_sizes) {
    void* buffer = malloc(buffer_size);
    memset(static_cast<char*>(buffer), 0, buffer_size);
    output_buffers.push_back(buffer);
  }
  ET_LOG(Info, "Allocated the outputs");

  /*
   * Step 6: The control core forwards the inference request and the worker
   * core runs the program.
   */

  // Run the inference on the inputs. CHECK-fails on error.
  worker::inference_loop(method, input_buffers, output_buffers);

  for (void* buffer : input_buffers) {
    free(buffer);
  }
  for (void* buffer : output_buffers) {
    free(buffer);
  }

  return 0;
}
