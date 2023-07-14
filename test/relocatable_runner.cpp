#include <memory>
#include <vector>

#include <executorch/core/Constants.h>
#include <executorch/runtime/executor/executor.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/embedded_data_loader.h>
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
 * initialize the execution plan once once for the model and avoid
 * re-initializing it for every inference. This can be achieved by restricting
 * the runtime contexts (torch::executor::Program and torch::executor::Executor)
 * to live in a pre-allocated, shared, and persistent memory.
 *
 * This tool demonstrates that the memory can be managed this way.
 */

static constexpr size_t kRuntimeMemorySize = 2 * kMB;
static uint8_t runtime_pool[kRuntimeMemorySize];

// This is to emulate the local memory that a particular instance of hardware
// own and shared across different model instances
static constexpr size_t kNonConstantMemorySize = 10 * kMB;
static uint8_t shared_local_non_constant_pool[kNonConstantMemorySize];

#define MAX_INPUTS_PER_MODEL 16
#define MAX_OUTPUTS_PER_MODEL 8

DEFINE_string(model_path, "model.ff", "Model serialized in flatbuffer format.");

// These functions represent the work done on a worker core.
namespace worker {

Program* load_program(
    const void* file_data,
    size_t file_data_len,
    MemoryAllocator& allocator) {
  // Wrap the data in a DataLoader. The Program will take a pointer to it, so it
  // must live for at least as long as the Program instance.
  auto loader = allocator.allocateInstance<util::EmbeddedDataLoader>();
  ET_CHECK(loader != nullptr);
  new (loader) util::EmbeddedDataLoader(file_data, file_data_len);

  // Load the program.
  Result<Program> program_result = Program::Load(loader);
  ET_CHECK(program_result.ok());

  // Move the Program into worker memory.
  auto program = allocator.allocateInstance<Program>();
  ET_CHECK(program != nullptr);
  new (program) Program(std::move(program_result.get()));

  return program;
}

MemoryManager* create_memory_manager(
    Program* program,
    const char* method_name,
    MemoryAllocator& worker_allocator) {
  // Create the runtime allocator.
  auto* runtime_allocator =
      worker_allocator.allocateInstance<MemoryAllocator>();
  ET_CHECK(runtime_allocator != nullptr);
  new (runtime_allocator) MemoryAllocator(sizeof(runtime_pool), runtime_pool);

  // Create the non-const allocator and the buffers it points to.
  size_t num_non_const_buffers =
      program->num_non_const_buffers(method_name).get();
  MemoryAllocator* non_const_allocators =
      worker_allocator.allocateList<MemoryAllocator>(num_non_const_buffers - 1);
  for (size_t id = 1; id < num_non_const_buffers; ++id) {
    const size_t buffer_size =
        program->get_non_const_buffer_size(id, method_name).get();
    ET_LOG(
        Info, "Setting up non-const buffer id %zu, size %zu.", id, buffer_size);
    void* buffer = worker_allocator.allocate(buffer_size);
    ET_CHECK(buffer != nullptr);
    new (&non_const_allocators[id - 1])
        MemoryAllocator(buffer_size, (uint8_t*)buffer);
    ET_LOG(
        Info,
        "Created non_const_allocators with size %zu and addr %p",
        buffer_size,
        buffer);
  }
  auto* non_const_allocator =
      worker_allocator.allocateInstance<HierarchicalAllocator>();
  ET_CHECK(non_const_allocator != nullptr);
  new (non_const_allocator)
      HierarchicalAllocator(num_non_const_buffers - 1, non_const_allocators);

  // The constant allocator is not currently used, but must be provided.
  auto* const_allocator = worker_allocator.allocateInstance<MemoryAllocator>();
  ET_CHECK(const_allocator != nullptr);
  new (const_allocator) MemoryAllocator(0, nullptr);

  // The temp allocator is not currently used, but must be provided.
  auto* temp_allocator = worker_allocator.allocateInstance<MemoryAllocator>();
  ET_CHECK(temp_allocator != nullptr);
  new (temp_allocator) MemoryAllocator(0, nullptr);

  // Assemble all of the allocators into the MemoryManager that the Executor
  // will use.
  auto* memory_manager = worker_allocator.allocateInstance<MemoryManager>();
  ET_CHECK(memory_manager != nullptr);
  new (memory_manager) MemoryManager(
      const_allocator, non_const_allocator, runtime_allocator, temp_allocator);

  return memory_manager;
}

ExecutionPlan* init_method(
    Program* program,
    const char* method_name,
    MemoryAllocator& worker_allocator,
    std::vector<size_t>& input_sizes,
    std::vector<size_t>& output_sizes) {
  MemoryManager* memory_manager =
      create_memory_manager(program, method_name, worker_allocator);

  //
  // Create an Executor and ExecutionPlan from the program, using the provided
  // allocators. The ExecutionPlan is what actually runs the model. It is
  // mutable, so should only be used by a single thread at at time, but it can
  // be reused.
  //

  auto* executor = worker_allocator.allocateInstance<Executor>();
  ET_CHECK(executor != nullptr);
  new (executor) Executor(program, memory_manager);

  Error status = executor->init_execution_plan(method_name);
  ET_CHECK_MSG(
      status == Error::Ok,
      "init_execution_plan('%s') failed with status 0x%" PRIx32,
      method_name,
      status);
  ET_LOG(Info, "Model method '%s' initialized.", method_name);
  auto& plan = executor->execution_plan();

  // Gather the byte size of each input/output tensor.
  const size_t input_size = plan.inputs_size();
  for (size_t i = 0; i < input_size; i++) {
    if (!plan.get_input(i).isTensor()) {
      ET_LOG(Info, "input %zu is not a tensor, skipping", i);
      continue;
    }
    const auto& t = plan.get_input(i).toTensor();
    input_sizes.push_back(t.nbytes());
  }

  const size_t output_size = plan.outputs_size();
  for (size_t i = 0; i < output_size; i++) {
    const auto& t = plan.get_output(i).toTensor();
    output_sizes.push_back(t.nbytes());
  }

  return &plan;
}

void inference_loop(
    ExecutionPlan* plan,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers) {
  ET_LOG(
      Info,
      "Assigning input pointers, receiving %lu inputs",
      input_buffers.size());

  // Prepare the inputs.
  {
    size_t bufi = 0;
    for (size_t i = 0; i < plan->inputs_size(); i++) {
      if (!plan->get_input(i).isTensor()) {
        ET_LOG(Info, "input %zu is not a tensor, skipping", i);
        continue;
      }
      const auto& t = plan->get_input(i).toTensor();
      ET_CHECK_MSG(
          bufi < input_buffers.size(), "Not enough input buffers for model");
      t.set_data(input_buffers[bufi++]);
    }
  }
  ET_LOG(Info, "Inputs prepared.");

  // Prepare the outputs.
  {
    size_t bufi = 0;
    for (size_t i = 0; i < plan->outputs_size(); i++) {
      if (!plan->get_output(i).isTensor()) {
        ET_LOG(Info, "output %zu is not a tensor, skipping", i);
        continue;
      }
      const auto& t = plan->get_output(i).toTensor();
      ET_CHECK_MSG(
          bufi < output_buffers.size(), "Not enough output buffers for model");
      t.set_data(output_buffers[bufi++]);
    }
  }
  ET_LOG(Info, "Outputs prepared.");

  // Run the model.
  Error status = plan->execute();
  ET_CHECK_MSG(
      status == Error::Ok,
      "plan->execute() failed with status 0x%" PRIx32,
      status);
  ET_LOG(Info, "Model executed successfully.");
}

} // namespace worker

/*
 * This is an example of how ExecuTorch stack should run on multiple
 * processors setup like Turing where there is a control core for memory
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
  constexpr size_t kWorkerBufferSize = 1 * kMB;
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
   * Step 4: The worker core sets up the Executor and initalizes the execution
   * plan. Here we let the control core read out the I/O info from the
   * execution plan. This can also be done on the control core from the
   * program flatbuffer, though there is no direct API at the moment.
   */

  // Get the method name to execute.
  const char* method_name = nullptr;
  {
    // Use the first method in the program.
    const size_t plan_index = 0;
    const auto method_name_result = program->get_method_name(plan_index);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Using method %s", method_name);

  std::vector<size_t> input_sizes;
  std::vector<size_t> output_sizes;

  ExecutionPlan* plan = worker::init_method(
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
  worker::inference_loop(plan, input_buffers, output_buffers);

  for (void* buffer : input_buffers) {
    free(buffer);
  }
  for (void* buffer : output_buffers) {
    free(buffer);
  }

  return 0;
}
