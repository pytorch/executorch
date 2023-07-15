#include <executorch/runtime/executor/executor.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/file_data_loader.h>
#include <executorch/util/util.h>
#include <stdio.h>

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

static constexpr size_t kRuntimeMemorySize = 1024;
static constexpr size_t kMemoryAmount = 512;

static uint8_t runtime_pool[kRuntimeMemorySize];
static uint8_t activation_pool[kMemoryAmount];

int main(int argc, char** argv) {
  runtime_init();

  ET_CHECK_MSG(argc == 2, "Expected model file argument.");

  MemoryAllocator const_allocator{MemoryAllocator(0, nullptr)};
  const_allocator.enable_profiling("const allocator");

  MemoryAllocator runtime_allocator{
      MemoryAllocator(kRuntimeMemorySize, runtime_pool)};
  runtime_allocator.enable_profiling("runtime allocator");

  MemoryAllocator temp_allocator{MemoryAllocator(0, nullptr)};
  temp_allocator.enable_profiling("temp allocator");

  MemoryAllocator non_const_allocators[1]{
      MemoryAllocator(kMemoryAmount, activation_pool)};
  non_const_allocators[0].enable_profiling("non_const_allocators");

  HierarchicalAllocator non_const_allocator{
      HierarchicalAllocator(1, non_const_allocators)};

  MemoryManager memory_manager{MemoryManager(
      &const_allocator,
      &non_const_allocator,
      &runtime_allocator,
      &temp_allocator)};

  Result<FileDataLoader> loader = FileDataLoader::From(argv[1]);
  ET_CHECK_MSG(
      loader.ok(), "FileDataLoader::From() failed: 0x%" PRIx32, loader.error());

  uint32_t prof_tok = EXECUTORCH_BEGIN_PROF("de-serialize model");
  const auto program = Program::Load(&loader.get());
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK_MSG(
      program.ok(), "Program::Load() failed: 0x%" PRIx32, program.error());

  ET_LOG(Info, "Model file %s is loaded.", argv[1]);

  prof_tok = EXECUTORCH_BEGIN_PROF("load model");
  Executor executor(&program.get(), &memory_manager);
  Error status = executor.init_execution_plan();
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK(status == Error::Ok);

  ET_LOG(Info, "Model initialized.");

  // Prepare for inputs
  // It assumes the input is one tensor.
  auto& plan = executor.execution_plan();

  auto inputs = torch::executor::util::PrepareInputTensors(plan);

  ET_LOG(Info, "Inputs prepared.");

  prof_tok = EXECUTORCH_BEGIN_PROF("run model");
  status = plan.execute();
  EXECUTORCH_END_PROF(prof_tok);
  ET_CHECK(status == Error::Ok);
  ET_LOG(Info, "Model executed successfully.");

  // print output
  auto output_list =
      runtime_allocator.allocateList<EValue>(plan.outputs_size());

  status = plan.get_outputs(output_list, plan.outputs_size());
  ET_CHECK(status == Error::Ok);

  // It assumes the outputs are all tensors.
  for (size_t i = 0; i < plan.outputs_size(); i++) {
    auto output_tensor = output_list[i].toTensor();
    auto data_output = output_tensor.const_data_ptr<float>();
    for (size_t j = 0; j < output_list[i].toTensor().numel(); ++j) {
      ET_LOG(Info, "%f", data_output[j]);
    }
  }
  torch::executor::util::FreeInputs(inputs);
  prof_result_t prof_result;
  EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);
  if (prof_result.num_bytes != 0) {
    FILE* ptr = fopen("prof_result.bin", "w+");
    fwrite(prof_result.prof_data, 1, prof_result.num_bytes, ptr);
    fclose(ptr);
  }

  return 0;
}
