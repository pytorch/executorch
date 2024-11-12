#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

DEFINE_string(
    model_path,
    "/home/icx-6338/ynimmaga/delegate.pte", //"model.pte",
    "Model serialized in flatbuffer format.");
DEFINE_int32(iteration, 1, "Iterations of inference.");

using executorch::extension::FileDataLoader;
using executorch::extension::prepare_input_tensors;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args:";
    for (int i = 1; i < argc; i++) {
      msg += " " + std::string(argv[i]);
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  const char* model_path = FLAGS_model_path.c_str();
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      static_cast<uint32_t>(loader.error()));

  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Using method %s", method_name);

  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      method_name,
      static_cast<uint32_t>(method_meta.error()));

  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  MemoryManager memory_manager(&method_allocator, &planned_memory);

  Result<Method> method = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      static_cast<uint32_t>(method.error()));
  ET_LOG(Info, "Method loaded.");

  auto inputs = prepare_input_tensors(*method);
  ET_CHECK_MSG(
      inputs.ok(),
      "Could not prepare inputs: 0x%" PRIx32,
      static_cast<uint32_t>(inputs.error()));
  ET_LOG(Info, "Inputs prepared.");

  auto before_exec = std::chrono::high_resolution_clock::now();
  Error status = Error::Ok;
  for (int i = 0; i < FLAGS_iteration; ++i) {
    status = method->execute();
  }
  auto after_exec = std::chrono::high_resolution_clock::now();
  double elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            after_exec - before_exec)
                            .count() / 1000.0;

  ET_LOG(
      Info,
      "%d inference took %f ms, avg %f ms",
      FLAGS_iteration,
      elapsed_time,
      elapsed_time / static_cast<float>(FLAGS_iteration));
  ET_CHECK_MSG(
      status == Error::Ok,
      "Execution of method %s failed with status 0x%" PRIx32,
      method_name,
      static_cast<uint32_t>(status));
  ET_LOG(Info, "Model executed successfully.");

  std::vector<EValue> outputs(method->outputs_size());
  ET_LOG(Info, "%zu outputs: ", outputs.size());
  status = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);
  //std::cout << executorch::extension::evalue_edge_items(100);
  //for (int i = 0; i < outputs.size(); ++i) {
  //  std::cout << "Output " << i << ": " << outputs[i] << std::endl;
  //}

  return 0;
}
