/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

/**
 * @file
 *
 * This tool can run ExecuTorch model files with Enn runtime.
 * It assumes all inputs and output are fp32, please give a list for input
 * files. And Enn backends is going to inference, and output results.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <gflags/gflags.h>

#include <fstream>
#include <memory>
#include <sstream>

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

DEFINE_string(model, "model.pte", "Model serialized in flatbuffer format.");
DEFINE_string(
    input,
    "",
    "Input file path, support multiple inputs: input_1 input_2 ...");

DEFINE_string(output_path, "", "Output Execution results to target directory.");

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

std::vector<std::string> split(std::string str, char delimiter = ' ') {
  std::vector<std::string> result;
  std::stringstream ss(str);
  std::string temp;
  while (std::getline(ss, temp, delimiter)) {
    if (!temp.empty()) {
      result.push_back(temp);
    }
  }
  return result;
}

class DataReader {
 public:
  typedef std::vector<uint8_t> data_t;

  DataReader(size_t size) : data_set_(size) {}

  void read(const std::string file_path) {
    ET_CHECK(index_ < data_set_.size());
    data_t& data = data_set_[index_];
    std::ifstream input_file(file_path.c_str(), std::ios::binary);
    ET_CHECK(input_file.is_open());
    input_file.seekg(0, std::ios::end);
    data.resize(input_file.tellg());
    input_file.seekg(0);
    input_file.read(reinterpret_cast<char*>(data.data()), data.size());
    input_file.close();
    ++index_;
  }

  void* get(int32_t index) {
    ET_CHECK(index < data_set_.size());
    return data_set_[index].data();
  }

  size_t nbytes(int32_t index) {
    ET_CHECK(index < data_set_.size());
    return data_set_[index].size();
  }

  ~DataReader() = default;

 private:
  std::vector<data_t> data_set_;
  int32_t index_ = 0;
};

void saveOutput(const exec_aten::Tensor& tensor, int32_t output_index) {
  if (FLAGS_output_path.empty()) {
    return;
  }
  auto output_file_name =
      FLAGS_output_path + "/output_" + std::to_string(output_index) + ".bin";
  std::ofstream fout(output_file_name.c_str(), std::ios::binary);
  ET_CHECK_MSG(
      fout.is_open(),
      "Directory or have no visit permission: %s",
      FLAGS_output_path.c_str());
  fout.write(tensor.const_data_ptr<char>(), tensor.nbytes());
  fout.close();
}

int main(int argc, char** argv) {
  runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args:";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += std::string(" ") + argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point to data that's already in memory, and
  // users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model.c_str();
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      (uint32_t)loader.error());

  // Parse the program file. This is immutable, and can also be reused between
  // multiple execution invocations across multiple threads.
  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  // Use the first method in the program.
  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Using method %s", method_name);

  // MethodMeta describes the memory requirements of the method.
  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      method_name,
      (uint32_t)method_meta.error());

  //
  // The runtime does not use malloc/new; it allocates all memory using the
  // MemoryManger provided by the client. Clients are responsible for allocating
  // the memory ahead of time, or providing MemoryAllocator subclasses that can
  // do it dynamically.
  //

  // The method allocator is used to allocate all dynamic C++ metadata/objects
  // used to represent the loaded method. This allocator is only used during
  // loading a method of the program, which will return an error if there was
  // not enough memory.
  //
  // The amount of memory required depends on the loaded method and the runtime
  // code itself. The amount of memory here is usually determined by running the
  // method and seeing how much memory is actually used, though it's possible to
  // subclass MemoryAllocator so that it calls malloc() under the hood (see
  // MallocMemoryAllocator).
  //
  // In this example we use a statically allocated memory pool.
  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};

  // The memory-planned buffers will back the mutable tensors used by the
  // method. The sizes of these buffers were determined ahead of time during the
  // memory-planning pasees.
  //
  // Each buffer typically corresponds to a different hardware memory bank. Most
  // mobile environments will only have a single buffer. Some embedded
  // environments may have more than one for, e.g., slow/large DRAM and
  // fast/small SRAM, or for memory associated with particular cores.
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the memory
  std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    // .get() will always succeed because id < num_memory_planned_buffers.
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu.", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  // Assemble all of the allocators into the MemoryManager that the Executor
  // will use.
  MemoryManager memory_manager(&method_allocator, &planned_memory);

  //
  // Load the method from the program, using the provided allocators. Running
  // the method can mutate the memory-planned buffers, so the method should only
  // be used by a single thread at at time, but it can be reused.
  //

  Result<Method> method = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      (uint32_t)method.error());

  auto input_files = split(FLAGS_input);
  ET_CHECK_MSG(
      input_files.size() == method->inputs_size(),
      "Please check the number of given input binary files");
  DataReader input_data_reader(input_files.size());
  for (const auto& input_file : input_files) {
    input_data_reader.read(input_file);
  }

  for (int input_index = 0; input_index < method->inputs_size();
       ++input_index) {
    MethodMeta method_meta = method->method_meta();
    Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(input_index);
    ET_CHECK_MSG(
        input_data_reader.nbytes(input_index) == tensor_meta->nbytes(),
        "Given inputs size is invalid");
    TensorImpl impl = TensorImpl(
        tensor_meta->scalar_type(),
        tensor_meta->sizes().size(),
        const_cast<TensorImpl::SizesType*>(tensor_meta->sizes().data()),
        input_data_reader.get(input_index),
        const_cast<TensorImpl::DimOrderType*>(tensor_meta->dim_order().data()));
    Error ret = method->set_input(Tensor(&impl), input_index);
    ET_CHECK_MSG(ret == Error::Ok, "Failed to set input tensor: %d", ret);
  }
  // Allocate input tensors and set all of their elements to 1. The `inputs`
  // variable owns the allocated memory and must live past the last call to
  // `execute()`.
  // auto inputs = util::prepare_input_tensors(*method);

  // Run the model.
  ET_LOG(Info, "Start inference.");
  auto start = std::chrono::high_resolution_clock::now();
  Error status = method->execute();
  auto end = std::chrono::high_resolution_clock::now();
  double elapse =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1000.0;
  ET_CHECK_MSG(
      status == Error::Ok,
      "Execution of method %s failed with status 0x%" PRIx32,
      method_name,
      static_cast<int32_t>(status));
  ET_LOG(Info, "End with elapsed time(ms): %f", elapse);

  // Get the outputs.
  std::vector<EValue> outputs(method->outputs_size());
  status = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);

  for (size_t output_index = 0; output_index < method->outputs_size();
       ++output_index) {
    auto output_tensor = outputs[output_index].toTensor();
    // Save the results to given directory in order.
    saveOutput(output_tensor, output_index);
  }

  return 0;
}
