/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2024 MediaTek Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run ExecuTorch model files that only use operators that
 * are covered by the portable kernels, with possible delegate to the
 * test_backend_compiler_lib.
 *
 * It sets all input tensor data to ones, and assumes that the outputs are
 * all fp32 tensors.
 */

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
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

static uint8_t method_allocator_pool[8 * 1024U * 1024U]; // 8 MB

// Model Path
DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format. Default to 'model.pte'");
DEFINE_string(
    input_list,
    "input_list.txt",
    "Model input list. Default to 'input_list.txt'");
DEFINE_string(
    output_folder,
    "outputs",
    "Model output folder. Default to 'outputs'");

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferCleanup;
using executorch::extension::FileDataLoader;
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
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;

using namespace std::filesystem;

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args:";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += std::string(" ") + argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  // Create output folder
  create_directories(FLAGS_output_folder);

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point to data that's already in memory, and
  // users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();
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
  Result<MethodMeta> method_meta_result = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta_result.ok(),
      "Failed to get method_meta for %s: 0x%" PRIx32,
      method_name,
      (uint32_t)method_meta_result.error());

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
  size_t num_memory_planned_buffers =
      method_meta_result->num_memory_planned_buffers();
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    // .get() will always succeed because id < num_memory_planned_buffers.
    size_t buffer_size = static_cast<size_t>(
        method_meta_result->memory_planned_buffer_size(id).get());
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
  ET_LOG(Info, "Method loaded.");

  std::ifstream input_list(FLAGS_input_list);
  ET_CHECK_MSG(
      input_list.is_open(),
      "Error: cannot open input file %s",
      FLAGS_input_list.c_str());

  auto split = [](std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
      token = s.substr(pos_start, pos_end - pos_start);
      pos_start = pos_end + delim_len;
      res.push_back(token);
    }
    res.push_back(s.substr(pos_start));
    return res;
  };

  MethodMeta method_meta = method->method_meta();
  size_t num_inputs = method_meta.num_inputs();
  std::string file_path;
  int inference_index = 0;
  while (std::getline(input_list, file_path)) {
    auto input_files = split(file_path, " ");
    if (input_files.size() == 0) {
      break;
    }
    ET_CHECK_MSG(
        input_files.size() == num_inputs,
        "Model expect %zu inputs but get %zu from input files",
        num_inputs,
        input_files.size());

    // Prepare the inputs.
    size_t num_allocated = 0;
    ET_LOG(Info, "Number of inputs: %zu", num_inputs);
    void** inputs = (void**)malloc(num_inputs * sizeof(void*));

    for (size_t i = 0; i < num_inputs; i++) {
      auto tag = method_meta.input_tag(i);
      if (tag.get() != Tag::Tensor) {
        ET_LOG(Debug, "Skipping malloc non-tensor input %zu", i);
        continue;
      }
      Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
      const auto nbytes = tensor_meta->nbytes();
      // This input is a tensor. Allocate a buffer for it.
      void* data_ptr = malloc(nbytes);

      // Read data from file
      std::ifstream fin(input_files[i], std::ios::binary);
      fin.seekg(0, fin.end);
      size_t file_size = fin.tellg();

      ET_CHECK_MSG(
          file_size == nbytes,
          "Input %zu size mismatch. file bytes: %zu, tensor bytes: %zu",
          i,
          file_size,
          nbytes);

      fin.seekg(0, fin.beg);
      fin.read(static_cast<char*>(data_ptr), file_size);
      fin.close();
      inputs[num_allocated++] = data_ptr;

      // Set backend input
      auto scalar_type = tensor_meta->scalar_type();
      auto sizes_raw = tensor_meta->sizes();
      auto dim = sizes_raw.size();
      auto dim_order_raw = tensor_meta->dim_order();
      std::vector sizes(sizes_raw.begin(), sizes_raw.end());
      std::vector dim_order(dim_order_raw.begin(), dim_order_raw.end());

      TensorImpl impl = TensorImpl(
          scalar_type, dim, sizes.data(), data_ptr, dim_order.data());

      Tensor tensor(&impl);
      Error ret = method->set_input(tensor, i);
      if (ret != Error::Ok) {
        ET_LOG(Error, "Failed to set input %zu: 0x%" PRIx32, i, (uint32_t)ret);
        // The BufferCleanup will free the inputs when it goes out of scope.
        BufferCleanup cleanup({inputs, num_allocated});
        return 1;
      }
    }
    BufferCleanup({inputs, num_allocated});
    ET_LOG(Info, "Inputs prepared.");

    // Run the model.
    auto before_exec = std::chrono::high_resolution_clock::now();
    Error status = Error::Ok;
    status = method->execute();
    auto after_exec = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
                              after_exec - before_exec)
                              .count() /
        1000.0;

    ET_LOG(Info, "Inference took %f ms", elapsed_time);
    ET_CHECK_MSG(
        status == Error::Ok,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        (uint32_t)status);
    ET_LOG(Info, "Model executed successfully.");

    // Get output data
    size_t output_size = method->outputs_size();
    ET_LOG(Info, "Number of outputs: %zu", output_size);
    std::vector<EValue> outputs(output_size);
    status = method->get_outputs(outputs.data(), output_size);
    ET_CHECK(status == Error::Ok);
    for (size_t i = 0; i < output_size; i++) {
      auto output_tensor = outputs[i].toTensor();
      auto output_file_name = FLAGS_output_folder + "/output_" +
          std::to_string(inference_index) + "_" + std::to_string(i) + ".bin";
      std::ofstream fout(output_file_name.c_str(), std::ios::binary);
      fout.write(output_tensor.const_data_ptr<char>(), output_tensor.nbytes());
      fout.close();
    }

    inference_index++;
  }

  return 0;
}
