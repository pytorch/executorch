/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run ExecuTorch model files with Qualcomm AI Engine Direct
 * and the portable kernels.
 *
 * User could specify arguments like desired input data, iterfations, etc.
 * Currently we assume that the outputs are all fp32 tensors.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/util.h>
#include <gflags/gflags.h>

#include <fstream>
#include <memory>

static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    prof_result_path,
    "prof_result.bin",
    "Executorch profiler output path.");
DEFINE_string(
    output_folder_path,
    "outputs",
    "Executorch inference data output path.");
DEFINE_string(input_list_path, "input_list.txt", "Model input list path.");
DEFINE_int32(iteration, 1, "Iterations of inference.");
DEFINE_int32(warm_up, 0, "Pre-run before inference.");

using namespace torch::executor;
using torch::executor::util::FileDataLoader;

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
  const char* model_path = FLAGS_model_path.c_str();
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(), "FileDataLoader::from() failed: 0x%" PRIx32, loader.error());

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
      "Failed to get method_meta for %s: 0x%x",
      method_name,
      (unsigned int)method_meta.error());

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
  method_allocator.enable_profiling("method allocator");

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
      method.error());
  ET_LOG(Info, "Method loaded.");

  // Prepare the inputs.
  // Use ones-initialized inputs.
  auto inputs = util::PrepareInputTensors(*method);
  ET_LOG(Info, "Inputs prepared.");

  std::ifstream input_list(FLAGS_input_list_path);
  if (input_list.is_open()) {
    size_t num_inputs = method->inputs_size();
    ET_LOG(Info, "Number of inputs: %zu", num_inputs);

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

    std::string file_path;
    int inference_index = 0;
    double elapsed_time = 0;
    while (std::getline(input_list, file_path)) {
      auto input_files = split(file_path, " ");
      if (input_files.size() == 0) {
        break;
      }
      ET_CHECK_MSG(
          input_files.size() == num_inputs,
          "Number of inputs (%zu) mismatch with input files (%zu)",
          num_inputs,
          input_files.size());

      for (int input_index = 0; input_index < num_inputs; ++input_index) {
        exec_aten::Tensor& t = method->mutable_input(input_index).toTensor();
        std::vector<char> input_data(t.nbytes());
        std::ifstream fin(input_files[input_index], std::ios::binary);
        fin.seekg(0, fin.end);
        size_t file_size = fin.tellg();

        ET_CHECK_MSG(
            file_size == t.nbytes(),
            "Input(%d) size mismatch. file bytes: %zu, tensor bytes: %zu",
            input_index,
            file_size,
            t.nbytes());

        fin.seekg(0, fin.beg);
        fin.read(input_data.data(), file_size);
        fin.close();

        std::vector<TensorImpl::SizesType> sizes(t.dim());
        for (int i = 0; i < sizes.size(); ++i) {
          sizes[i] = t.sizes().data()[i];
        }

        auto t_impl = TensorImpl(
            t.scalar_type(), t.dim(), sizes.data(), input_data.data());
        Error ret = method->set_input(EValue(Tensor(&t_impl)), input_index);
        ET_CHECK_MSG(ret == Error::Ok, "Failed to set input tensor: %d", ret);
      }

      Error status = Error::Ok;
      // Warm up
      ET_LOG(Info, "Perform %d inference for warming up", FLAGS_warm_up);
      for (int i = 0; i < FLAGS_warm_up; ++i) {
        status = method->execute();
      }

      // Inference with designated iterations
      ET_LOG(Info, "Start inference (%d)", inference_index);
      auto before_exec = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < FLAGS_iteration; ++i) {
        status = method->execute();
      }
      auto after_exec = std::chrono::high_resolution_clock::now();
      double interval_infs =
          std::chrono::duration_cast<std::chrono::microseconds>(
              after_exec - before_exec)
              .count() /
          1000.0;
      elapsed_time += interval_infs;

      ET_LOG(
          Info,
          "%d inference took %f ms, avg %f ms",
          FLAGS_iteration,
          interval_infs,
          interval_infs / (float)FLAGS_iteration);
      ET_CHECK_MSG(
          status == Error::Ok,
          "Execution of method %s failed with status 0x%" PRIx32,
          method_name,
          status);

      std::vector<EValue> outputs(method->outputs_size());
      status = method->get_outputs(outputs.data(), method->outputs_size());
      ET_CHECK(status == Error::Ok);
      // The following code assumes all output EValues are floating point
      // tensors. We need to handle other types of EValues and tensor
      // dtypes. Furthermore, we need a util to print tensors in a more
      // interpretable (e.g. size, dtype) and readable way.
      // TODO for the above at T159700776
      for (size_t output_index = 0; output_index < method->outputs_size();
           output_index++) {
        auto output_tensor = outputs[output_index].toTensor();
        auto output_file_name = FLAGS_output_folder_path + "/output_" +
            std::to_string(inference_index) + "_" +
            std::to_string(output_index) + ".raw";
        std::ofstream fout(output_file_name.c_str(), std::ios::binary);
        fout.write(
            output_tensor.const_data_ptr<char>(), output_tensor.nbytes());
        fout.close();
      }

      // Dump the profiling data to the specified file.
      torch::executor::prof_result_t prof_result;
      EXECUTORCH_DUMP_PROFILE_RESULTS(&prof_result);
      if (prof_result.num_bytes != 0) {
        FILE* ptr = fopen(FLAGS_prof_result_path.c_str(), "w+");
        fwrite(prof_result.prof_data, 1, prof_result.num_bytes, ptr);
        fclose(ptr);
      }
      ++inference_index;
    }
    ET_LOG(
        Info,
        "%d inference took %f ms, avg %f ms",
        inference_index,
        elapsed_time,
        elapsed_time / inference_index);
  } else {
    // if no input is provided, run with default input as executor_runner.
    Error status = method->execute();
    ET_CHECK_MSG(
        status == Error::Ok,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        status);
    ET_LOG(Info, "Model executed successfully.");
  }

  util::FreeInputs(inputs);
  return 0;
}
