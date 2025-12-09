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
 * User could specify arguments like desired input data, iterations, etc.
 * Currently we assume that the outputs are all fp32 tensors.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <gflags/gflags.h>

#include <chrono>
#include <fstream>
#include <memory>
#include <numeric>
static uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    output_folder_path,
    "outputs",
    "Executorch inference data output path.");
DEFINE_string(input_list_path, "input_list.txt", "Model input list path.");
DEFINE_int32(iteration, 1, "Iterations of inference.");
DEFINE_int32(warm_up, 0, "Pre-run before inference.");
DEFINE_bool(
    shared_buffer,
    false,
    "Specifies to use shared buffers for zero-copy usecase between the application and device/co-processor associated with the backend.");
DEFINE_uint32(method_index, 0, "Index of methods to be specified.");

DEFINE_string(
    etdump_path,
    "etdump.etdp",
    "If etdump generation is enabled an etdump will be written out to this path");

DEFINE_bool(
    dump_intermediate_outputs,
    false,
    "Dump intermediate outputs to etdump file.");

DEFINE_string(
    debug_output_path,
    "debug_output.bin",
    "Path to dump debug outputs to.");

DEFINE_string(
    input_shape_path,
    "",
    "Path to file with input shapes specified (used in dynamic shape scenario).");

DEFINE_string(
    output_shape_path,
    "",
    "Path to file with output shapes specified (used in dynamic shape scenario).");

DEFINE_int32(
    debug_buffer_size,
    20000000, // 20MB
    "Size of the debug buffer in bytes to allocate for intermediate outputs and program outputs logging.");

DEFINE_string(
    performance_output_path,
    "inference_speed.txt",
    "Records inference speed. For CI purpose.");

DEFINE_int32(
    log_level,
    0,
    "Log level between 1-5, higher is more verbose. "
    "This is a runtime option and will override the log level set during AOT. "
    "Refer to QnnExecuTorchLogLevel under qc_compiler_spec.fbs for more info.");
DEFINE_int32(
    htp_performance_mode,
    0,
    "HTP Performance mode between 0-8. "
    "This is a runtime option and will override the performance mode set during AOT. "
    "Refer to QnnExecuTorchHtpPerformanceMode under qc_compiler_spec.fbs for more info.");
DEFINE_int32(
    profile_level,
    0,
    "Profile level between 0-2. "
    "Level 3(Optrace) must be turned on during AOT and cannot be enabled during runtime. "
    "This is a runtime option and will override the profile level set during AOT. "
    "Refer to QnnExecuTorchProfileLevel under qc_compiler_spec.fbs for more info.");

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::etdump::ETDumpGen;
using executorch::etdump::ETDumpResult;
using executorch::extension::FileDataLoader;
using executorch::extension::prepare_input_tensors;
using executorch::runtime::BackendOption;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::EventTracerDebugLogLevel;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;

class CustomMemory {
 public:
  CustomMemory(bool shared_buffer) : shared_buffer_(shared_buffer){};
  bool Allocate(size_t bytes, size_t alignment) {
    if (shared_buffer_) {
      ptr_ = QnnExecuTorchAllocCustomMem(bytes, alignment);
    } else {
      input_data_.resize(bytes);
      ptr_ = input_data_.data();
    }
    return ptr_ != nullptr;
  }

  ~CustomMemory() {
    if (shared_buffer_) {
      if (ptr_ != nullptr) {
        QnnExecuTorchFreeCustomMem(ptr_);
      }
    }
  }

  void* GetPtr() {
    return ptr_;
  }

  CustomMemory(const CustomMemory&) = delete;
  CustomMemory(CustomMemory&&) = delete;
  CustomMemory& operator=(const CustomMemory&) = delete;
  CustomMemory& operator=(CustomMemory&&) = delete;

 private:
  bool shared_buffer_{false};
  void* ptr_{nullptr};
  std::vector<char> input_data_;
};

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

  // Set runtime options
  executorch::runtime::BackendOptions<3> backend_options;
  if (!gflags::GetCommandLineFlagInfoOrDie("log_level").is_default) {
    ET_LOG(Info, "Setting runtime log level: %d", FLAGS_log_level);
    ET_CHECK_MSG(
        backend_options.set_option(QNN_RUNTIME_LOG_LEVEL, FLAGS_log_level) ==
            Error::Ok,
        "Failed to set backend options: %s",
        QNN_RUNTIME_LOG_LEVEL);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("htp_performance_mode").is_default) {
    ET_LOG(
        Info,
        "Setting runtime performance mode: %d",
        FLAGS_htp_performance_mode);
    ET_CHECK_MSG(
        backend_options.set_option(
            QNN_RUNTIME_HTP_PERFORMANCE_MODE, FLAGS_htp_performance_mode) ==
            Error::Ok,
        "Failed to set backend options: %s",
        QNN_RUNTIME_HTP_PERFORMANCE_MODE);
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("profile_level").is_default) {
    ET_LOG(Info, "Setting runtime profile level: %d", FLAGS_profile_level);
    ET_CHECK_MSG(
        backend_options.set_option(
            QNN_RUNTIME_PROFILE_LEVEL, FLAGS_profile_level) == Error::Ok,
        "Failed to set backend options: %s",
        QNN_RUNTIME_PROFILE_LEVEL);
  }
  ET_CHECK_MSG(
      set_option(QNN_BACKEND, backend_options.view()) == Error::Ok,
      "Failed to set runtime options.");

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point to data that's already in memory, and
  // users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();
  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(
      loader.ok(),
      "FileDataLoader::from() failed: 0x%" PRIx32,
      (int)loader.error());

  // Parse the program file. This is immutable, and can also be reused between
  // multiple execution invocations across multiple threads.
  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model file %s", model_path);
    return 1;
  }
  ET_LOG(Info, "Model file %s is loaded.", model_path);

  // Use the designated method in the program, default to the first one
  const char* method_name = nullptr;
  {
    const auto method_name_result =
        program->get_method_name(FLAGS_method_index);
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
  ETDumpGen etdump_gen;
  Result<Method> method =
      program->load_method(method_name, &memory_manager, &etdump_gen);
  ET_CHECK_MSG(
      method.ok(),
      "Loading of method %s failed with status 0x%" PRIx32,
      method_name,
      (int)method.error());
  ET_LOG(Info, "Method loaded.");

  void* debug_buffer;
  if (FLAGS_dump_intermediate_outputs) {
    debug_buffer = malloc(FLAGS_debug_buffer_size);
    Span<uint8_t> buffer((uint8_t*)debug_buffer, FLAGS_debug_buffer_size);
    etdump_gen.set_debug_buffer(buffer);
    etdump_gen.set_event_tracer_debug_level(
        EventTracerDebugLogLevel::kIntermediateOutputs);
  }

  // Prepare the inputs.
  // Allocate data memory for inputs and outputs
  std::vector<std::unique_ptr<CustomMemory>> in_custom_mem;
  std::vector<std::unique_ptr<CustomMemory>> out_custom_mem;
  in_custom_mem.reserve(method->inputs_size());
  out_custom_mem.reserve(method->outputs_size());

  for (int input_index = 0; input_index < method->inputs_size();
       ++input_index) {
    MethodMeta method_meta = method->method_meta();
    Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(input_index);
    in_custom_mem.push_back(
        std::make_unique<CustomMemory>(FLAGS_shared_buffer));
    std::unique_ptr<CustomMemory>& custom_mem_ptr = in_custom_mem.back();
    ET_CHECK_MSG(
        custom_mem_ptr->Allocate(
            tensor_meta->nbytes(), MemoryAllocator::kDefaultAlignment),
        "Failed to allocate custom memory. tensor index: %d, bytes: %zu",
        input_index,
        tensor_meta->nbytes());
    TensorImpl impl = TensorImpl(
        tensor_meta->scalar_type(),
        /*dim=*/tensor_meta->sizes().size(),
        const_cast<TensorImpl::SizesType*>(tensor_meta->sizes().data()),
        custom_mem_ptr->GetPtr(),
        const_cast<TensorImpl::DimOrderType*>(tensor_meta->dim_order().data()));
    Error ret = method->set_input(Tensor(&impl), input_index);
    ET_CHECK_MSG(ret == Error::Ok, "Failed to set input tensor: %d", (int)ret);
  }
  for (int output_index = 0; output_index < method->outputs_size();
       ++output_index) {
    const Tensor& t = method->get_output(output_index).toTensor();
    out_custom_mem.push_back(
        std::make_unique<CustomMemory>(FLAGS_shared_buffer));
    std::unique_ptr<CustomMemory>& custom_mem_ptr = out_custom_mem.back();
    ET_CHECK_MSG(
        custom_mem_ptr->Allocate(
            t.nbytes(), MemoryAllocator::kDefaultAlignment),
        "Failed to allocate custom memory. tensor index: %d, bytes: %zu",
        output_index,
        t.nbytes());
    Error ret = method->set_output_data_ptr(
        custom_mem_ptr->GetPtr(), t.nbytes(), output_index);
    if (ret != Error::Ok) {
      // This can error if the outputs are already pre-allocated. Ignore
      // this error because it doesn't affect correctness, but log it.
      ET_LOG(
          Info,
          "ignoring error from set_output_data_ptr(): 0x%" PRIx32,
          (int)ret);
    }
  }
  ET_LOG(Info, "Inputs prepared.");

  // Fill in data for input
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

    // dynamic shape related
    std::vector<std::vector<int32_t>> expected_input_shapes,
        expected_output_shapes;
    if (!FLAGS_input_shape_path.empty() && !FLAGS_output_shape_path.empty()) {
      std::ifstream input_shape_list(FLAGS_input_shape_path);
      std::ifstream output_shape_list(FLAGS_output_shape_path);
      std::string shape_content;
      while (std::getline(input_shape_list, shape_content)) {
        auto dims = split(shape_content, ", ");
        std::vector<int32_t> shape;
        for (std::string& dim : dims) {
          shape.push_back(std::stoi(dim));
        }
        expected_input_shapes.emplace_back(std::move(shape));
      }
      while (std::getline(output_shape_list, shape_content)) {
        auto dims = split(shape_content, ", ");
        std::vector<int32_t> shape;
        for (std::string& dim : dims) {
          shape.push_back(std::stoi(dim));
        }
        expected_output_shapes.emplace_back(std::move(shape));
      }
    }

    std::string file_path;
    int inference_index = 0;
    double elapsed_time = 0;
    while (std::getline(input_list, file_path)) {
      // to avoid case where \r\n is used as EOL
      if (!file_path.empty() && file_path.back() == '\r') {
        file_path.pop_back();
      }

      auto input_files = split(file_path, " ");
      if (input_files.size() == 0) {
        break;
      }
      ET_CHECK_MSG(
          input_files.size() == num_inputs,
          "Number of inputs (%zu) mismatch with input files (%zu)",
          num_inputs,
          input_files.size());

      // dynamic shape related
      if (!expected_input_shapes.empty()) {
        ET_CHECK_MSG(
            expected_input_shapes.size() == num_inputs,
            "Number of inputs (%zu) mismatch with input shapes (%zu)",
            num_inputs,
            expected_input_shapes.size());
      }

      for (int input_index = 0; input_index < num_inputs; ++input_index) {
        MethodMeta method_meta = method->method_meta();
        Result<TensorInfo> tensor_meta =
            method_meta.input_tensor_meta(input_index);

        std::ifstream fin(input_files[input_index], std::ios::binary);
        fin.seekg(0, fin.end);
        size_t file_size = fin.tellg();

        fin.seekg(0, fin.beg);
        fin.read(
            static_cast<char*>(in_custom_mem[input_index]->GetPtr()),
            file_size);
        fin.close();

        if (expected_input_shapes.empty()) {
          ET_CHECK_MSG(
              file_size == tensor_meta->nbytes(),
              "Input(%d) size mismatch. file bytes: %zu, tensor bytes: %zu",
              input_index,
              file_size,
              tensor_meta->nbytes());
        }

        // For pre-allocated use case, we need to call set_input
        // to copy data for the input tensors since they doesn't
        // share the data with in_custom_mem.
        TensorImpl impl = TensorImpl(
            tensor_meta->scalar_type(),
            /*dim=*/tensor_meta->sizes().size(),
            const_cast<TensorImpl::SizesType*>(
                expected_input_shapes.empty()
                    ? tensor_meta->sizes().data()
                    : expected_input_shapes[input_index].data()),
            in_custom_mem[input_index]->GetPtr(),
            const_cast<TensorImpl::DimOrderType*>(
                tensor_meta->dim_order().data()));
        Error ret = method->set_input(Tensor(&impl), input_index);
        ET_CHECK_MSG(
            ret == Error::Ok, "Failed to set input tensor: %d", (int)ret);
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
          (int)status);

      std::vector<EValue> outputs(method->outputs_size());
      status = method->get_outputs(outputs.data(), method->outputs_size());
      ET_CHECK(status == Error::Ok);
      for (size_t output_index = 0; output_index < method->outputs_size();
           output_index++) {
        auto output_tensor = outputs[output_index].toTensor();
        size_t nbytes = output_tensor.nbytes();
        if (!expected_output_shapes.empty()) {
          nbytes = std::accumulate(
              expected_output_shapes[output_index].begin(),
              expected_output_shapes[output_index].end(),
              executorch::runtime::elementSize(output_tensor.scalar_type()),
              std::multiplies<int>());
        }
        auto output_file_name = FLAGS_output_folder_path + "/output_" +
            std::to_string(inference_index) + "_" +
            std::to_string(output_index) + ".raw";
        std::ofstream fout(output_file_name.c_str(), std::ios::binary);
        fout.write(output_tensor.const_data_ptr<char>(), nbytes);
        fout.close();
      }

      ++inference_index;
    }
    ET_LOG(
        Info,
        "Total %d inference took %f ms, avg %f ms",
        inference_index,
        elapsed_time,
        elapsed_time / inference_index);

    // Save avg inference time for CI
    std::ofstream outfile(FLAGS_performance_output_path.c_str());
    if (outfile.is_open()) {
      double avg_time = elapsed_time / inference_index;
      outfile << avg_time;
      outfile.close();
    } else {
      ET_CHECK_MSG(false, "Error saving the inference speed file");
    }
  } else {
    // if no input is provided, fill the inputs with default values
    auto inputs = prepare_input_tensors(*method);
    ET_CHECK_MSG(
        inputs.ok(),
        "Could not prepare inputs: 0x%" PRIx32,
        (uint32_t)inputs.error());
    ET_LOG(
        Info,
        "Input list not provided. Inputs prepared with default values set.");

    // Run the method
    Error status = method->execute();
    ET_CHECK_MSG(
        status == Error::Ok,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        (int)status);
    ET_LOG(Info, "Model executed successfully.");

    // Warm up
    ET_LOG(Info, "Perform %d inferences for warming up", FLAGS_warm_up);
    for (int i = 0; i < FLAGS_warm_up; ++i) {
      status = method->execute();
    }

    // Inference with designated iterations
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

    ET_LOG(
        Info,
        "%d inferences took %f ms, avg %f ms",
        FLAGS_iteration,
        interval_infs,
        interval_infs / (float)FLAGS_iteration);
  }

  // Dump the etdump data containing profiling/debugging data to the specified
  // file.
  ETDumpResult result = etdump_gen.get_etdump_data();
  if (result.buf != nullptr && result.size > 0) {
    ET_LOG(
        Info,
        "Write etdump to %s, Size = %zu",
        FLAGS_etdump_path.c_str(),
        result.size);
    FILE* f = fopen(FLAGS_etdump_path.c_str(), "w+");
    fwrite((uint8_t*)result.buf, 1, result.size, f);
    fclose(f);
    free(result.buf);
  }

  if (FLAGS_dump_intermediate_outputs) {
    ET_LOG(
        Info,
        "Write debug output binary to %s, Size = %zu",
        FLAGS_debug_output_path.c_str(),
        (size_t)FLAGS_debug_buffer_size);
    FILE* f = fopen(FLAGS_debug_output_path.c_str(), "w+");
    fwrite((uint8_t*)debug_buffer, 1, FLAGS_debug_buffer_size, f);
    fclose(f);
    free(debug_buffer);
  }

  return 0;
}
