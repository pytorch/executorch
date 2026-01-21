/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/direct_mode/QnnExecuTorchIdlWrapper.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::etdump::ETDumpResult;
using executorch::extension::FileDataLoader;
using executorch::runtime::Error;
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

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

namespace executorch {
namespace backends {
namespace qnn {

// Code logic below is similar to qnn_executor_runner to load pte file.
QnnExecuTorchIdlWrapper::QnnExecuTorchIdlWrapper(
    const char* pte_path,
    const int method_index) {
  Result<FileDataLoader> loader =
      FileDataLoader::from(pte_path, QNN_CTX_BIN_ALIGNMENT);
  FARF(RUNTIME_HIGH, "Loading file %s.", pte_path);
  if (!loader.ok()) {
    FARF(
        RUNTIME_ERROR,
        "FileDataLoader::from() failed: 0x%x",
        (int)loader.error());
    return;
  }
  loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));

  Result<Program> program = Program::load(loader_.get());
  if (!program.ok()) {
    FARF(RUNTIME_ERROR, "Failed to parse model file %s", pte_path);
    return;
  }
  program_ = std::make_unique<Program>(std::move(program.get()));

  FARF(RUNTIME_HIGH, "Model file %s is loaded.", pte_path);
  auto method_name = program_->get_method_name(method_index);
  if (!method_name.ok()) {
    FARF(RUNTIME_ERROR, "Program has no methods.");
    return;
  } else {
    FARF(RUNTIME_HIGH, "Using method %s", method_name.get());
  }

  Result<MethodMeta> method_meta = program_->method_meta(method_name.get());
  if (!method_meta.ok()) {
    FARF(
        RUNTIME_ERROR,
        "Failed to get method_meta for %s: 0x%x",
        method_name.get(),
        (unsigned int)method_meta.error());
    return;
  }
  method_meta_ = std::make_unique<MethodMeta>(std::move(method_meta.get()));

  memory_allocator_ = std::make_unique<MemoryAllocator>(
      sizeof(method_allocator_pool_), method_allocator_pool_);
  size_t num_memory_planned_buffers =
      method_meta_->num_memory_planned_buffers();
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta_->memory_planned_buffer_size(id).get());
    FARF(
        RUNTIME_HIGH,
        "Setting up planned buffer %zu, size %zu.",
        id,
        buffer_size);
    planned_buffers_.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans_.push_back({planned_buffers_.back().get(), buffer_size});
    planned_memory_ = std::make_unique<HierarchicalAllocator>(
        Span<Span<uint8_t>>{planned_spans_.data(), planned_spans_.size()});
    memory_manager_ = std::make_unique<MemoryManager>(
        memory_allocator_.get(), planned_memory_.get());
  }

  Result<Method> method =
      program_->load_method(*method_name, memory_manager_.get(), &etdump_gen_);
  if (!method.ok()) {
    FARF(
        RUNTIME_ERROR,
        "Loading of method %s failed with status 0x%" PRIx32,
        *method_name,
        (int)method.error());
    return;
  } else {
    FARF(RUNTIME_HIGH, "Method loaded.");
  }
  method_ = std::make_unique<Method>(std::move(method.get()));

  input_tensors_.resize(method_->inputs_size());
  for (int i = 0; i < input_tensors_.size(); i++) {
    Result<TensorInfo> tensor_info = method_meta_->input_tensor_meta(i);
    input_tensors_[i].resize(tensor_info->nbytes());
    input_tensor_impls_.emplace_back(TensorImpl(
        tensor_info->scalar_type(),
        tensor_info->sizes().size(),
        const_cast<TensorImpl::SizesType*>(tensor_info->sizes().data()),
        input_tensors_[i].data(),
        const_cast<TensorImpl::DimOrderType*>(
            tensor_info->dim_order().data())));
    Error ret = method_->set_input(Tensor(&input_tensor_impls_.back()), i);
    if (ret != Error::Ok) {
      FARF(RUNTIME_ERROR, "Failed to set input tensor: %d", (int)ret);
      return;
    }
  }

  output_tensors_.resize(method_->outputs_size());
  for (int i = 0; i < output_tensors_.size(); ++i) {
    Result<TensorInfo> tensor_info = method_meta_->output_tensor_meta(i);
    output_tensors_[i].resize(tensor_info->nbytes());
    Error ret = method_->set_output_data_ptr(
        output_tensors_[i].data(), tensor_info->nbytes(), i);
    if (ret != Error::Ok) {
      FARF(RUNTIME_ERROR, "Failed to set output tensor: %d", (int)ret);
      return;
    }
  }
}

Error QnnExecuTorchIdlWrapper::execute_all(
    const char* input_list_path,
    const char* output_folder_path,
    int& num_inferences_performed,
    double& total_execute_interval,
    double& total_read_file_interval,
    double& total_save_file_interval) {
  Error status = Error::Ok;
  TimePoint execute_start, execute_end, read_start, read_end, save_start,
      save_end;
  std::ifstream input_list(input_list_path);
  int inference_index = 0;
  if (input_list.is_open()) {
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
    while (std::getline(input_list, file_path)) {
      auto input_files = split(file_path, " ");
      if (input_files.size() == 0) {
        break;
      }
      if (input_files.size() != method_->inputs_size()) {
        FARF(
            RUNTIME_ERROR,
            "Input file provides %d inputs while model has %d inputs",
            input_files.size(),
            method_->inputs_size());
        status = Error::Internal;
        return status;
      }
      for (int i = 0; i < method_->inputs_size(); i++) {
        read_start = Clock::now();
        int fd = open(input_files[i].c_str(), O_RDONLY);
        if (fd == -1) {
          FARF(RUNTIME_ERROR, "Failed to open input file.");
          status = Error::Internal;
          return status;
        }

        ssize_t bytes =
            read(fd, input_tensors_[i].data(), input_tensors_[i].size());
        if (bytes < 0) {
          FARF(RUNTIME_ERROR, "Failed to read data from file to input_tensor.");
          status = Error::Internal;
          return status;
        }
        close(fd);
        read_end = Clock::now();
        total_read_file_interval +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                read_end - read_start)
                .count() /
            1000.0;
      }

      execute_start = Clock::now();
      status = method_->execute();
      execute_end = Clock::now();
      total_execute_interval +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              execute_end - execute_start)
              .count() /
          1000.0;
      if (status != Error::Ok) {
        FARF(RUNTIME_ERROR, "Execution failed with status 0x%x", (int)status);
        return status;
      }

      save_start = Clock::now();
      for (size_t i = 0; i < method_->outputs_size(); i++) {
        auto output_file_name = std::string(output_folder_path) + "/output_" +
            std::to_string(inference_index) + "_" + std::to_string(i) + ".raw";
        int fd = open(output_file_name.c_str(), O_WRONLY | O_CREAT, 0644);
        if (fd == -1) {
          FARF(RUNTIME_ERROR, "Failed to open output file.");
          status = Error::Internal;
          return status;
        }

        ssize_t bytes =
            write(fd, output_tensors_[i].data(), output_tensors_[i].size());
        if (bytes < 0) {
          FARF(RUNTIME_ERROR, "Failed to write data to output file.");
          close(fd);
          status = Error::Internal;
          return status;
        }
        close(fd);
      }
      save_end = Clock::now();
      total_save_file_interval +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              save_end - save_start)
              .count() /
          1000.0;
      ++inference_index;
    }
  }
  num_inferences_performed = inference_index;
  return status;
}

Error QnnExecuTorchIdlWrapper::enable_intermediate_tensor_dump(
    const int debug_buffer_size) {
  debug_buffer_ = malloc(debug_buffer_size);
  debug_buffer_size_ = debug_buffer_size;
  Span<uint8_t> buffer((uint8_t*)debug_buffer_, debug_buffer_size_);
  etdump_gen_.set_debug_buffer(buffer);
  etdump_gen_.set_event_tracer_debug_level(
      EventTracerDebugLogLevel::kIntermediateOutputs);
  return Error::Ok;
}

Error QnnExecuTorchIdlWrapper::dump_etdp(const char* etdump_path) {
  Error status = Error::Ok;
  ETDumpResult result = etdump_gen_.get_etdump_data();
  if (result.buf != nullptr && result.size > 0) {
    FARF(
        RUNTIME_HIGH,
        "Write etdump to %s, size = %zu",
        etdump_path,
        result.size);
    FILE* f = fopen(etdump_path, "w+");
    fwrite((uint8_t*)result.buf, 1, result.size, f);
    fclose(f);
    free(result.buf);
  } else {
    FARF(RUNTIME_ERROR, "Unable to generate etdump to %s", etdump_path);
    status = Error::Internal;
  }
  return status;
}

Error QnnExecuTorchIdlWrapper::dump_intermediate_tensor(
    const char* debug_output_path) {
  Error status = Error::Ok;
  if (debug_buffer_ != nullptr) {
    FARF(
        RUNTIME_HIGH,
        "Write debug output binary to %s, Size = %zu",
        debug_output_path,
        debug_buffer_size_);
    FILE* f = fopen(debug_output_path, "w+");
    fwrite((uint8_t*)debug_buffer_, 1, debug_buffer_size_, f);
    fclose(f);
    free(debug_buffer_);
  } else {
    FARF(
        RUNTIME_ERROR,
        "Unable to dump intermediate tensor, please ensure intermediate_tensor_dump is enabled prior to execution.");
    status = Error::Internal;
  }
  return status;
}

} // namespace qnn
} // namespace backends
} // namespace executorch
