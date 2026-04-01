/*
 * Copyright 2024-2026 NXP
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.

 * Example script to compile the model for the NXP Neutron NPU
 */

/*
 * This is an example ExecuTorch runner running on host CPU and Neutron
 * simulator - NSYS. Example illustrates how to use the ExecuTorch with the
 * Neutron Backend.
 */

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

using torch::executor::Error;
using torch::executor::Result;
using torch::executor::util::FileDataLoader;

static uint8_t __attribute__((
    aligned(16))) method_allocator_pool[512 * 1024 * 1024U]; // 512 MB
static uint8_t __attribute__((
    aligned(16))) tmp_allocator_pool[512 * 1024 * 1024U]; // 512 MB

#include <executorch/backends/nxp/runtime/NeutronDriver.h>
#include <executorch/backends/nxp/runtime/NeutronErrors.h>

#ifdef NEUTRON_CMODEL
// The following header is needed only for NSYS backend.
#include "NeutronEnvConfig.h"
#endif

#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <gflags/gflags.h>

DEFINE_string(model, "", "Path to serialized model.");
DEFINE_string(dataset, "", "Path to model dataset folder.");
DEFINE_string(
    inputs,
    "",
    "Path to inputs. Usage: "
    "/path/to/inputdata1,/path/to/inputdata2");
DEFINE_string(output, "", "Path to folder where output tensors are saved.");

#ifdef NEUTRON_CMODEL
DEFINE_string(firmware, "", "Relative path to firmware *.elf file.");
DEFINE_string(nsys, "", "Relative path to nsys.");
DEFINE_string(nsys_config, "", "Relative path to nsys config *.ini file");
#endif

#define DEFAULT_OUTPUT_DIR "results"

void processInputs(std::vector<std::string>& inputsData, std::string& inputs) {
  std::string segment;
  std::stringstream ss(inputs);

  while (std::getline(ss, segment, ',')) {
    inputsData.push_back(segment);
  }
}

bool isDirectory(std::string path) {
  struct stat sb;
  if (stat(path.c_str(), &sb) == -1) {
    fprintf(stderr, "Unable to determine stats of a path!\n");
    exit(-1);
  }
  return S_ISDIR(sb.st_mode);
}

void setInputs(
    torch::executor::Method& method,
    std::vector<std::string>& inputFiles) {
  if (method.inputs_size() != inputFiles.size()) {
    fprintf(
        stderr,
        "Mismatch: method has %ld inputs, whereas the loaded data contains %ld entries!\n",
        method.inputs_size(),
        inputFiles.size());
    exit(-1);
  }
  std::vector<torch::executor::EValue> values(method.inputs_size());
  Error status = method.get_inputs(values.data(), values.size());
  if (status != Error::Ok) {
    fprintf(stderr, "Failed to get_inputs...\n");
    exit(-1);
  }
  for (size_t i = 0; i < values.size(); i++) {
    fprintf(stderr, "Loading file %s\n", inputFiles[i].c_str());
    FILE* datasetFile = fopen(inputFiles[i].c_str(), "r");
    fseek(datasetFile, 0, SEEK_END);
    size_t inputSize = ftell(datasetFile);
    fseek(datasetFile, 0, SEEK_SET);
    if (inputSize == values[i].toTensor().nbytes()) {
      // Input is in floats
      fread(values[i].toTensor().mutable_data_ptr(), 1, inputSize, datasetFile);
    } else if (
        (inputSize == values[i].toTensor().numel()) &&
        (values[i].toTensor().scalar_type() ==
         torch::executor::ScalarType::Float)) {
      // Input is in bytes, convert to floats
      printf("Converting inputs to floats...\n");
      uint8_t* ptr = (uint8_t*)malloc(inputSize);
      fread(ptr, 1, inputSize, datasetFile);
      for (size_t j = 0; j < inputSize; j++) {
        values[i].toTensor().mutable_data_ptr<float>()[j] = ptr[j];
      }
      free(ptr);
    } else {
      // Input mismatch
      fprintf(
          stderr,
          "Mismatch in the %ld-th input tensor: expected %ld elements x %ld bytes each, loaded %ld bytes!\n",
          i,
          values[i].toTensor().numel(),
          values[i].toTensor().element_size(),
          inputSize);
      fclose(datasetFile);
      exit(-1);
    }
    fclose(datasetFile);
  }
}

void saveOutputs(
    torch::executor::Method& method,
    std::string& outputPath,
    const std::string& runPathPrefix = ".") {
  struct stat st;
  if (stat(outputPath.c_str(), &st) == -1) {
    mkdir(outputPath.c_str(), 0700);
  }
  if (stat((outputPath + "/" + runPathPrefix).c_str(), &st) == -1) {
    mkdir((outputPath + "/" + runPathPrefix).c_str(), 0700);
  }
  std::vector<torch::executor::EValue> values(method.outputs_size());
  Error status = method.get_outputs(values.data(), values.size());
  if (status != Error::Ok) {
    fprintf(stderr, "Failed to get_outputs...\n");
    exit(-1);
  }
  for (size_t i = 0; i < values.size(); i++) {
    int precision = 4 - std::to_string(i).size();
    std::string fileName = outputPath + "/" + runPathPrefix + "/" +
        std::to_string(i).insert(0, precision, '0') + ".bin";
    printf("Saving file %s\n", fileName.c_str());
    FILE* datasetFile = fopen(fileName.c_str(), "w");
    fwrite(
        values[i].toTensor().data_ptr(),
        1,
        values[i].toTensor().nbytes(),
        datasetFile);
    fclose(datasetFile);
  }
}

template <typename T>
void printClassificationOutput(
    const torch::executor::EValue& value,
    std::string& outputPath,
    const std::string& runPathPrefix) {
  T maxVal = value.toTensor().mutable_data_ptr<T>()[0];
  size_t maxIdx = 0;
  for (size_t j = 1; j < value.toTensor().numel(); j++) {
    T val = value.toTensor().mutable_data_ptr<T>()[j];
    if (val > maxVal) {
      maxVal = val;
      maxIdx = j;
    }
  }
  struct stat st;
  std::string resultsFile{outputPath + "/results.txt"};
  if (stat(outputPath.c_str(), &st) == -1) {
    mkdir(outputPath.c_str(), 0700);
  }
  FILE* results = fopen(resultsFile.c_str(), "a+");
  // Print classification results and save to results.txt.
  std::cout << "Top1 class " << runPathPrefix << " = " << maxIdx << std::endl;
  fprintf(results, "%s %d ", runPathPrefix.c_str(), maxIdx);
  std::cout << "Confidence = " << static_cast<float_t>(maxVal) << std::endl;
  fprintf(results, "%f ", static_cast<float_t>(maxVal));
  fprintf(results, "\n");
  fclose(results);
}

void printOutput(
    torch::executor::Method& method,
    std::string& outputPath,
    const std::string& runPathPrefix = ".") {
  // The single tensor is considered to be a classification result.
  if (method.outputs_size() == 1) {
    std::vector<torch::executor::EValue> values(method.outputs_size());
    Error status = method.get_outputs(values.data(), values.size());
    if (status != Error::Ok) {
      fprintf(stderr, "Failed to get_outputs...\n");
      exit(-1);
    }
    switch (values[0].toTensor().scalar_type()) {
      case torch::executor::ScalarType::Byte:
        printClassificationOutput<uint8_t>(
            values[0], outputPath, runPathPrefix);
        break;
      case torch::executor::ScalarType::Char:
        printClassificationOutput<int8_t>(values[0], outputPath, runPathPrefix);
        break;
      case torch::executor::ScalarType::Short:
        printClassificationOutput<int16_t>(
            values[0], outputPath, runPathPrefix);
        break;
      case torch::executor::ScalarType::Int:
        printClassificationOutput<int32_t>(
            values[0], outputPath, runPathPrefix);
        break;
      case torch::executor::ScalarType::Long:
        printClassificationOutput<int64_t>(
            values[0], outputPath, runPathPrefix);
        break;
      case torch::executor::ScalarType::Float:
        printClassificationOutput<float>(values[0], outputPath, runPathPrefix);
        break;
      case torch::executor::ScalarType::Double:
        printClassificationOutput<double>(values[0], outputPath, runPathPrefix);
        break;
      default:
        fprintf(
            stderr,
            "Unsupported tensor data type: %d\n",
            values[0].toTensor().scalar_type());
        exit(-1);
    }
  }
}

int main(int argc, char* argv[]) {
  DIR* datasetDir = nullptr;
  struct dirent* dataset = nullptr;

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Check that model name and inputs have been specified.
  if (FLAGS_model.empty()) {
    std::cout << "Please specify path to model using the --model option.\n";
    exit(-1);
  }
  if (FLAGS_dataset.empty() && FLAGS_inputs.empty()) {
    std::cout << "Please specify path to dataset using the --dataset option or "
                 "inputs using --inputs option\n";
    exit(-1);
  }
  if (!FLAGS_dataset.empty() && !FLAGS_inputs.empty()) {
    std::cout << "Cannot specify both inputs list and dataset directory\n";
    exit(-1);
  }
  if (!FLAGS_dataset.empty()) {
    datasetDir = opendir(FLAGS_dataset.c_str());
    if (!datasetDir) {
      std::cout << "Dataset path is not valid\n";
      exit(-1);
    }
  }
  if (FLAGS_output.empty()) {
    FLAGS_output = DEFAULT_OUTPUT_DIR;
  }

#ifdef NEUTRON_CMODEL
  if (!FLAGS_nsys_config.empty()) {
    storeNsysConfigPath(FLAGS_nsys_config.c_str());
  } else if (getenv("NSYS_CONFIG_PATH")) {
    storeNsysConfigPath(getenv("NSYS_CONFIG_PATH"));
  } else {
    std::cout << "ERROR: missing --nsys_config argument\n";
    exit(-1);
  }

  if (!FLAGS_firmware.empty()) {
    storeFirmwarePath(FLAGS_firmware.c_str());
  } else if (getenv("NSYS_FIRMWARE_PATH")) {
    storeFirmwarePath(getenv("NSYS_FIRMWARE_PATH"));
  } else {
    std::cout << "ERROR: missing --firmware argument\n";
    exit(-1);
  }

  if (!FLAGS_nsys.empty()) {
    storeNsysPath(FLAGS_nsys.c_str());
  } else if (getenv("NSYS_PATH")) {
    storeNsysPath(getenv("NSYS_PATH"));
  } else {
    std::cout << "ERROR: missing --nsys argument\n";
    exit(-1);
  }
#endif

  NeutronError error = ENONE;
  error = neutronInit();
  if (error != ENONE) {
    fprintf(stderr, "Internal Neutron NPU driver error %x in init!\n", error);
    exit(-1);
  }

  torch::executor::runtime_init();

  printf("Started..\n");

  Result<FileDataLoader> loader = FileDataLoader::from(FLAGS_model.c_str());
  if (!loader.ok()) {
    fprintf(stderr, "Model PTE loading failed\n");
    exit(-1);
  } else {
    printf("Model file %s loaded\n", FLAGS_model.c_str());
  }

  Result<torch::executor::Program> program =
      torch::executor::Program::load(&loader.get());
  if (!program.ok()) {
    fprintf(stderr, "Program loading failed\n");
    exit(-1);
  } else {
    printf("Program loaded\n");
  }

  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    if (!method_name_result.ok()) {
      fprintf(stderr, "Program has no methods...\n");
      exit(-1);
    }
    method_name = *method_name_result;
  }
  printf("Using method (%s)...\n", method_name);

  Result<torch::executor::MethodMeta> method_meta =
      program->method_meta(method_name);
  if (!method_meta.ok()) {
    fprintf(
        stderr,
        "Failed to get method_meta for (%s): %" PRIu32,
        method_name,
        (unsigned int)method_meta.error());
    exit(-1);
  }

  printf("Creating MemoryAllocator...\n");
  torch::executor::MemoryAllocator method_allocator{
      torch::executor::MemoryAllocator(
          sizeof(method_allocator_pool), method_allocator_pool)};
  torch::executor::MemoryAllocator tmp_allocator{
      torch::executor::MemoryAllocator(
          sizeof(tmp_allocator_pool), tmp_allocator_pool)};

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the memory
  std::vector<torch::executor::Span<uint8_t>>
      planned_spans; // Passed to the allocator
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();

  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    printf("Setting up planned buffer %lu, size %lu...\n", id, buffer_size);

    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }

  printf("Creating HierarchicalAllocator....\n");
  torch::executor::HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  torch::executor::MemoryManager memory_manager(
      &method_allocator, &planned_memory, &tmp_allocator);

  Result<torch::executor::Method> method =
      program->load_method(method_name, &memory_manager);
  if (!method.ok()) {
    fprintf(
        stderr,
        "Loading of method (%s) failed with status %" PRIu32 "...\n",
        method_name,
        (unsigned int)method.error());
    exit(-1);
  }
  printf("Method loaded...\n");

  Error status = Error::Ok;
  if (!FLAGS_dataset.empty()) {
    // Go through entire dataset for this model.
    FLAGS_dataset += "/";
    while (dataset = readdir(datasetDir)) {
      if (!strcmp(dataset->d_name, ".") || !strcmp(dataset->d_name, ".."))
        continue;

      std::vector<std::string> inputsData;
      inputsData.push_back(FLAGS_dataset + dataset->d_name);
      // Set input and call inferrence.
      setInputs(method.get(), inputsData);

      status = method->execute();
      if (status != Error::Ok) {
        fprintf(
            stderr,
            "Execution of method %s failed with status %" PRIu32 "...\n",
            method_name,
            (unsigned int)status);
        exit(-1);
      } else {
        printf("Method executed successfully...\n");
      }

      // Save outputs in binary files.
      saveOutputs(method.get(), FLAGS_output, dataset->d_name);
      // Print result with highest confidence.
      printOutput(method.get(), FLAGS_output, dataset->d_name);
    }
    closedir(datasetDir);
  } else if (!FLAGS_inputs.empty()) {
    std::vector<std::string> inputPaths;

    // Validate and process inputs and separate into two lists.
    processInputs(inputPaths, FLAGS_inputs);

    if (std::all_of(inputPaths.begin(), inputPaths.end(), isDirectory)) {
      // Inputs are in directories - use files in each directory as the inputs.
      std::vector<std::string> inputsData;
      for (std::string& inputDir : inputPaths) {
        datasetDir = opendir(inputDir.c_str());
        while (dataset = readdir(datasetDir)) {
          if (!strcmp(dataset->d_name, ".") || !strcmp(dataset->d_name, ".."))
            continue;

          inputsData.push_back(inputDir + "/" + dataset->d_name);
        }
        closedir(datasetDir);

        setInputs(method.get(), inputsData);

        status = method->execute();
        if (status != Error::Ok) {
          fprintf(
              stderr,
              "Execution of method %s failed with status %" PRIu32 "...\n",
              method_name,
              (unsigned int)status);
          exit(-1);
        } else {
          printf("Method executed successfully...\n");
        }

        if (inputDir.back() == '/')
          inputDir.pop_back();

        auto pos = inputDir.find_last_of('/');
        if (pos != std::string::npos)
          inputDir = inputDir.substr(pos + 1);

        // Save outputs in binary files.
        saveOutputs(method.get(), FLAGS_output, inputDir.c_str());
        inputsData.clear();
      }
    } else {
      // Inputs are files.
      setInputs(method.get(), inputPaths);

      status = method->execute();
      if (status != Error::Ok) {
        fprintf(
            stderr,
            "Execution of method %s failed with status %" PRIu32 "...\n",
            method_name,
            (unsigned int)status);
        exit(-1);
      } else {
        printf("Method executed successfully...\n");
      }

      // Save outputs in binary files.
      saveOutputs(method.get(), FLAGS_output);
    }
  }

  printf("Finished...\n");

  error = neutronDeinit();

  return 0;
}
