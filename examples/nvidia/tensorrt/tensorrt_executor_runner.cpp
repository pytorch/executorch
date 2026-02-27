/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * TensorRT Executor Runner - runs ExecuTorch models with TensorRT delegation
 * on NVIDIA GPUs.
 *
 * This tool loads .pte files exported with the TensorRT backend and executes
 * them using TensorRT for accelerated inference.
 *
 * Usage:
 *   tensorrt_executor_runner --model_path=model_tensorrt.pte
 *   tensorrt_executor_runner --model_path=model.pte --num_executions=10
 */

#include <cinttypes>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#ifdef USE_ATEN_LIB
#include <torch/torch.h>
#endif

namespace {

// Memory pools for the runtime
uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB
uint8_t temp_allocator_pool[1024U * 1024U]; // 1 MB

} // namespace

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
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

namespace {

struct RunnerArgs {
  std::string model_path = "model.pte";
  uint32_t num_executions = 1;
  bool verbose = false;
};

void print_usage(const char* program_name) {
  std::cerr << "Usage: " << program_name << " [options]\n"
            << "Options:\n"
            << "  --model_path=PATH    Path to .pte model file (default: "
               "model.pte)\n"
            << "  --num_executions=N   Number of times to run inference "
               "(default: 1)\n"
            << "  --verbose            Enable verbose output\n"
            << "  --help               Show this help message\n";
}

bool parse_args(int argc, char** argv, RunnerArgs& args) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return false;
    } else if (arg.rfind("--model_path=", 0) == 0) {
      args.model_path = arg.substr(13);
    } else if (arg.rfind("--num_executions=", 0) == 0) {
      args.num_executions =
          static_cast<uint32_t>(std::stoul(arg.substr(17)));
    } else if (arg == "--verbose" || arg == "-v") {
      args.verbose = true;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return false;
    }
  }
  return true;
}

void print_tensor_info(const Tensor& tensor, const char* name) {
  std::cout << name << ": shape=[";
  for (size_t i = 0; i < tensor.dim(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << tensor.size(i);
  }
  std::cout << "], dtype=" << static_cast<int>(tensor.scalar_type()) << "\n";
}

void print_outputs(Method& method) {
  const size_t num_outputs = method.outputs_size();
  std::cout << "\n=== Model Outputs (" << num_outputs << " total) ===\n";

  std::vector<EValue> outputs(num_outputs);
  Error status = method.get_outputs(outputs.data(), num_outputs);
  if (status != Error::Ok) {
    std::cerr << "Failed to get outputs\n";
    return;
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    std::cout << "Output " << i << ": ";
    if (outputs[i].isTensor()) {
      const auto& tensor = outputs[i].toTensor();
      print_tensor_info(tensor, "");

      // Print first few values for verification
      if (tensor.scalar_type() == ScalarType::Float) {
        const float* data = tensor.const_data_ptr<float>();
        const size_t numel = static_cast<size_t>(tensor.numel());
        const size_t print_count = std::min(numel, static_cast<size_t>(10));
        std::cout << "  Values: [";
        for (size_t j = 0; j < print_count; ++j) {
          if (j > 0) {
            std::cout << ", ";
          }
          std::cout << data[j];
        }
        if (numel > print_count) {
          std::cout << ", ... (" << numel - print_count << " more)";
        }
        std::cout << "]\n";
      }
    } else {
      std::cout << "(non-tensor)\n";
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  // Initialize ExecuTorch runtime
  executorch::runtime::runtime_init();

  // Parse command line arguments
  RunnerArgs args;
  if (!parse_args(argc, argv, args)) {
    return 1;
  }

  ET_LOG(Info, "TensorRT Executor Runner");
  ET_LOG(Info, "Model: %s", args.model_path.c_str());
  ET_LOG(Info, "Executions: %u", args.num_executions);

  // Load the model file
  Result<FileDataLoader> loader_result =
      FileDataLoader::from(args.model_path.c_str());
  if (!loader_result.ok()) {
    ET_LOG(
        Error,
        "Failed to create data loader for %s: 0x%" PRIx32,
        args.model_path.c_str(),
        static_cast<uint32_t>(loader_result.error()));
    return 1;
  }
  FileDataLoader loader = std::move(loader_result.get());
  ET_LOG(Info, "Model file loaded successfully");

  // Parse the program
  Result<Program> program_result = Program::load(&loader);
  if (!program_result.ok()) {
    ET_LOG(
        Error,
        "Failed to parse program: 0x%" PRIx32,
        static_cast<uint32_t>(program_result.error()));
    return 1;
  }
  Program program = std::move(program_result.get());
  ET_LOG(Info, "Program parsed successfully");

  // Get the first method name
  const char* method_name = nullptr;
  {
    const auto method_name_result = program.get_method_name(0);
    if (!method_name_result.ok()) {
      ET_LOG(Error, "Program has no methods");
      return 1;
    }
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Using method: %s", method_name);

  // Get method metadata for memory planning
  Result<MethodMeta> method_meta_result = program.method_meta(method_name);
  if (!method_meta_result.ok()) {
    ET_LOG(
        Error,
        "Failed to get method metadata: 0x%" PRIx32,
        static_cast<uint32_t>(method_meta_result.error()));
    return 1;
  }
  MethodMeta method_meta = std::move(method_meta_result.get());

  // Set up memory allocators
  MemoryAllocator method_allocator{
      MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};
  MemoryAllocator temp_allocator{
      MemoryAllocator(sizeof(temp_allocator_pool), temp_allocator_pool)};

  // Set up planned memory buffers
  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  const size_t num_buffers = method_meta.num_memory_planned_buffers();

  for (size_t id = 0; id < num_buffers; ++id) {
    const size_t buffer_size =
        static_cast<size_t>(method_meta.memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Allocating planned buffer %zu: %zu bytes", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }

  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  // Create memory manager
  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

  // Load the method
  Result<Method> method_result =
      program.load_method(method_name, &memory_manager);
  if (!method_result.ok()) {
    ET_LOG(
        Error,
        "Failed to load method %s: 0x%" PRIx32,
        method_name,
        static_cast<uint32_t>(method_result.error()));
    return 1;
  }
  Method method = std::move(method_result.get());
  ET_LOG(Info, "Method loaded successfully");

  // Prepare inputs once (fills with ones by default)
  auto inputs_result =
      executorch::extension::prepare_input_tensors(method, {}, {});
  if (!inputs_result.ok()) {
    ET_LOG(
        Error,
        "Failed to prepare inputs: 0x%" PRIx32,
        static_cast<uint32_t>(inputs_result.error()));
    return 1;
  }
  auto inputs = std::move(inputs_result.get());

  if (args.verbose) {
    ET_LOG(Info, "Inputs prepared: %zu tensors", method.inputs_size());
  }

  // Warmup runs to ensure GPU is fully initialized
  constexpr uint32_t kWarmupRuns = 3;
  for (uint32_t i = 0; i < kWarmupRuns; ++i) {
    Error status = method.execute();
    if (status != Error::Ok) {
      ET_LOG(
          Error,
          "Warmup execution failed: 0x%" PRIx32,
          static_cast<uint32_t>(status));
      return 1;
    }
  }

  if (args.verbose) {
    ET_LOG(Info, "Warmup completed (%u runs)", kWarmupRuns);
  }

  // Run benchmark
  et_timestamp_t total_time = 0;

  for (uint32_t i = 0; i < args.num_executions; ++i) {
    const et_timestamp_t start = executorch::runtime::pal_current_ticks();
    Error status = method.execute();
    const et_timestamp_t end = executorch::runtime::pal_current_ticks();
    total_time += end - start;

    if (status != Error::Ok) {
      ET_LOG(
          Error,
          "Execution failed: 0x%" PRIx32,
          static_cast<uint32_t>(status));
      return 1;
    }
  }

  // Calculate timing
  const auto tick_ratio = et_pal_ticks_to_ns_multiplier();
  constexpr double NS_PER_MS = 1000000.0;
  const double total_ms = static_cast<double>(total_time) *
      tick_ratio.numerator / tick_ratio.denominator / NS_PER_MS;
  const double avg_ms = total_ms / args.num_executions;

  ET_LOG(
      Info,
      "Executed %u time(s) in %.3f ms total (%.3f ms avg)",
      args.num_executions,
      total_ms,
      avg_ms);

  // Print outputs
  print_outputs(method);

  ET_LOG(Info, "TensorRT execution completed successfully");
  return 0;
}
