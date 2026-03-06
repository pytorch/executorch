/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark runner for ExecuTorch models with TensorRT delegation.
 *
 * Discovers .pte and .onnx files in a directory and benchmarks each one.
 * For .pte files, uses ExecuTorch Module API with the TensorRT backend.
 * For .onnx files, compiles to a TRT engine and benchmarks natively.
 *
 * Usage:
 *   benchmark                         # all models in current dir
 *   benchmark -m mv3                  # mv3_tensorrt.pte in current dir
 *   benchmark -d /tmp/trt             # all models in /tmp/trt
 *   benchmark -d /tmp/trt -m mv3      # mv3 .pte and .onnx in /tmp/trt
 *   benchmark -n 200 -w 5             # 200 iterations, 5 warmup
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::aten::ScalarType;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Tag;

namespace {

constexpr uint32_t kDefaultIterations = 100;
constexpr uint32_t kDefaultWarmup = 3;
constexpr const char* kPteSuffix = ".pte";
constexpr const char* kOnnxSuffix = ".onnx";
constexpr const char* kTrtSuffix = "_tensorrt.pte";

struct Args {
  std::string model_dir = ".";
  std::string model_name;
  uint32_t iterations = kDefaultIterations;
  uint32_t warmup = kDefaultWarmup;
  bool verbose = false;
};

struct BenchmarkResult {
  std::string name;
  std::string format;
  uint32_t iterations{0};
  double avg_ms{0};
  double total_ms{0};
  size_t gpu_mem_kb{0};
  size_t cpu_mem_kb{0};
  bool success{false};
  std::string error;
};

// ---------------------------------------------------------------------------
// Memory measurement
// ---------------------------------------------------------------------------

/// Read process resident set size from /proc/self/status (Linux).
size_t get_rss_kb() {
  FILE* f = fopen("/proc/self/status", "r");
  if (!f) {
    return 0;
  }
  char line[256];
  size_t rss_kb = 0;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      sscanf(line + 6, "%zu", &rss_kb); // NOLINT(cert-err34-c)
      break;
    }
  }
  fclose(f);
  return rss_kb;
}

/// Get free GPU memory in KB via cudaMemGetInfo.
/// On unified memory systems (Jetson), this reflects free device memory
/// shared between CPU and GPU.
size_t get_gpu_free_kb() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
    return free_bytes / 1024;
  }
  return 0;
}

struct MemorySnapshot {
  size_t gpu_free_kb{0};
  size_t rss_kb{0};
};

MemorySnapshot take_memory_snapshot() {
  // Sync GPU to ensure all pending allocations/frees are reflected.
  cudaDeviceSynchronize();
  return {get_gpu_free_kb(), get_rss_kb()};
}

void record_memory_delta(
    const MemorySnapshot& before,
    const MemorySnapshot& after,
    BenchmarkResult& result) {
  result.gpu_mem_kb = (before.gpu_free_kb > after.gpu_free_kb)
      ? before.gpu_free_kb - after.gpu_free_kb
      : 0;
  result.cpu_mem_kb = (after.rss_kb > before.rss_kb)
      ? after.rss_kb - before.rss_kb
      : 0;
}

// ---------------------------------------------------------------------------
// TRT helpers
// ---------------------------------------------------------------------------

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      ET_LOG(Info, "TensorRT: %s", msg);
    }
  }
};

int64_t volume(const nvinfer1::Dims& dims) {
  int64_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    v *= dims.d[i];
  }
  return v;
}

size_t dtype_size(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kINT64:
      return 8;
    default:
      return 4;
  }
}

void compute_timing(
    et_timestamp_t total,
    uint32_t iterations,
    BenchmarkResult& result) {
  auto ratio = et_pal_ticks_to_ns_multiplier();
  result.total_ms = static_cast<double>(total) * ratio.numerator /
      ratio.denominator / 1000000.0;
  result.avg_ms = result.total_ms / iterations;
  result.success = true;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

void print_usage() {
  printf(
      "Usage: benchmark [options]\n"
      "\n"
      "Options:\n"
      "  -d, --model_dir DIR    Directory with .pte/.onnx files (default: .)\n"
      "  -m, --model_name NAME  Run only this model\n"
      "  -n, --num_executions N Timed iterations (default: %u)\n"
      "  -w, --warmup N         Warmup runs (default: %u)\n"
      "  -v, --verbose          Verbose logging\n"
      "  -h, --help             Show this message\n",
      kDefaultIterations,
      kDefaultWarmup);
}

bool parse_args(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage();
      return false;
    } else if (arg == "-v" || arg == "--verbose") {
      args.verbose = true;
    } else if ((arg == "-d" || arg == "--model_dir") && i + 1 < argc) {
      args.model_dir = argv[++i];
    } else if ((arg == "-m" || arg == "--model_name") && i + 1 < argc) {
      args.model_name = argv[++i];
    } else if ((arg == "-n" || arg == "--num_executions") && i + 1 < argc) {
      args.iterations = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
      args.warmup = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else {
      fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
      print_usage();
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// File discovery
// ---------------------------------------------------------------------------

bool ends_with(const std::string& s, const char* suffix) {
  size_t len = strlen(suffix);
  return s.size() >= len && s.compare(s.size() - len, len, suffix) == 0;
}

std::string stem(const std::string& path) {
  auto slash = path.rfind('/');
  const auto& filename =
      (slash != std::string::npos) ? path.substr(slash + 1) : path;
  if (ends_with(filename, kTrtSuffix)) {
    return filename.substr(0, filename.size() - strlen(kTrtSuffix));
  }
  if (ends_with(filename, kPteSuffix)) {
    return filename.substr(0, filename.size() - strlen(kPteSuffix));
  }
  if (ends_with(filename, kOnnxSuffix)) {
    return filename.substr(0, filename.size() - strlen(kOnnxSuffix));
  }
  return filename;
}

std::vector<std::string> find_models(
    const std::string& dir,
    const std::string& name) {
  std::vector<std::string> paths;

  if (!name.empty()) {
    for (const char* suffix : {kTrtSuffix, kPteSuffix, kOnnxSuffix}) {
      std::string path = dir + "/" + name + suffix;
      if (FILE* f = fopen(path.c_str(), "r")) {
        fclose(f);
        paths.push_back(path);
      }
    }
    return paths;
  }

  auto scan = [&](const std::string& d) {
    DIR* dp = opendir(d.c_str());
    if (!dp) {
      return;
    }
    while (auto* entry = readdir(dp)) {
      std::string f = entry->d_name;
      if (ends_with(f, kPteSuffix) || ends_with(f, kOnnxSuffix)) {
        paths.push_back(d + "/" + f);
      }
    }
    closedir(dp);
  };

  scan(dir);

  std::sort(paths.begin(), paths.end());
  return paths;
}

// ---------------------------------------------------------------------------
// PTE benchmark (ExecuTorch Module API)
// ---------------------------------------------------------------------------

BenchmarkResult benchmark_pte(
    const std::string& path,
    uint32_t iterations,
    uint32_t warmup,
    bool verbose) {
  BenchmarkResult result;
  result.name = stem(path);
  result.format = "pte";
  result.iterations = iterations;

  auto mem_before = take_memory_snapshot();

  Module module(path, Module::LoadMode::File);

  auto meta_result = module.method_meta("forward");
  if (!meta_result.ok()) {
    result.error = "Failed to get method metadata";
    return result;
  }
  auto meta = meta_result.get();

  std::vector<executorch::extension::TensorPtr> input_tensors;
  std::vector<std::vector<float>> float_data;
  std::vector<std::vector<int64_t>> int64_data;
  std::vector<std::vector<uint16_t>> bf16_data;
  std::vector<EValue> inputs;
  for (size_t i = 0; i < meta.num_inputs(); ++i) {
    auto tag = meta.input_tag(i);
    if (tag.ok() && tag.get() == Tag::Tensor) {
      auto tmeta = meta.input_tensor_meta(i);
      if (tmeta.ok()) {
        auto sizes = tmeta.get().sizes();
        auto dtype = tmeta.get().scalar_type();
        auto sizes_vec = std::vector<executorch::aten::SizesType>(
            sizes.begin(), sizes.end());
        int64_t numel = 1;
        for (auto s : sizes_vec) {
          numel *= s;
        }

        executorch::extension::TensorPtr tensor;
        if (dtype == ScalarType::Long || dtype == ScalarType::Int) {
          int64_data.emplace_back(numel, 1);
          tensor = executorch::extension::make_tensor_ptr(
              sizes_vec, int64_data.back());
        } else if (dtype == ScalarType::BFloat16) {
          bf16_data.emplace_back(numel, 0x3F80);
          tensor = executorch::extension::make_tensor_ptr(
              sizes_vec,
              bf16_data.back().data(),
              executorch::aten::ScalarType::BFloat16);
        } else {
          float_data.emplace_back(numel, 1.0f);
          tensor = executorch::extension::make_tensor_ptr(
              sizes_vec, float_data.back());
        }
        inputs.push_back(EValue(*tensor));
        input_tensors.push_back(std::move(tensor));
      }
    }
  }

  printf("  warming up ...\r");
  fflush(stdout);
  for (uint32_t i = 0; i < warmup; ++i) {
    auto r = module.forward(inputs);
    if (!r.ok()) {
      result.error = "Forward failed during warmup";
      return result;
    }
  }

  auto mem_after = take_memory_snapshot();
  record_memory_delta(mem_before, mem_after, result);

  et_timestamp_t total = 0;
  for (uint32_t i = 0; i < iterations; ++i) {
    printf("  [%u/%u]\r", i + 1, iterations);
    fflush(stdout);
    auto start = executorch::runtime::pal_current_ticks();
    auto r = module.forward(inputs);
    auto end = executorch::runtime::pal_current_ticks();
    if (!r.ok()) {
      result.error = "Forward failed at iteration " + std::to_string(i);
      return result;
    }
    total += end - start;
  }
  printf("              \r");

  compute_timing(total, iterations, result);
  return result;
}

// ---------------------------------------------------------------------------
// Raw TRT benchmark from PTE (extracts engine from delegate blob)
// ---------------------------------------------------------------------------

BenchmarkResult benchmark_pte_raw_trt(
    const std::string& path,
    uint32_t iterations,
    uint32_t warmup,
    bool verbose) {
  BenchmarkResult result;
  result.name = stem(path);
  result.format = "pte-raw";
  result.iterations = iterations;

  auto mem_before = take_memory_snapshot();

  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    result.error = "Cannot open file";
    return result;
  }
  fseek(f, 0, SEEK_END);
  long pos = ftell(f);
  if (pos < 0) {
    fclose(f);
    result.error = "Failed to determine file size";
    return result;
  }
  size_t file_size = static_cast<size_t>(pos);
  fseek(f, 0, SEEK_SET);
  std::vector<char> file_data(file_size);
  size_t bytes_read = fread(file_data.data(), 1, file_size, f);
  fclose(f);
  if (bytes_read != file_size) {
    result.error = "Failed to read file";
    return result;
  }

  const char kMagic[4] = {'T', 'R', '0', '1'};
  const void* engine_data = nullptr;
  size_t engine_size = 0;

  for (size_t i = 0; i + 32 <= file_size; ++i) {
    if (memcmp(file_data.data() + i, kMagic, 4) == 0) {
      const auto* hdr = reinterpret_cast<const uint8_t*>(file_data.data() + i);
      uint32_t eng_offset = 0;
      uint64_t eng_size = 0;
      memcpy(&eng_offset, hdr + 12, 4);
      memcpy(&eng_size, hdr + 16, 8);
      if (eng_size > 0 && i + eng_offset + eng_size <= file_size) {
        engine_data = file_data.data() + i + eng_offset;
        engine_size = static_cast<size_t>(eng_size);
        break;
      }
    }
  }

  if (!engine_data || engine_size == 0) {
    result.error = "Cannot find TRT engine in PTE file";
    return result;
  }

  TrtLogger logger;
  auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(logger));
  auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engine_data, engine_size));
  if (!engine) {
    result.error = "Failed to deserialize TRT engine from PTE";
    return result;
  }

  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine->createExecutionContext());

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    result.error = "Failed to create CUDA stream";
    return result;
  }

  std::vector<void*> buffers;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const auto* name = engine->getIOTensorName(i);
    auto shape = engine->getTensorShape(name);
    auto dt = engine->getTensorDataType(name);
    size_t bytes = static_cast<size_t>(volume(shape)) * dtype_size(dt);
    void* buf = nullptr;
    if (cudaMalloc(&buf, bytes) != cudaSuccess) {
      result.error = "Failed to allocate GPU buffer";
      for (const auto& b : buffers) {
        cudaFree(b);
      }
      cudaStreamDestroy(stream);
      return result;
    }
    cudaMemset(buf, 0, bytes);
    context->setTensorAddress(name, buf);
    buffers.push_back(buf);
  }

  printf("  warming up ...\r");
  fflush(stdout);
  for (uint32_t i = 0; i < warmup; ++i) {
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
  }

  auto mem_after = take_memory_snapshot();
  record_memory_delta(mem_before, mem_after, result);

  et_timestamp_t total = 0;
  for (uint32_t i = 0; i < iterations; ++i) {
    printf("  [%u/%u]\r", i + 1, iterations);
    fflush(stdout);
    auto start = executorch::runtime::pal_current_ticks();
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    auto end = executorch::runtime::pal_current_ticks();
    total += end - start;
  }
  printf("              \r");

  for (const auto& buf : buffers) {
    cudaFree(buf);
  }
  cudaStreamDestroy(stream);

  compute_timing(total, iterations, result);
  return result;
}

// ---------------------------------------------------------------------------
// ONNX benchmark (TensorRT native)
// ---------------------------------------------------------------------------

BenchmarkResult benchmark_onnx(
    const std::string& path,
    uint32_t iterations,
    uint32_t warmup,
    bool verbose) {
  BenchmarkResult result;
  result.name = stem(path);
  result.format = "onnx-trt";
  result.iterations = iterations;

  auto mem_before = take_memory_snapshot();

  TrtLogger logger;

  auto builder =
      std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    result.error = "Failed to create TRT builder";
    return result;
  }

  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(
          1 << static_cast<int>(
              nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger));

  if (verbose) {
    ET_LOG(Info, "Parsing ONNX: %s", path.c_str());
  }

  if (!parser->parseFromFile(
          path.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    result.error = "Failed to parse ONNX file";
    return result;
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
      builder->createBuilderConfig());
  config->setMemoryPoolLimit(
      nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
  if (config->getFlag(nvinfer1::BuilderFlag::kTF32)) {
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
  }

  if (verbose) {
    ET_LOG(Info, "Building TRT engine from ONNX...");
  }

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    result.error = "Failed to build TRT engine";
    return result;
  }

  auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(logger));
  auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine) {
    result.error = "Failed to deserialize engine";
    return result;
  }

  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine->createExecutionContext());

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    result.error = "Failed to create CUDA stream";
    return result;
  }

  std::vector<void*> buffers;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const auto* name = engine->getIOTensorName(i);
    auto shape = engine->getTensorShape(name);
    auto dt = engine->getTensorDataType(name);
    size_t bytes = static_cast<size_t>(volume(shape)) * dtype_size(dt);
    void* buf = nullptr;
    if (cudaMalloc(&buf, bytes) != cudaSuccess) {
      result.error = "Failed to allocate GPU buffer";
      for (const auto& b : buffers) {
        cudaFree(b);
      }
      cudaStreamDestroy(stream);
      return result;
    }
    cudaMemset(buf, 0, bytes);
    context->setTensorAddress(name, buf);
    buffers.push_back(buf);
  }

  printf("  warming up ...\r");
  fflush(stdout);
  for (uint32_t i = 0; i < warmup; ++i) {
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
  }

  auto mem_after = take_memory_snapshot();
  record_memory_delta(mem_before, mem_after, result);

  et_timestamp_t total = 0;
  for (uint32_t i = 0; i < iterations; ++i) {
    printf("  [%u/%u]\r", i + 1, iterations);
    fflush(stdout);
    auto start = executorch::runtime::pal_current_ticks();
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    auto end = executorch::runtime::pal_current_ticks();
    total += end - start;
  }
  printf("              \r");

  for (const auto& buf : buffers) {
    cudaFree(buf);
  }
  cudaStreamDestroy(stream);

  compute_timing(total, iterations, result);
  return result;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void print_summary(const std::vector<BenchmarkResult>& results) {
  printf("\n");
  printf(
      "%-20s %-10s %6s %10s %10s %10s %10s   %s\n",
      "MODEL",
      "FORMAT",
      "RUNS",
      "AVG (ms)",
      "TOTAL (ms)",
      "GPU (KB)",
      "CPU (KB)",
      "STATUS");
  printf(
      "%-20s %-10s %6s %10s %10s %10s %10s   %s\n",
      "--------------------",
      "----------",
      "------",
      "----------",
      "----------",
      "----------",
      "----------",
      "------");

  for (const auto& r : results) {
    if (r.success) {
      printf(
          "%-20s %-10s %6u %10.3f %10.3f %10zu %10zu   OK\n",
          r.name.c_str(),
          r.format.c_str(),
          r.iterations,
          r.avg_ms,
          r.total_ms,
          r.gpu_mem_kb,
          r.cpu_mem_kb);
    } else {
      printf(
          "%-20s %-10s %6s %10s %10s %10s %10s   FAIL: %s\n",
          r.name.c_str(),
          r.format.c_str(),
          "-",
          "-",
          "-",
          "-",
          "-",
          r.error.c_str());
    }
  }
  printf("\n");
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  Args args;
  if (!parse_args(argc, argv, args)) {
    return 1;
  }

  auto files = find_models(args.model_dir, args.model_name);
  if (files.empty()) {
    if (!args.model_name.empty()) {
      fprintf(
          stderr,
          "Error: model '%s' not found in '%s'\n",
          args.model_name.c_str(),
          args.model_dir.c_str());
    } else {
      fprintf(
          stderr,
          "Error: no .pte/.onnx files found in '%s'\n",
          args.model_dir.c_str());
    }
    return 1;
  }

  if (args.verbose) {
    ET_LOG(
        Info,
        "Found %zu model(s), warmup=%u, iterations=%u",
        files.size(),
        args.warmup,
        args.iterations);
  }

  std::vector<BenchmarkResult> results;
  for (const auto& path : files) {
    printf("Benchmarking: %s ...\n", path.c_str());

    BenchmarkResult result;
    if (ends_with(path, kOnnxSuffix)) {
      result = benchmark_onnx(path, args.iterations, args.warmup, args.verbose);
    } else {
      result = benchmark_pte(path, args.iterations, args.warmup, args.verbose);
    }

    if (result.success) {
      printf(
          "  %.3f ms avg (%u iterations), GPU %zu KB, CPU %zu KB\n",
          result.avg_ms,
          result.iterations,
          result.gpu_mem_kb,
          result.cpu_mem_kb);
    } else {
      printf("  FAILED: %s\n", result.error.c_str());
    }
    results.push_back(std::move(result));

    if (ends_with(path, kPteSuffix)) {
      printf("Benchmarking: %s (raw TRT) ...\n", path.c_str());
      auto raw_result = benchmark_pte_raw_trt(
          path, args.iterations, args.warmup, args.verbose);
      if (raw_result.success) {
        printf(
            "  %.3f ms avg (%u iterations), GPU %zu KB, CPU %zu KB\n",
            raw_result.avg_ms,
            raw_result.iterations,
            raw_result.gpu_mem_kb,
            raw_result.cpu_mem_kb);
      } else {
        printf("  FAILED: %s\n", raw_result.error.c_str());
      }
      results.push_back(std::move(raw_result));
    }
  }

  if (results.size() > 1) {
    print_summary(results);
  }

  return 0;
}
