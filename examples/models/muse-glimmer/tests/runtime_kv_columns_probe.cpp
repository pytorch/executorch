/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A serialized diagnostic transport over the actual CUDA batching executor.
// Raw cache IDs are never mixed with ServingRuntime or SessionTable IDs.
#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <gflags/gflags.h>
#include <nlohmann/json.hpp>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define ET_RUNTIME_KV_PROBE_HAS_NVTX 1
#else
#define ET_RUNTIME_KV_PROBE_HAS_NVTX 0
#endif
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

DEFINE_string(pte, "", "Solo text-only CUDA model");
DEFINE_string(data_path, "", "External weight blob");
DEFINE_string(
    output_dir,
    "",
    "Existing unique directory for numerical outputs");
DEFINE_int32(
    max_sessions,
    21,
    "Logical cache admission including snapshot owners");
DEFINE_int32(max_context, 131072, "Logical model context");
DEFINE_int32(initial_kv_capacity, 512, "Initial logical rows per column");

namespace cb = executorch::backends::cuda::batching;
using executorch::runtime::Error;
using Json = nlohmann::json;

namespace {

class DiagnosticRange final {
 public:
  explicit DiagnosticRange(const std::string& label) {
#if ET_RUNTIME_KV_PROBE_HAS_NVTX
    nvtxRangePushA(label.c_str());
#else
    (void)label;
#endif
  }
  ~DiagnosticRange() {
#if ET_RUNTIME_KV_PROBE_HAS_NVTX
    nvtxRangePop();
#endif
  }
  DiagnosticRange(const DiagnosticRange&) = delete;
  DiagnosticRange& operator=(const DiagnosticRange&) = delete;
};

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Json snapshot(cb::CudaExecutor& executor) {
  require(cudaDeviceSynchronize() == cudaSuccess, "snapshot synchronization");
  size_t free_bytes = 0, total_bytes = 0;
  require(
      cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess,
      "device memory");
  int device = 0;
  require(cudaGetDevice(&device) == cudaSuccess, "device ordinal");
  const auto kv = executor.kv_metrics();
  Json result = {
      {"pid", getpid()},
      {"physical_columns", kv.physical_columns},
      {"declared_columns", kv.declared_columns},
      {"live_sequences", kv.live_sequences},
      {"logical_length", kv.logical_length},
      {"flat_capacity", kv.flat_capacity},
      {"live_KV_bytes", kv.allocated_bytes},
      {"row_growth_count", kv.growth_count},
      {"column_resize_count", kv.column_resize_count},
      {"device_used_bytes", total_bytes - free_bytes},
  };
  const auto pool =
      executorch::extension::cuda::CudaAllocator::pool_for_device(device);
  if (pool != nullptr) {
    uint64_t used = 0, reserved = 0, used_high = 0, reserved_high = 0;
    require(
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used) ==
                cudaSuccess &&
            cudaMemPoolGetAttribute(
                pool, cudaMemPoolAttrReservedMemCurrent, &reserved) ==
                cudaSuccess,
        "allocator memory attributes");
    require(
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_high) ==
                cudaSuccess &&
            cudaMemPoolGetAttribute(
                pool, cudaMemPoolAttrReservedMemHigh, &reserved_high) ==
                cudaSuccess,
        "allocator high water attributes");
    result["allocator_used_bytes"] = used;
    result["allocator_reserved_bytes"] = reserved;
    result["allocator_used_high_bytes"] = used_high;
    result["allocator_reserved_high_bytes"] = reserved_high;
  }
  result["method_calls"] = Json::array();
  for (const auto& method : executor.method_calls()) {
    const auto graph = executor.graph_metrics(method.name);
    result["method_calls"].push_back(
        {{"method", method.name},
         {"calls", method.calls},
         {"captures", graph.captures},
         {"replays", graph.replays},
         {"capture_failures", graph.capture_failures}});
  }
  return result;
}

Json run_step(cb::CudaExecutor& executor, const Json& request) {
  cb::RawTargetStep step;
  step.tokens = request.at("tokens").get<std::vector<int64_t>>();
  step.seq_ids = request.at("seq_ids").get<std::vector<int32_t>>();
  require(
      !step.tokens.empty() && step.tokens.size() == step.seq_ids.size(),
      "token and sequence widths");
  auto& control = executor.cache_control();
  std::map<int32_t, int64_t> next;
  for (const auto id : step.seq_ids) {
    const auto inserted = next.emplace(id, control.pos(id));
    step.positions.push_back(inserted.first->second++);
  }
  step.logit_indices = request.at("select").get<std::vector<int64_t>>();
  std::vector<int64_t> argmax;
  Json outputs = Json::array();
  const bool save_logits = request.value("save_logits", false);
  const std::string tag = request.value("tag", std::string("step"));
  require(
      !tag.empty() && tag.find("..") == std::string::npos &&
          tag.find_first_not_of(
              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.") ==
              std::string::npos,
      "diagnostic tag");
  size_t part = 0;
  const bool ok =
      executor.execute_raw(step, [&](const cb::RawTargetOutput& out) {
        if (request.value("check_reentrancy", false)) {
          require(
              !executor.execute_raw(
                  step, [](const cb::RawTargetOutput&) { return true; }),
              "nested execute must be rejected");
        }
        const size_t rows = out.selected_indices.size();
        if (rows == 0) {
          return true;
        }
        if (save_logits) {
          const auto& logits = out.logits;
          require(
              logits.dim() == 2 &&
                  logits.scalar_type() == executorch::aten::ScalarType::Float &&
                  logits.strides()[1] == 1,
              "expected dense-row FP32 selected logits");
          const size_t vocab = executor.vocab_size();
          std::vector<float> host(rows * vocab);
          if (logits.device_type() == executorch::aten::DeviceType::CUDA) {
            require(
                cudaDeviceSynchronize() == cudaSuccess,
                "logits synchronization");
            require(
                cudaMemcpy2D(
                    host.data(),
                    vocab * sizeof(float),
                    logits.const_data_ptr<float>(),
                    logits.strides()[0] * sizeof(float),
                    vocab * sizeof(float),
                    rows,
                    cudaMemcpyDeviceToHost) == cudaSuccess,
                "logits device copy");
          } else {
            for (size_t row = 0; row < rows; ++row) {
              std::copy_n(
                  logits.const_data_ptr<float>() + row * logits.strides()[0],
                  vocab,
                  host.data() + row * vocab);
            }
          }
          const auto path = std::filesystem::path(FLAGS_output_dir) /
              (tag + "-" + std::to_string(part++) + ".f32");
          require(!std::filesystem::exists(path), "preserve prior logits");
          std::ofstream file(path, std::ios::binary);
          file.write(
              reinterpret_cast<const char*>(host.data()),
              static_cast<std::streamsize>(host.size() * sizeof(float)));
          require(file.good(), "write logits");
          outputs.push_back(
              {{"file", path.string()},
               {"shape", {rows, vocab}},
               {"selected_indices", out.selected_indices}});
        }
        if (request.value("argmax", true)) {
          const auto sampled = executor.argmax_rows(out.logits, rows);
          require(sampled.has_value(), "device argmax");
          argmax.insert(argmax.end(), sampled->begin(), sampled->end());
        }
        return true;
      });
  require(ok, "actual executor step failed; do not continue owners");
  return {{"argmax", argmax}, {"logits", outputs}};
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  require(
      !FLAGS_pte.empty() && std::filesystem::is_directory(FLAGS_output_dir),
      "model and unique output directory required");
  const int protocol_fd = dup(STDOUT_FILENO);
  require(protocol_fd >= 0, "duplicate stdout");
  require(dup2(STDERR_FILENO, STDOUT_FILENO) >= 0, "route model diagnostics");
  std::FILE* protocol = fdopen(protocol_fd, "w");
  require(protocol != nullptr, "protocol output");
  try {
    cb::CudaExecutorConfig config;
    config.model_path = FLAGS_pte;
    if (!FLAGS_data_path.empty()) {
      config.data_files.push_back(FLAGS_data_path);
    }
    config.max_sessions = FLAGS_max_sessions;
    config.max_session_tokens = FLAGS_max_context;
    config.initial_capacity = FLAGS_initial_kv_capacity;
    config.options.cuda_graph_for_static_methods = true;
    auto created = cb::CudaExecutor::create(config);
    require(created.ok(), "create actual CUDA executor");
    auto executor = std::move(*created);
    require(executor->initialize(), "initialize actual CUDA executor");
    std::fprintf(
        protocol,
        "%s\n",
        Json({{"ready", true}, {"snapshot", snapshot(*executor)}})
            .dump()
            .c_str());
    std::fflush(protocol);
    std::string line;
    while (std::getline(std::cin, line)) {
      const auto request = Json::parse(line);
      const std::string op = request.at("op");
      const DiagnosticRange range(request.value("tag", op));
      Json result;
      const auto start = std::chrono::steady_clock::now();
      auto& control = executor->cache_control();
      if (op == "new") {
        const auto id = control.seq_new();
        require(id.has_value(), "new owner");
        result["seq_id"] = *id;
      } else if (op == "remove") {
        require(
            control.seq_rm(request.at("seq_id").get<int32_t>()),
            "remove owner");
      } else if (op == "clone") {
        const auto id = control.seq_clone(
            request.at("seq_id").get<int32_t>(), request.at("upto").get<int>());
        require(id.has_value(), "clone owner");
        result["seq_id"] = *id;
      } else if (op == "step") {
        result = run_step(*executor, request);
      } else if (op != "metrics") {
        throw std::runtime_error("unknown diagnostic operation");
      }
      require(cudaDeviceSynchronize() == cudaSuccess, "operation completion");
      result["elapsed_seconds"] = std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
      result["snapshot"] = snapshot(*executor);
      result["op"] = op;
      std::fprintf(protocol, "%s\n", result.dump().c_str());
      std::fflush(protocol);
    }
  } catch (const std::exception& error) {
    std::fprintf(
        protocol,
        "%s\n",
        Json({{"failed", true}, {"detail", error.what()}}).dump().c_str());
    std::fflush(protocol);
    std::fclose(protocol);
    return 1;
  }
  std::fclose(protocol);
  return 0;
}
