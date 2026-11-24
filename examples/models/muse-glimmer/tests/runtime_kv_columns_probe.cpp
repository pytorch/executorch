/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A serialized correctness transport over the actual CUDA batching executor.
// Raw cache IDs are never mixed with ServingRuntime or SessionTable IDs.
#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/runner.h>
#include <gflags/gflags.h>
#include <nlohmann/json.hpp>
#include <unistd.h>
#include "../../../../.ci/scripts/muse_glimmer_batching_workload.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
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
    8,
    "Logical cache admission including snapshot owners");
DEFINE_int32(max_context, 64, "Logical model context");
DEFINE_int32(initial_kv_capacity, 4, "Initial logical rows per column");

namespace cb = executorch::backends::cuda::batching;
namespace batching = executorch::extension::llm::batching;
using executorch::runtime::Error;
using Json = nlohmann::json;

DEFINE_int64(
    kv_compaction_delay_ms,
    500,
    "Quiet interval before column compaction; -1 disables automatic compaction");

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Json snapshot(cb::CudaExecutor& executor) {
  require(cudaDeviceSynchronize() == cudaSuccess, "snapshot synchronization");
  const auto kv = executor.kv_metrics();
  Json result = {
      {"physical_columns", kv.physical_columns},
      {"declared_columns", kv.declared_columns},
      {"live_sequences", kv.live_sequences},
      {"live_KV_bytes", kv.allocated_bytes},
      {"column_resize_count", kv.column_resize_count},
  };
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
        return true;
      });
  require(ok, "actual executor step failed; do not continue owners");
  return {{"logits", outputs}};
}

// Read cache metrics only on the engine thread, after the actual CI decorator
// forwards idle maintenance. The caller observes a synchronized test snapshot.
class RunnerObserver final : public batching::Executor {
 public:
  explicit RunnerObserver(cb::CudaExecutor& executor)
      : target_(executor), evidence_(executor) {}
  bool initialize() override {
    return evidence_.initialize();
  }
  std::size_t preferred_batch_tokens() const override {
    return evidence_.preferred_batch_tokens();
  }
  std::optional<batching::SessionId> open_session() override {
    return evidence_.open_session();
  }
  void close_session(batching::SessionId id) override {
    evidence_.close_session(id);
  }
  void set_sampling(
      batching::SessionId id,
      const batching::SamplingParams& params,
      std::optional<uint64_t> seed) override {
    evidence_.set_sampling(id, params, seed);
  }
  bool execute(const batching::BatchInput& input, batching::BatchOutput& out)
      override {
    return evidence_.execute(input, out);
  }
  bool on_idle() override {
    const bool ok = evidence_.on_idle();
    const auto metrics = target_.kv_metrics();
    std::lock_guard<std::mutex> lock(mutex_);
    columns_ = metrics.physical_columns;
    owners_ = metrics.live_sequences;
    ++idle_calls_;
    return ok;
  }
  bool wait_for(int columns, int owners) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (columns_ == columns && owners_ == owners && idle_calls_ > 0) {
          return true;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
  }

 private:
  cb::CudaExecutor& target_;
  muse_glimmer_ci::EvidenceExecutor evidence_;
  std::mutex mutex_;
  int64_t columns_ = 0;
  int64_t owners_ = 0;
  int idle_calls_ = 0;
};

Json check_runner(cb::CudaExecutor& executor) {
  require(executor.kv_metrics().live_sequences == 0, "fresh Runner cache");
  RunnerObserver observer(executor);
  batching::Runner runner(
      observer, batching::DecodeFirstScheduler::create(32, 8, 16));
  std::vector<batching::Session> sessions;
  auto open = [&] {
    auto future = runner.open_session_async();
    require(
        future.wait_for(std::chrono::seconds(10)) == std::future_status::ready,
        "Runner admission timeout");
    auto session = future.get();
    require(session.has_value(), "Runner admission failed");
    sessions.push_back(std::move(*session));
  };
  for (int i = 0; i < 8; ++i) {
    open();
  }
  Json output = Json::array();
  auto generate = [&](const std::vector<size_t>& indices, int salt) {
    auto tokens = std::make_shared<std::vector<std::vector<batching::Token>>>(
        indices.size());
    std::vector<batching::GenerationHandle> handles;
    batching::GenConfig config;
    config.max_new_tokens = 3;
    config.seed = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      handles.push_back(sessions[indices[i]].generate_async(
          std::vector<batching::Token>{
              1, 2, 3, static_cast<batching::Token>(salt)},
          config,
          [tokens, i](const batching::GenerationUpdate& update) {
            (*tokens)[i].insert(
                (*tokens)[i].end(), update.tokens.begin(), update.tokens.end());
          }));
    }
    for (auto& handle : handles) {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(10);
      while (!handle.done() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      require(handle.done(), "Runner generation timeout");
      handle.wait();
      require(
          handle.finish_reason() == batching::FinishReason::NewTokenLimit,
          "Runner generation failed");
    }
    for (const auto& row : *tokens) {
      require(row.size() == 3, "Runner token budget");
    }
    output.push_back(*tokens);
  };
  generate({0, 1, 2, 3, 4, 5, 6, 7}, 4);
  require(observer.wait_for(9, 8), "Runner burst physical width");
  const auto survivor_position = sessions[7].position();
  for (size_t i = 0; i < 7; ++i) {
    sessions[i] = batching::Session{};
  }
  const int idle_columns = FLAGS_kv_compaction_delay_ms == -1 ? 9 : 3;
  require(
      observer.wait_for(idle_columns, 1),
      "Runner quiet compaction through CI wrapper");
  require(
      sessions[7].position() == survivor_position,
      "inactive open history retained");
  generate({7}, 5);
  for (int i = 0; i < 7; ++i) {
    open();
  }
  generate({7, 8, 9, 10, 11, 12, 13, 14}, 6);
  require(observer.wait_for(9, 8), "Runner regrowth physical width");
  runner.shutdown();
  return {{"outputs", output}, {"idle_columns", idle_columns}};
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
    config.kv_column_compaction_delay = FLAGS_kv_compaction_delay_ms == -1
        ? std::nullopt
        : std::make_optional(
              std::chrono::milliseconds(FLAGS_kv_compaction_delay_ms));
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
      Json result;
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
      } else if (op == "compact") {
        require(
            executor->compact_kv_columns(request.value("spare_columns", 0)) ==
                executorch::runtime::Error::Ok,
            "compact physical columns");
      } else if (op == "wait") {
        const int milliseconds = request.at("milliseconds").get<int>();
        require(
            milliseconds >= 0 && milliseconds <= 1000,
            "bounded diagnostic wait");
        std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
      } else if (op == "idle") {
        result["maintained"] = executor->on_idle();
        require(result["maintained"].get<bool>(), "idle maintenance");
      } else if (op == "runner") {
        result = check_runner(*executor);
      } else if (op == "step") {
        result = run_step(*executor, request);
      } else if (op != "metrics") {
        throw std::runtime_error("unknown diagnostic operation");
      }
      require(cudaDeviceSynchronize() == cudaSuccess, "operation completion");
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
