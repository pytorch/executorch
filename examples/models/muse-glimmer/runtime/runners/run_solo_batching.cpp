/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Batched Muse Glimmer text generation on CUDA, on the batching extension:
// CudaExecutor runs the artifact export_solo_batching.py writes, the batching
// Runner carries every prompt's generation, and DecodeFirstScheduler packs
// their decodes and prefill chunks into shared forwards.
//
// Each prompt is one session. Generated text is printed per prompt, and
// --report_json writes what CI checks: each generation, GPU memory at each
// stage, and the KV pool's usage.

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <nlohmann/json.hpp>

#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/runner.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

DEFINE_string(model_path, "", "The .pte export_solo_batching.py wrote.");
DEFINE_string(data_path, "", "Its aoti_cuda_blob.ptd.");
DEFINE_string(tokenizer_path, "", "tokenizer.json.");
DEFINE_string(
    prompt_file,
    "",
    "A file holding one already-formatted prompt, verbatim.");
DEFINE_string(
    prompts_file,
    "",
    "A file holding one already-formatted prompt per line; \\n in a line is "
    "a newline. Combined with --prompt_file, that prompt comes first.");
DEFINE_string(
    prompt_tokens_file,
    "",
    "Instead of text prompts: one prompt per line as space-separated token "
    "ids, BOS included. --tokenizer_path is then optional.");
DEFINE_int32(max_new_tokens, 512, "Maximum tokens generated per prompt.");
DEFINE_double(temperature, 0.0, "Sampling temperature; 0 is greedy.");
DEFINE_double(top_p, 1.0, "Nucleus sampling probability.");
DEFINE_int32(top_k, 0, "Top-k sampling limit; 0 disables it.");
DEFINE_uint64(seed, 42, "Per-generation sampling seed.");
DEFINE_int32(max_sessions, 4, "Resident sessions the KV pool reserves for.");
DEFINE_int32(
    max_session_tokens,
    4096,
    "Tokens one session may hold: its prompt plus what it generates.");
DEFINE_int32(
    kv_initial_capacity,
    -1,
    "Initial KV pool rows; -1 keeps the cache's default. It grows on demand.");
DEFINE_int32(
    max_decode_sequences,
    0,
    "Decodes admitted to one forward; 0 admits one per session.");
DEFINE_bool(cuda_graph, true, "Capture decode into a CUDA graph.");
DEFINE_bool(
    weight_sharing,
    true,
    "Share one copy of the weights between decode and prefill.");
DEFINE_int32(bos_id, 200000, "BOS token id.");
DEFINE_int32(eos_id, 200001, "EOS token id.");
DEFINE_string(report_json, "", "Write a JSON report here.");

namespace batching = ::executorch::extension::llm::batching;
namespace cuda_batching = ::executorch::backends::cuda::batching;
namespace metadata = ::executorch::extension::llm;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;

namespace {

constexpr int kBFloat16 = 15; // ScalarType::BFloat16

struct GpuMemory {
  std::size_t used = 0;
  std::size_t total = 0;
};

GpuMemory gpu_memory() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  return {total_bytes - free_bytes, total_bytes};
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return {};
  }
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

std::string unescape_newlines(const std::string& line) {
  std::string out;
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == 'n') {
      out.push_back('\n');
      ++i;
    } else {
      out.push_back(line[i]);
    }
  }
  return out;
}

std::vector<std::vector<batching::Token>> load_prompt_tokens() {
  std::vector<std::vector<batching::Token>> prompts;
  std::ifstream in(FLAGS_prompt_tokens_file);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ids(line);
    std::vector<batching::Token> tokens;
    batching::Token token;
    while (ids >> token) {
      tokens.push_back(token);
    }
    if (!tokens.empty()) {
      prompts.push_back(std::move(tokens));
    }
  }
  return prompts;
}

std::vector<std::string> load_prompts() {
  std::vector<std::string> prompts;
  if (!FLAGS_prompt_file.empty()) {
    prompts.push_back(read_file(FLAGS_prompt_file));
  }
  if (!FLAGS_prompts_file.empty()) {
    std::ifstream in(FLAGS_prompts_file);
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty()) {
        prompts.push_back(unescape_newlines(line));
      }
    }
  }
  return prompts;
}

// Turn-ending tokens: the model's EOS plus Harmony's <|eot|> and
// <|end_of_text|>. <|eom|> continues the assistant turn, so it is not one.
std::vector<batching::Token> stop_tokens(::tokenizers::Tokenizer& tokenizer) {
  std::unordered_set<batching::Token> ids{
      static_cast<batching::Token>(FLAGS_eos_id)};
  for (const char* piece : {"<|eot|>", "<|end_of_text|>"}) {
    if (auto id = tokenizer.piece_to_id(piece); id.ok()) {
      ids.insert(static_cast<batching::Token>(*id));
    }
  }
  return {ids.begin(), ids.end()};
}

// K and V, every layer, one token.
std::int64_t kv_bytes_per_cell(const metadata::cache::CacheGeometry& geometry) {
  std::int64_t bytes = 0;
  for (const auto& layer : geometry.layers) {
    bytes += 2 * static_cast<std::int64_t>(layer.n_kv_heads) * layer.head_dim *
        2; // bf16
  }
  return bytes;
}

const char* reason_name(const std::optional<batching::FinishReason>& reason) {
  if (!reason) {
    return "never started";
  }
  switch (*reason) {
    case batching::FinishReason::StopToken:
      return "stop_token";
    case batching::FinishReason::NewTokenLimit:
      return "token_limit";
    case batching::FinishReason::Cancelled:
      return "cancelled";
    case batching::FinishReason::Failed:
      return "failed";
  }
  return "unknown";
}

struct Job {
  std::string prompt;
  std::vector<batching::Token> prompt_tokens;
  std::vector<batching::Token> generated;
  std::optional<batching::Session> session;
  batching::GenerationHandle handle;
  std::optional<batching::FinishReason> reason;
  std::string error;
  std::string text;
};

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::executorch::runtime::runtime_init();
  const bool text = FLAGS_prompt_tokens_file.empty();
  if (FLAGS_model_path.empty() || (text && FLAGS_tokenizer_path.empty())) {
    std::cerr << "--model_path and --tokenizer_path are required" << std::endl;
    return 1;
  }
  std::vector<Job> jobs;
  ::tokenizers::HFTokenizer tokenizer;
  const bool have_tokenizer = !FLAGS_tokenizer_path.empty();
  if (have_tokenizer &&
      tokenizer.load(FLAGS_tokenizer_path) != ::tokenizers::Error::Ok) {
    std::cerr << "could not load " << FLAGS_tokenizer_path << std::endl;
    return 1;
  }
  if (text) {
    for (auto& prompt : load_prompts()) {
      Job job{std::move(prompt)};
      auto encoded = tokenizer.encode(job.prompt, /*bos=*/0, /*eos=*/0);
      if (!encoded.ok()) {
        std::cerr << "could not encode a prompt" << std::endl;
        return 1;
      }
      job.prompt_tokens.push_back(static_cast<batching::Token>(FLAGS_bos_id));
      for (const auto token : *encoded) {
        job.prompt_tokens.push_back(static_cast<batching::Token>(token));
      }
      jobs.push_back(std::move(job));
    }
  } else {
    for (auto& tokens : load_prompt_tokens()) {
      Job job;
      job.prompt_tokens = std::move(tokens);
      jobs.push_back(std::move(job));
    }
  }
  if (jobs.empty()) {
    std::cerr << "no prompts: pass --prompt_file, --prompts_file or "
                 "--prompt_tokens_file"
              << std::endl;
    return 1;
  }
  for (const Job& job : jobs) {
    if (job.prompt_tokens.size() + FLAGS_max_new_tokens >
        static_cast<std::size_t>(FLAGS_max_session_tokens)) {
      std::cerr << "a prompt plus --max_new_tokens exceeds --max_session_tokens"
                << std::endl;
      return 1;
    }
  }
  const auto stops = have_tokenizer
      ? stop_tokens(tokenizer)
      : std::vector<batching::Token>{static_cast<batching::Token>(FLAGS_eos_id)};

  // Create the CUDA context first, so its cost is not counted as the model's.
  cudaFree(nullptr);
  const GpuMemory before_load = gpu_memory();
  std::vector<std::string> data_files;
  if (!FLAGS_data_path.empty()) {
    data_files.push_back(FLAGS_data_path);
  }
  auto module = std::make_unique<Module>(
      FLAGS_model_path,
      data_files,
      Module::LoadMode::MmapUseMlockIgnoreErrors,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/false);
  if (module->load() != Error::Ok) {
    std::cerr << "could not load " << FLAGS_model_path << std::endl;
    return 1;
  }
  const auto geometry = metadata::read_cache_geometry(*module);
  if (!geometry.ok()) {
    std::cerr << "the program publishes no KV cache geometry" << std::endl;
    return 1;
  }
  const std::int64_t cell_bytes = kv_bytes_per_cell(*geometry);

  cuda_batching::CudaExecutorOptions options;
  options.cuda_graph_for_decode = FLAGS_cuda_graph;
  options.weight_sharing_across_methods = FLAGS_weight_sharing;
  auto executor = cuda_batching::CudaExecutor::create(
      std::move(module),
      FLAGS_max_sessions,
      FLAGS_max_session_tokens,
      kBFloat16,
      FLAGS_kv_initial_capacity,
      options);
  if (!executor.ok()) {
    std::cerr << "CudaExecutor::create failed: 0x" << std::hex
              << static_cast<int>(executor.error()) << std::endl;
    return 1;
  }

  const std::size_t width = (*executor)->preferred_batch_tokens();
  const std::size_t decode_slots = FLAGS_max_decode_sequences > 0
      ? static_cast<std::size_t>(FLAGS_max_decode_sequences)
      : static_cast<std::size_t>(FLAGS_max_sessions);
  if (decode_slots >= width) {
    std::cerr << "--max_decode_sequences leaves no room for prefill in a "
              << width << "-token forward" << std::endl;
    return 1;
  }
  auto scheduler = batching::DecodeFirstScheduler::create(
      width, decode_slots, width - decode_slots);
  batching::Runner runner(**executor, std::move(scheduler));

  // Opening a session runs the engine's first step, which loads both methods.
  for (Job& job : jobs) {
    job.session = runner.open_session_async().get();
    if (!job.session) {
      std::cerr << "could not open a session; raise --max_sessions"
                << std::endl;
      runner.shutdown();
      return 1;
    }
  }
  const GpuMemory after_load = gpu_memory();

  std::mutex mutex;
  const auto generate_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    batching::GenConfig config;
    config.max_new_tokens = FLAGS_max_new_tokens;
    config.sampling.temperature = static_cast<float>(FLAGS_temperature);
    config.sampling.top_p = static_cast<float>(FLAGS_top_p);
    config.sampling.top_k = FLAGS_top_k;
    config.seed = FLAGS_seed;
    config.stop_tokens = stops;
    Job& job = jobs[i];
    job.handle = job.session->generate_async(
        job.prompt_tokens,
        config,
        [&mutex, &job](const batching::GenerationUpdate& update) {
          std::lock_guard<std::mutex> guard(mutex);
          job.generated.insert(
              job.generated.end(), update.tokens.begin(), update.tokens.end());
        });
  }
  for (Job& job : jobs) {
    job.handle.wait();
    job.reason = job.handle.finish_reason();
    job.error = job.handle.error_message();
  }
  const double generate_seconds = std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - generate_start)
                                      .count();
  const GpuMemory after_generate = gpu_memory();
  const ::executorch::backends::cuda::OffGraphKVMetrics kv =
      (*executor)->kv_metrics();
  for (Job& job : jobs) {
    job.session.reset();
  }
  runner.shutdown();
  const batching::EngineMetrics engine = runner.metrics();

  bool ok = true;
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    Job& job = jobs[i];
    std::uint64_t previous = job.prompt_tokens.back();
    for (const auto token : job.generated) {
      if (!have_tokenizer) {
        break;
      }
      if (auto piece = tokenizer.decode(previous, token); piece.ok()) {
        job.text += *piece;
      }
      previous = token;
    }
    std::printf(
        "=== [%zu] %zu prompt tokens, %zu generated, %s ===\n%s\n",
        i,
        job.prompt_tokens.size(),
        job.generated.size(),
        reason_name(job.reason),
        job.text.c_str());
    if (!job.reason || *job.reason == batching::FinishReason::Failed ||
        *job.reason == batching::FinishReason::Cancelled) {
      std::fprintf(stderr, "[%zu] failed: %s\n", i, job.error.c_str());
      ok = false;
    }
  }
  const double mib = 1024.0 * 1024.0;
  std::size_t generated_total = 0;
  for (const Job& job : jobs) {
    generated_total += job.generated.size();
  }
  std::printf(
      "Generate: %zu tokens across %zu sessions in %.2f s (%.1f tok/s)\n",
      generated_total,
      jobs.size(),
      generate_seconds,
      generate_seconds > 0 ? generated_total / generate_seconds : 0.0);
  std::printf(
      "Engine: %" PRIu64 " steps, %" PRIu64 " decode sessions, %" PRIu64
      " prefill sessions, %" PRIu64 " decode tokens, %" PRIu64
      " prefill tokens\n",
      engine.steps,
      engine.decode_sessions_total,
      engine.prefill_sessions_total,
      engine.decode_tokens_total,
      engine.prefill_tokens_total);
  std::printf(
      "KV pool: %" PRId64 " rows allocated, %" PRId64 " cells in use, %.1f MiB, "
      "%" PRId64 " growths\n",
      kv.flat_capacity,
      kv.logical_length,
      kv.allocated_bytes / mib,
      kv.growth_count);
  std::printf(
      "GPU memory: %.1f MiB before load, %.1f MiB after load, %.1f MiB after "
      "generate\n",
      before_load.used / mib,
      after_load.used / mib,
      after_generate.used / mib);
  std::printf("GPU peak memory usage: %.1f MiB\n", after_generate.used / mib);

  if (!FLAGS_report_json.empty()) {
    nlohmann::json report;
    for (const Job& job : jobs) {
      report["generations"].push_back(
          {{"prompt", job.prompt},
           {"prompt_tokens", job.prompt_tokens.size()},
           {"tokens", job.generated},
           {"text", job.text},
           {"finish_reason", reason_name(job.reason)},
           {"error", job.error}});
    }
    report["gpu"] = {
        {"total_bytes", before_load.total},
        {"used_before_load_bytes", before_load.used},
        {"used_after_load_bytes", after_load.used},
        {"used_after_generate_bytes", after_generate.used}};
    report["kv"] = {
        {"rows", kv.flat_capacity},
        {"cells_in_use", kv.logical_length},
        {"allocated_bytes", kv.allocated_bytes},
        {"growth_count", kv.growth_count},
        {"bytes_per_cell", cell_bytes},
        {"initial_capacity", FLAGS_kv_initial_capacity}};
    std::error_code size_error;
    report["weights_bytes"] = FLAGS_data_path.empty()
        ? 0
        : std::filesystem::file_size(FLAGS_data_path, size_error);
    report["config"] = {
        {"max_sessions", FLAGS_max_sessions},
        {"max_session_tokens", FLAGS_max_session_tokens},
        {"max_new_tokens", FLAGS_max_new_tokens},
        {"cuda_graph", FLAGS_cuda_graph},
        {"weight_sharing", FLAGS_weight_sharing},
        {"step_width", width}};
    report["timing"] = {
        {"generate_seconds", generate_seconds},
        {"generated_tokens", generated_total},
        {"tokens_per_second",
         generate_seconds > 0 ? generated_total / generate_seconds : 0.0}};
    report["engine"] = {
        {"steps", engine.steps},
        {"steps_failed", engine.steps_failed},
        {"decode_sessions_total", engine.decode_sessions_total},
        {"prefill_sessions_total", engine.prefill_sessions_total},
        {"decode_tokens_total", engine.decode_tokens_total},
        {"prefill_tokens_total", engine.prefill_tokens_total}};
    std::ofstream out(FLAGS_report_json);
    out << report.dump(2) << std::endl;
    if (!out) {
      std::cerr << "could not write " << FLAGS_report_json << std::endl;
      return 1;
    }
  }
  return ok ? 0 : 1;
}
