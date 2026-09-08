/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Sample application demonstrating continuous batching and streaming output
// for independent prompts submitted by a bounded set of worker threads.
//
// Required flags are --pte and --tokenizer. Each remaining positional argument
// is a prompt. --n_workers controls concurrent submission, and generated text is
// streamed to <out_prefix>_<prompt-index>.txt.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/module_executor.h>
#include <executorch/extension/llm/batching/runner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/text_stream.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

DEFINE_string(pte, "", "Path to the .pte exported with --use-offgraph-cache");
DEFINE_string(tokenizer, "", "Path to a supported tokenizer file");
DEFINE_string(out_prefix, "gen", "Output files are <prefix>_<n>.txt");
DEFINE_int32(max_session_tokens, 2048, "Maximum tokens retained per session");
DEFINE_string(
    kv_storage_dtype,
    "bf16",
    "KV storage dtype: bf16, fp16, or fp32");
DEFINE_int32(
    kv_initial_capacity,
    -1,
    "Initial cache pool capacity; -1 keeps the cache default");
DEFINE_int32(max_new_tokens, 128, "Maximum generated tokens per prompt");
DEFINE_int32(n_workers, 4, "Number of prompt workers");
DEFINE_int32(flush_every, 8, "Flush each output file every N generated tokens");
DEFINE_int32(
    max_decode_sequences,
    32,
    "Maximum decode sequences admitted to one batch");
DEFINE_double(temperature, 0.0, "Sampling temperature; 0 is greedy");
DEFINE_double(top_p, 1.0, "Nucleus sampling probability");
DEFINE_int32(top_k, 0, "Top-k sampling limit; 0 disables it");
DEFINE_uint64(seed, 42, "Per-generation sampling seed");
DEFINE_bool(metrics, true, "Print per-generation and engine reports");
DEFINE_string(
    chat,
    "llama3",
    "Chat template: llama3, gemma, gemma4, or 0 for raw text");

namespace batching = ::executorch::extension::llm::batching;
using ::executorch::extension::Module;
using ::executorch::extension::llm::TextStream;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace {

struct Emitter {
  Emitter(
      const tokenizers::Tokenizer& tokenizer,
      batching::Token previous,
      const std::string& path,
      std::size_t flush_every)
      : file(path, std::ios::binary),
        flush_every(flush_every),
        stream(
            tokenizer,
            [this](const std::string& piece) {
              file.write(
                  piece.data(), static_cast<std::streamsize>(piece.size()));
            },
            previous) {}

  void append(batching::Token token) {
    if (stream.append(token) != Error::Ok || !file) {
      throw std::runtime_error("failed to decode or write output");
    }
    if (++tokens_since_flush == flush_every) {
      file.flush();
      tokens_since_flush = 0;
      if (!file) {
        throw std::runtime_error("failed to flush output");
      }
    }
  }

  void finish() {
    stream.flush();
    file.flush();
    if (!file) {
      throw std::runtime_error("failed to flush output");
    }
  }

  std::ofstream file;
  const std::size_t flush_every;
  std::size_t tokens_since_flush = 0;
  TextStream stream;
};

struct WorkerResult {
  std::optional<batching::Session> session;
  batching::GenerationHandle handle;
  std::optional<batching::FinishReason> reason;
  std::optional<batching::GenerationMetrics> metrics;
  std::string message;
  std::string output_path;

  bool failed() const {
    return !reason || *reason == batching::FinishReason::Cancelled ||
        *reason == batching::FinishReason::Failed;
  }
};

const char* reason_name(const std::optional<batching::FinishReason>& reason) {
  if (!reason) {
    return "never started";
  }
  switch (*reason) {
    case batching::FinishReason::StopToken:
      return "stop token";
    case batching::FinishReason::NewTokenLimit:
      return "token limit";
    case batching::FinishReason::Cancelled:
      return "cancelled";
    case batching::FinishReason::Failed:
      return "failed";
  }
  return "unknown";
}

int storage_dtype(const std::string& name) {
  using ScalarType = ::executorch::runtime::etensor::ScalarType;
  if (name == "bf16") {
    return static_cast<int>(ScalarType::BFloat16);
  }
  if (name == "fp16") {
    return static_cast<int>(ScalarType::Half);
  }
  if (name == "fp32") {
    return static_cast<int>(ScalarType::Float);
  }
  return -1;
}

bool wrap_turn(
    const std::string& chat,
    const std::string& prompt,
    std::string& out) {
  if (chat == "0") {
    out = prompt;
  } else if (chat == "llama3") {
    out = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" +
        prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
  } else if (chat == "gemma") {
    out = "<bos><start_of_turn>user\n" + prompt +
        "<end_of_turn>\n<start_of_turn>model\n";
  } else if (chat == "gemma4") {
    out = "<bos><|turn>user\n" + prompt + "<turn|>\n<|turn>model\n";
  } else {
    return false;
  }
  return true;
}

Result<std::optional<int64_t>> optional_const_int(
    Module& module,
    const char* name) {
  const auto methods = module.method_names();
  if (!methods.ok()) {
    return methods.error();
  }
  if (methods->count(name) == 0) {
    return std::optional<int64_t>{};
  }
  const auto result = module.execute(name);
  if (!result.ok()) {
    return result.error();
  }
  if (result->size() != 1 || !result->at(0).isInt()) {
    return Error::InvalidProgram;
  }
  return std::optional<int64_t>{result->at(0).toInt()};
}

bool get_stop_tokens(
    tokenizers::Tokenizer& tokenizer,
    Module& module,
    const std::string& chat,
    std::vector<batching::Token>& out) {
  auto ids = ::executorch::extension::llm::get_eos_ids(&tokenizer, &module);
  if (chat != "0") {
    const char* turn_end = chat == "llama3" ? "<|eot_id|>"
        : chat == "gemma4"                  ? "<turn|>"
                                            : "<end_of_turn>";
    auto id = tokenizer.piece_to_id(turn_end);
    if (!id.ok()) {
      return false;
    }
    ids.insert(*id);
  }
  out.assign(ids.begin(), ids.end());
  return true;
}

void submit_prompt(
    batching::Runner& runner,
    const tokenizers::Tokenizer& tokenizer,
    const std::string& prompt,
    const std::vector<batching::Token>& stop_tokens,
    WorkerResult& result) {
  try {
    std::string wrapped;
    if (!wrap_turn(FLAGS_chat, prompt, wrapped)) {
      result.message = "unknown --chat template: " + FLAGS_chat;
      return;
    }
    auto encoded = tokenizer.encode(wrapped, FLAGS_chat == "0" ? 1 : 0, 0);
    if (!encoded.ok() || encoded->empty()) {
      result.message = "could not encode prompt";
      return;
    }
    if (encoded->size() >
        static_cast<std::size_t>(
            FLAGS_max_session_tokens - FLAGS_max_new_tokens)) {
      result.message =
          "prompt plus --max_new_tokens exceeds --max_session_tokens";
      return;
    }

    result.session = runner.open_session_async().get();
    if (!result.session) {
      result.message = "could not open session";
      return;
    }

    auto emitter = std::make_shared<Emitter>(
        tokenizer,
        encoded->back(),
        result.output_path,
        static_cast<std::size_t>(FLAGS_flush_every));
    if (!emitter->file) {
      result.message = "could not open output file";
      return;
    }

    batching::GenConfig config;
    config.max_new_tokens = FLAGS_max_new_tokens;
    config.sampling.temperature = static_cast<float>(FLAGS_temperature);
    config.sampling.top_p = static_cast<float>(FLAGS_top_p);
    config.sampling.top_k = FLAGS_top_k;
    config.stop_tokens = stop_tokens;
    config.seed = FLAGS_seed;

    result.handle = result.session->generate_async(
        std::move(*encoded),
        std::move(config),
        [emitter](const batching::GenerationUpdate& update) {
          std::size_t count = update.tokens.size();
          if (update.finish_reason == batching::FinishReason::StopToken &&
              count > 0) {
            --count;
          }
          for (std::size_t i = 0; i < count; ++i) {
            emitter->append(update.tokens[i]);
          }
          if (update.finish_reason) {
            emitter->finish();
          }
        });
  } catch (const std::exception& error) {
    result.message = error.what();
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  const std::vector<std::string> prompts(argv + 1, argv + argc);

  if (FLAGS_pte.empty() || FLAGS_tokenizer.empty() || prompts.empty()) {
    std::cerr << "usage: " << argv[0]
              << " --pte model.pte --tokenizer tokenizer-file \"prompt\" [...]"
              << std::endl;
    return 1;
  }
  if (FLAGS_max_session_tokens <= 0 || FLAGS_max_new_tokens <= 0 ||
      FLAGS_max_decode_sequences <= 0 || FLAGS_n_workers <= 0 ||
      FLAGS_flush_every <= 0) {
    std::cerr
        << "session, generation, decode, worker, and flush limits must be "
           "positive"
        << std::endl;
    return 1;
  }
  if (FLAGS_temperature < 0.0 || FLAGS_top_p <= 0.0 || FLAGS_top_p > 1.0 ||
      FLAGS_top_k < 0) {
    std::cerr << "invalid sampling parameters" << std::endl;
    return 1;
  }
  if (FLAGS_kv_initial_capacity < -1) {
    std::cerr << "--kv_initial_capacity must be -1 or non-negative"
              << std::endl;
    return 1;
  }
  if (prompts.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    std::cerr << "too many prompts" << std::endl;
    return 1;
  }
  const int kv_dtype = storage_dtype(FLAGS_kv_storage_dtype);
  if (kv_dtype < 0) {
    std::cerr << "--kv_storage_dtype must be bf16, fp16, or fp32" << std::endl;
    return 1;
  }
  if (FLAGS_max_new_tokens > FLAGS_max_session_tokens) {
    std::cerr << "--max_new_tokens exceeds --max_session_tokens" << std::endl;
    return 1;
  }

  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(FLAGS_tokenizer);
  if (!tokenizer) {
    std::cerr << "could not load tokenizer: " << FLAGS_tokenizer << std::endl;
    return 1;
  }

  auto module = std::make_unique<Module>(FLAGS_pte);
  if (module->load() != Error::Ok) {
    std::cerr << "could not load " << FLAGS_pte << std::endl;
    return 1;
  }

  const auto model_max_context = optional_const_int(*module, "get_max_ctx_len");
  if (!model_max_context.ok()) {
    std::cerr << "could not read get_max_ctx_len" << std::endl;
    return 1;
  }
  if (*model_max_context &&
      (**model_max_context <= 0 ||
       FLAGS_max_session_tokens > **model_max_context)) {
    std::cerr << "--max_session_tokens " << FLAGS_max_session_tokens
              << " exceeds the model context limit " << **model_max_context
              << std::endl;
    return 1;
  }

  std::vector<batching::Token> stop_tokens;
  if (!get_stop_tokens(*tokenizer, *module, FLAGS_chat, stop_tokens)) {
    std::cerr << "tokenizer has no turn-end token for --chat=" << FLAGS_chat
              << std::endl;
    return 1;
  }

  const std::size_t worker_count =
      std::min(prompts.size(), static_cast<std::size_t>(FLAGS_n_workers));
  auto executor = batching::ModuleExecutor::create(
      std::move(module),
      static_cast<int>(prompts.size()),
      FLAGS_max_session_tokens,
      kv_dtype,
      FLAGS_kv_initial_capacity);
  if (!executor.ok()) {
    std::cerr << "could not create executor: "
              << ::executorch::runtime::to_string(executor.error())
              << std::endl;
    return 1;
  }

  const std::size_t width = (*executor)->preferred_batch_tokens();
  const std::size_t decode_slots =
      static_cast<std::size_t>(FLAGS_max_decode_sequences);
  if (width == 0 || decode_slots >= width) {
    std::cerr << "--max_decode_sequences " << decode_slots
              << " leaves no room for prefill in a " << width
              << "-token forward" << std::endl;
    return 1;
  }
  if (decode_slots > width / 4) {
    std::cerr << "warning: --max_decode_sequences " << decode_slots
              << " leaves only " << width - decode_slots
              << " prefill tokens of a " << width << "-token forward"
              << std::endl;
  }

  auto scheduler = batching::DecodeFirstScheduler::create(
      width, decode_slots, width - decode_slots);
  if (!scheduler) {
    std::cerr << "the scheduler refused those limits" << std::endl;
    return 1;
  }

  batching::Runner runner(**executor, std::move(scheduler));
  std::vector<WorkerResult> results(prompts.size());
  for (std::size_t i = 0; i < results.size(); ++i) {
    results[i].output_path =
        FLAGS_out_prefix + "_" + std::to_string(i) + ".txt";
    std::ofstream output(
        results[i].output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
      runner.shutdown();
      std::cerr << "could not create " << results[i].output_path << std::endl;
      return 1;
    }
  }

  std::atomic<std::size_t> next_prompt{0};
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  try {
    for (std::size_t i = 0; i < worker_count; ++i) {
      workers.emplace_back([&] {
        while (true) {
          const std::size_t prompt_index =
              next_prompt.fetch_add(1, std::memory_order_relaxed);
          if (prompt_index >= prompts.size()) {
            return;
          }
          submit_prompt(
              runner,
              *tokenizer,
              prompts[prompt_index],
              stop_tokens,
              results[prompt_index]);
        }
      });
    }
  } catch (const std::exception& error) {
    for (auto& worker : workers) {
      worker.join();
    }
    runner.shutdown();
    std::cerr << "could not start worker: " << error.what() << std::endl;
    return 1;
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (WorkerResult& result : results) {
    if (!result.handle.valid()) {
      continue;
    }
    result.handle.wait();
    result.metrics = result.handle.metrics();
    result.reason = result.handle.finish_reason();
    result.message = result.handle.error_message();
  }

  runner.shutdown();
  const batching::EngineMetrics engine = runner.metrics();

  if (FLAGS_metrics) {
    std::cout << "\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
      if (results[i].metrics) {
        std::cout << "[" << i << "] "
                  << batching::format_report(*results[i].metrics);
      }
    }
    std::cout << "\n" << batching::format_report(engine);
  }

  std::size_t failures = 0;
  for (const WorkerResult& result : results) {
    failures += result.failed() ? 1 : 0;
  }
  if (failures > 0) {
    std::cout << "\nfailures:\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
      if (!results[i].failed()) {
        continue;
      }
      std::cout << "  [" << i << "] " << results[i].output_path << ": "
                << reason_name(results[i].reason);
      if (!results[i].message.empty()) {
        std::cout << ": " << results[i].message;
      }
      std::cout << "\n";
    }
  }

  std::cout << prompts.size() - failures << "/" << prompts.size()
            << " generations completed" << std::endl;
  return failures == 0 ? 0 : 1;
}
