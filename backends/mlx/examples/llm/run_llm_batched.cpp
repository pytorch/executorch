/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Continuously batches independent prompts through one MLX-backed module.
//
//   mlx_run_llm_batched --pte model.pte --tokenizer tokenizer.json \
//       --out_prefix gen "first prompt" "second prompt"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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

struct PreparedPrompt {
  std::vector<batching::Token> tokens;
  std::string output_path;
};

struct Emitter {
  Emitter(const tokenizers::Tokenizer& tokenizer, batching::Token previous)
      : stream(
            tokenizer,
            [this](const std::string& piece) { text += piece; },
            previous) {}

  std::string text;
  TextStream stream;
};

struct Outcome {
  std::optional<batching::FinishReason> reason;
  std::string message;
  bool output_failed = false;

  bool failed() const {
    return output_failed || !reason ||
        *reason == batching::FinishReason::Cancelled ||
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

bool write_output(const std::string& path, const std::string& text) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  file.write(text.data(), static_cast<std::streamsize>(text.size()));
  file.flush();
  return file.good();
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);
  const std::vector<std::string> prompts(argv + 1, argv + argc);

  if (FLAGS_pte.empty() || FLAGS_tokenizer.empty() || prompts.empty()) {
    std::cerr << "usage: " << argv[0]
              << " --pte model.pte --tokenizer tokenizer-file \"prompt\" [...]"
              << std::endl;
    return 1;
  }
  if (FLAGS_max_session_tokens <= 0 || FLAGS_max_new_tokens <= 0 ||
      FLAGS_max_decode_sequences <= 0) {
    std::cerr << "session, generation, and decode limits must be positive"
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

  std::vector<PreparedPrompt> prepared;
  prepared.reserve(prompts.size());
  const std::size_t max_prompt_tokens =
      static_cast<std::size_t>(FLAGS_max_session_tokens - FLAGS_max_new_tokens);
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    std::string wrapped;
    if (!wrap_turn(FLAGS_chat, prompts[i], wrapped)) {
      std::cerr << "unknown --chat template: " << FLAGS_chat << std::endl;
      return 1;
    }
    auto encoded = tokenizer->encode(
        wrapped, /*bos=*/FLAGS_chat == "0" ? 1 : 0, /*eos=*/0);
    if (!encoded.ok() || encoded->empty()) {
      std::cerr << "could not encode prompt " << i << std::endl;
      return 1;
    }
    if (encoded->size() > max_prompt_tokens) {
      std::cerr << "prompt " << i << " plus --max_new_tokens exceeds "
                << "--max_session_tokens" << std::endl;
      return 1;
    }
    prepared.push_back(
        PreparedPrompt{
            std::move(*encoded),
            FLAGS_out_prefix + "_" + std::to_string(i) + ".txt"});
  }

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
  std::vector<batching::Session> sessions;
  sessions.reserve(prepared.size());
  for (std::size_t i = 0; i < prepared.size(); ++i) {
    auto session = runner.open_session_async().get();
    if (!session) {
      std::cerr << "could not open session " << i << std::endl;
      runner.shutdown();
      return 1;
    }
    sessions.push_back(std::move(*session));
  }

  std::vector<std::shared_ptr<Emitter>> emitters;
  std::vector<batching::GenerationHandle> handles;
  emitters.reserve(prepared.size());
  handles.reserve(prepared.size());

  for (std::size_t i = 0; i < prepared.size(); ++i) {
    auto emitter =
        std::make_shared<Emitter>(*tokenizer, prepared[i].tokens.back());
    batching::GenConfig config;
    config.max_new_tokens = FLAGS_max_new_tokens;
    config.sampling.temperature = static_cast<float>(FLAGS_temperature);
    config.sampling.top_p = static_cast<float>(FLAGS_top_p);
    config.sampling.top_k = FLAGS_top_k;
    config.stop_tokens = stop_tokens;
    config.seed = FLAGS_seed;

    handles.push_back(
        sessions[i].generate_async(
            std::move(prepared[i].tokens),
            std::move(config),
            [emitter](const batching::GenerationUpdate& update) {
              std::size_t count = update.tokens.size();
              if (update.finish_reason == batching::FinishReason::StopToken &&
                  count > 0) {
                --count;
              }
              for (std::size_t j = 0; j < count; ++j) {
                if (emitter->stream.append(update.tokens[j]) != Error::Ok) {
                  throw std::runtime_error(
                      "tokenizer failed while decoding output");
                }
              }
              if (update.finish_reason) {
                emitter->stream.flush();
              }
            }));
    emitters.push_back(std::move(emitter));
  }

  std::vector<batching::GenerationMetrics> per_generation(prepared.size());
  std::vector<Outcome> outcomes(prepared.size());
  for (std::size_t i = 0; i < handles.size(); ++i) {
    handles[i].wait();
    per_generation[i] = handles[i].metrics();
    outcomes[i].reason = handles[i].finish_reason();
    outcomes[i].message = handles[i].error_message();
  }

  runner.shutdown();
  const batching::EngineMetrics engine = runner.metrics();

  for (std::size_t i = 0; i < prepared.size(); ++i) {
    if (!write_output(prepared[i].output_path, emitters[i]->text)) {
      outcomes[i].output_failed = true;
      if (!outcomes[i].message.empty()) {
        outcomes[i].message += "; ";
      }
      outcomes[i].message += "could not write " + prepared[i].output_path;
    }
  }

  if (FLAGS_metrics) {
    std::cout << "\n";
    for (std::size_t i = 0; i < per_generation.size(); ++i) {
      std::cout << "[" << i << "] "
                << batching::format_report(per_generation[i]);
    }
    std::cout << "\n" << batching::format_report(engine);
  }

  std::size_t failures = 0;
  for (const Outcome& outcome : outcomes) {
    failures += outcome.failed() ? 1 : 0;
  }
  if (failures > 0) {
    std::cout << "\nfailures:\n";
    for (std::size_t i = 0; i < outcomes.size(); ++i) {
      if (!outcomes[i].failed()) {
        continue;
      }
      std::cout << "  [" << i << "] " << prepared[i].output_path << ": "
                << reason_name(outcomes[i].reason);
      if (!outcomes[i].message.empty()) {
        std::cout << ": " << outcomes[i].message;
      }
      std::cout << "\n";
    }
  }

  std::cout << prompts.size() - failures << "/" << prompts.size()
            << " generations completed" << std::endl;
  return failures == 0 ? 0 : 1;
}
