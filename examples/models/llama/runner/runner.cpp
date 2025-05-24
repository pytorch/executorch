/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llama/runner/runner.h>

#include <executorch/extension/llm/runner/util.h>

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>

namespace example {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace llm = ::executorch::extension::llm;

namespace {
static constexpr auto kEnableDynamicShape = "enable_dynamic_shape";
static constexpr auto kBosId = "get_bos_id";
static constexpr auto kEosIds = "get_eos_ids";
static constexpr auto kMaxSeqLen = "get_max_seq_len";
static constexpr auto kMaxContextLen = "get_max_context_len";
static constexpr auto kVocabSize = "get_vocab_size";
static constexpr auto kUseKVCache = "use_kv_cache";
static constexpr auto kUseSDPAWithKVCache = "use_sdpa_with_kv_cache";

std::unique_ptr<::tokenizers::Tokenizer> load_tokenizer(
    const std::string& tokenizer_path) {
  auto json_tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (json_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded json tokenizer");
    return json_tokenizer;
  }

  auto tiktoken_tokenizer = get_tiktoken_for_llama();
  if (tiktoken_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded TikToken tokenizer");
    return tiktoken_tokenizer;
  }

  auto bpe_tokenizer = std::make_unique<::tokenizers::Llama2cTokenizer>();
  if (bpe_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded BPE tokenizer");
    return bpe_tokenizer;
  }

  return nullptr;
}
} // namespace

std::unique_ptr<Runner> Runner::create(
    const std::string& model_path,
    const std::string& tokenizer_path,
    std::optional<const std::string> data_path,
    float temperature) {
  ET_LOG(
      Info,
      "Creating LLaMa runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());

  // Create the Module
  std::unique_ptr<Module> module;
  if (data_path.has_value()) {
    module = std::make_unique<Module>(
        model_path, data_path.value(), Module::LoadMode::File);
  } else {
    module = std::make_unique<Module>(model_path, Module::LoadMode::File);
  }

  // Initialize metadata with default values
  std::unordered_map<std::string, int64_t> metadata({
      {kEnableDynamicShape, false},
      {kMaxSeqLen, 128},
      {kMaxContextLen, 128},
      {kUseKVCache, true},
      {kUseSDPAWithKVCache, false},
  });

  // Create and load tokenizer
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer =
      load_tokenizer(tokenizer_path);

  // Fallback to BPE tokenizer if tiktoken fails
  if (tokenizer == nullptr) {
    ET_LOG(
        Info,
        "Failed to load %s as a Tiktoken, Sentencepiece or Llama2.c tokenizer, make sure the artifact is one of these types",
        tokenizer_path.c_str());
    return nullptr;
  }

  ET_LOG(Info, "Reading metadata from model");

  // Set tokenizer-related metadata
  metadata[kBosId] = tokenizer->bos_tok();
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      std::unordered_set<uint64_t>{tokenizer->eos_tok()});
  metadata[kVocabSize] = tokenizer->vocab_size();

  // Read metadata from the model
  auto method_names_result = module->method_names();
  if (method_names_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed reading method names");
    return nullptr;
  }
  const auto method_names = method_names_result.get();

  for (auto& pair : metadata) {
    const auto& method_name = pair.first;
    auto& value = pair.second;

    if (method_names.count(method_name)) {
      auto get_result = module->get(method_name);
      value = get_result.get().toScalar().to<decltype(metadata)::mapped_type>();
    } else {
      ET_LOG(
          Info,
          "Method %s not found, using the default value %" PRId64,
          method_name.c_str(),
          value);
    }
    ET_LOG(Info, "Metadata: %s = %" PRId64, method_name.c_str(), value);
  }

  // Get EOS IDs if available
  if (method_names.count(kEosIds)) {
    eos_ids->clear();
    auto execute_result = module->execute(kEosIds);
    if (execute_result.error() != Error::Ok) {
      ET_LOG(Error, "Failed to execute %s", kEosIds);
      return nullptr;
    }
    for (const auto& eos_id : execute_result.get()) {
      auto value = eos_id.toScalar().to<int64_t>();
      eos_ids->emplace(value);
      ET_LOG(Info, "eos_id = %" PRId64, value);
    }
  }

  // Create text_decoder_runner. Use a shared_ptr so that it can be shared with
  // TextPrefiller and TextTokenGenerator
  auto text_decoder_runner = std::make_unique<llm::TextDecoderRunner>(
      module.get(), metadata.at(kUseKVCache));

  // Create text_prefiller
  auto text_prefiller = std::make_unique<llm::TextPrefiller>(
      text_decoder_runner.get(),
      metadata.at(kUseKVCache),
      metadata.at(kEnableDynamicShape),
      metadata.at(kMaxSeqLen));

  // Create text_token_generator with stats
  auto stats = std::make_unique<llm::Stats>();
  auto text_token_generator = std::make_unique<llm::TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(kUseKVCache),
      std::move(eos_ids),
      stats.get());

  // Create and return the Runner instance
  return std::make_unique<Runner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::move(text_prefiller),
      std::move(text_token_generator),
      std::move(stats),
      temperature);
}

Runner::Runner(
    std::unordered_map<std::string, int64_t> metadata,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<::executorch::extension::Module> module,
    std::unique_ptr<::executorch::extension::llm::TextDecoderRunner>
        text_decoder_runner,
    std::unique_ptr<::executorch::extension::llm::TextPrefiller> text_prefiller,
    std::unique_ptr<::executorch::extension::llm::TextTokenGenerator>
        text_token_generator,
    std::unique_ptr<::executorch::extension::llm::Stats> stats,
    float temperature)
    : tokenizer_(std::move(tokenizer)),
      metadata_(std::move(metadata)),
      module_(std::move(module)),
      text_decoder_runner_(std::move(text_decoder_runner)),
      text_prefiller_(std::move(text_prefiller)),
      text_token_generator_(std::move(text_token_generator)),
      stats_(std::move(stats)),
      temperature_(temperature) {
  // Note: This constructor assumes that text_prefiller and text_token_generator
  // already have references to the Module and TextDecoderRunner they need
}

bool Runner::is_loaded() const {
  return text_prefiller_->is_loaded() && text_token_generator_->is_loaded();
}

Error Runner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(text_prefiller_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(text_token_generator_->load());
  return Error::Ok;
}

// Don't print with the same priority during warmup
#define RUNNER_ET_LOG(warmup, format, ...) \
  if (warmup) {                            \
    ET_LOG(Debug, format, __VA_ARGS__);    \
  } else {                                 \
    ET_LOG(Info, format, __VA_ARGS__);     \
  }

Error Runner::generate(
    const std::string& prompt,
    const ::executorch::extension::llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const llm::Stats&)> stats_callback) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    stats_->model_load_start_ms = llm::time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_->model_load_end_ms = llm::time_in_ms();
  }

  if (config.warming) {
    ET_LOG(Info, "Doing a warmup run...");
  }

  RUNNER_ET_LOG(
      config.warming,
      "RSS after loading model: %f MiB (0 if unsupported)",
      llm::get_rss_bytes() / 1024.0 / 1024.0);

  // Wrap the token_callback with print function
  std::function<void(const std::string&)> wrapped_callback =
      [token_callback, config](const std::string& piece) {
        if (!config.warming) {
          llm::safe_printf(piece.c_str());
          fflush(stdout);
        }
        if (token_callback) {
          token_callback(piece);
        }
      };
  // First token time only measures the time it takes to encode the prompt and
  // return a response token.

  stats_->inference_start_ms = llm::time_in_ms();
  shouldStop_ = false;

  ::tokenizers::Result<std::vector<uint64_t>> encode_res = tokenizer_->encode(
      prompt,
      /* bos */ 0,
      /* eos */ 0);

  ET_CHECK_TK_OK_OR_RETURN_ERROR(
      encode_res.error(), "Failed to encode prompt %s", prompt.c_str());

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();

  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      num_prompt_tokens < metadata_.at(kMaxContextLen),
      "num_prompt_tokens %d >= max_seq_len_ %" PRId64
      ", Max seq length exceeded - please increase max seq len value in your export script",
      num_prompt_tokens,
      metadata_.at(kMaxContextLen));

  // Determine max_new_tokens using the GenerationConfig's resolve method
  int max_new_tokens = config.resolve_max_new_tokens(
      metadata_.at(kMaxContextLen), num_prompt_tokens);

  ET_LOG(Info, "Max new tokens resolved: %d", max_new_tokens);

  // Prefill first
  // Here feed all tokens to the model and get the next predicted token
  // after the prompt. After that we will enter generate loop.

  // print prompts
  if (config.echo) {
    wrapped_callback(prompt);
  }
  int64_t pos = 0;
  auto prefill_res = text_prefiller_->prefill(prompt_tokens, pos);
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  uint64_t cur_token = prefill_res.get();
  stats_->first_token_ms = llm::time_in_ms();
  stats_->prompt_eval_end_ms = llm::time_in_ms();

  // print the first token from prefill. No prev_token so use cur_token for it.
  wrapped_callback(
      ET_UNWRAP_TOKENIZER(tokenizer_->decode(cur_token, cur_token)));
  RUNNER_ET_LOG(
      config.warming,
      "RSS after prompt prefill: %f MiB (0 if unsupported)",
      llm::get_rss_bytes() / 1024.0 / 1024.0);

  // start the main loop
  prompt_tokens.push_back(cur_token);

  // Generate max_new_tokens - 1 because prefill already generated 1 token.
  int64_t num_generated_tokens = ET_UNWRAP(text_token_generator_->generate(
      prompt_tokens,
      num_prompt_tokens,
      max_new_tokens - 1,
      temperature_ == -1.0f ? config.temperature : temperature_,
      wrapped_callback));

  stats_->inference_end_ms = llm::time_in_ms();
  if (!config.warming) {
    printf("\n");
  }
  RUNNER_ET_LOG(
      config.warming,
      "RSS after finishing text generation: %f MiB (0 if unsupported)",
      llm::get_rss_bytes() / 1024.0 / 1024.0);

  if (num_generated_tokens == max_new_tokens) {
    RUNNER_ET_LOG(config.warming, "Max new tokens %i reached!", max_new_tokens);
  }

  stats_->num_prompt_tokens = num_prompt_tokens;
  stats_->num_generated_tokens = num_generated_tokens;

  if (config.warming) {
    ET_LOG(Info, "Warmup run finished!");
  } else {
    // Do not print report during warmup
    ::executorch::llm::print_report(*stats_);
  }
  if (stats_callback) {
    stats_callback(*stats_);
  }

  return Error::Ok;
}

Error Runner::warmup(const std::string& prompt, int32_t max_new_tokens) {
  // Create a GenerationConfig for warmup
  llm::GenerationConfig config{
      .echo = false, .max_new_tokens = max_new_tokens, .warming = true};

  // Call generate with the warmup config
  Error err = generate(prompt, config);

  // Reset stats after warmup, not resetting the std::unique_ptr!
  stats_->reset();
  return err;
}

void Runner::stop() {
  if (is_loaded()) {
    text_token_generator_->stop();
  } else {
    ET_LOG(Error, "Token generator is not loaded, cannot stop");
  }
}
} // namespace example
