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

#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/tiktoken.h>

namespace executorch::extension::llm {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

static constexpr auto kEnableDynamicShape = "enable_dynamic_shape";
static constexpr auto kBosId = "get_bos_id";
static constexpr auto kEosIds = "get_eos_ids";
static constexpr auto kMaxSeqLen = "get_max_seq_len";
static constexpr auto kMaxContextLen = "get_max_context_len";
static constexpr auto kVocabSize = "get_vocab_size";
static constexpr auto kUseKVCache = "use_kv_cache";
static constexpr auto kUseSDPAWithKVCache = "use_sdpa_with_kv_cache";

TextLLMRunner::TextLLMRunner(
    std::unordered_map<std::string, int64_t> metadata,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<::executorch::extension::Module> module,
    std::unique_ptr<TextDecoderRunner> text_decoder_runner,
    std::unique_ptr<TextPrefiller> text_prefiller,
    std::unique_ptr<TextTokenGenerator> text_token_generator,
    std::unique_ptr<Stats> stats,
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

bool TextLLMRunner::is_loaded() const {
  return text_prefiller_->is_loaded() && text_token_generator_->is_loaded();
}

Error TextLLMRunner::load() {
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

Error TextLLMRunner::generate(
    const std::string& prompt,
    const GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    stats_->model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_->model_load_end_ms = time_in_ms();
  }

  if (config.warming) {
    ET_LOG(Info, "Doing a warmup run...");
  }

  RUNNER_ET_LOG(
      config.warming,
      "RSS after loading model: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

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

  stats_->inference_start_ms = time_in_ms();
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
  stats_->first_token_ms = time_in_ms();
  stats_->prompt_eval_end_ms = time_in_ms();

  // print the first token from prefill. No prev_token so use cur_token for it.
  wrapped_callback(
      ET_UNWRAP_TOKENIZER(tokenizer_->decode(cur_token, cur_token)));
  RUNNER_ET_LOG(
      config.warming,
      "RSS after prompt prefill: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  // start the main loop
  prompt_tokens.push_back(cur_token);

  // Generate max_new_tokens - 1 because prefill already generated 1 token.
  int64_t num_generated_tokens = ET_UNWRAP(text_token_generator_->generate(
      prompt_tokens,
      num_prompt_tokens,
      max_new_tokens - 1,
      temperature_ == -1.0f ? config.temperature : temperature_,
      wrapped_callback));

  stats_->inference_end_ms = time_in_ms();
  if (!config.warming) {
    printf("\n");
  }
  RUNNER_ET_LOG(
      config.warming,
      "RSS after finishing text generation: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  if (num_generated_tokens == max_new_tokens) {
    RUNNER_ET_LOG(config.warming, "Max new tokens %i reached!", max_new_tokens);
  }

  stats_->num_prompt_tokens = num_prompt_tokens;
  stats_->num_generated_tokens = num_generated_tokens;

  if (config.warming) {
    ET_LOG(Info, "Warmup run finished!");
  } else {
    // Do not print report during warmup
    print_report(*stats_);
  }
  if (stats_callback) {
    stats_callback(*stats_);
  }

  return Error::Ok;
}

Error TextLLMRunner::warmup(const std::string& prompt, int32_t max_new_tokens) {
  // Create a GenerationConfig for warmup
  GenerationConfig config{
      .echo = false, .max_new_tokens = max_new_tokens, .warming = true};

  // Call generate with the warmup config
  Error err = generate(prompt, config);

  // Reset stats after warmup, not resetting the std::unique_ptr!
  stats_->reset();
  return err;
}

void TextLLMRunner::stop() {
  if (is_loaded()) {
    text_token_generator_->stop();
  } else {
    ET_LOG(Error, "Token generator is not loaded, cannot stop");
  }
}

std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(
    const std::string& tokenizer_path,
    std::unique_ptr<std::vector<std::string>> special_tokens,
    std::optional<std::string> pattern,
    size_t bos_token_index,
    size_t eos_token_index) {
  auto json_tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (json_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded json tokenizer");
    return json_tokenizer;
  }
  std::unique_ptr<::tokenizers::Tiktoken> tiktoken_tokenizer;
  if (special_tokens != nullptr && !pattern.has_value()) {
    tiktoken_tokenizer = std::make_unique<::tokenizers::Tiktoken>(
        std::move(special_tokens), bos_token_index, eos_token_index);
  } else if (special_tokens != nullptr && pattern.has_value()) {
    tiktoken_tokenizer = std::make_unique<::tokenizers::Tiktoken>(
        pattern.value(),
        std::move(special_tokens),
        bos_token_index,
        eos_token_index);
  } else {
    tiktoken_tokenizer = std::make_unique<::tokenizers::Tiktoken>();
  }
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

std::unordered_map<std::string, int64_t> get_llm_metadata(
    tokenizers::Tokenizer* tokenizer,
    Module* module) {
  // Initialize metadata with default values
  std::unordered_map<std::string, int64_t> metadata({
      {llm::kEnableDynamicShape, false},
      {llm::kMaxSeqLen, 128},
      {llm::kMaxContextLen, 128},
      {llm::kUseKVCache, true},
      {llm::kUseSDPAWithKVCache, false},
  });

  // Read metadata from the model
  auto method_names_result = module->method_names();
  if (method_names_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed reading method names");
    return metadata;
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
  // Set tokenizer-related metadata
  metadata[llm::kBosId] = tokenizer->bos_tok();
  metadata[llm::kVocabSize] = tokenizer->vocab_size();
  return metadata;
}

std::unordered_set<uint64_t> get_eos_ids(
    tokenizers::Tokenizer* tokenizer,
    Module* module) {
  std::unordered_set<uint64_t> eos_ids = {tokenizer->eos_tok()};
  // Get EOS IDs if available
  auto method_names_result = module->method_names();
  if (method_names_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed reading method names");
    return eos_ids;
  }
  const auto method_names = method_names_result.get();

  if (method_names.count(llm::kEosIds)) {
    eos_ids.clear();
    auto execute_result = module->execute(llm::kEosIds);
    if (execute_result.error() != Error::Ok) {
      ET_LOG(Error, "Failed to execute %s", llm::kEosIds);
      return eos_ids;
    }
    for (const auto& eos_id : execute_result.get()) {
      auto value = eos_id.toScalar().to<int64_t>();
      eos_ids.emplace(value);
      ET_LOG(Info, "eos_id = %" PRId64, value);
    }
  }
  return eos_ids;
}

std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path,
    float temperature) {
  // Sanity check tokenizer
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer is null or not loaded");
    return nullptr;
  }

  // Create the Module
  std::unique_ptr<Module> module;
  if (data_path.has_value()) {
    module = std::make_unique<Module>(
        model_path, data_path.value(), Module::LoadMode::File);
  } else {
    module = std::make_unique<Module>(model_path, Module::LoadMode::File);
  }

  // Get metadata from Module
  ET_LOG(Info, "Reading metadata from model");
  auto metadata = llm::get_llm_metadata(tokenizer.get(), module.get());

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      llm::get_eos_ids(tokenizer.get(), module.get()));

  // Create text_decoder_runner. Use a shared_ptr so that it can be shared with
  // TextPrefiller and TextTokenGenerator
  auto text_decoder_runner = std::make_unique<TextDecoderRunner>(
      module.get(), metadata.at(kUseKVCache));

  // Create text_prefiller
  auto text_prefiller = std::make_unique<TextPrefiller>(
      text_decoder_runner.get(),
      metadata.at(kUseKVCache),
      metadata.at(kEnableDynamicShape),
      metadata.at(kMaxSeqLen));

  // Create text_token_generator with stats
  auto stats = std::make_unique<Stats>();
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(kUseKVCache),
      std::move(eos_ids),
      stats.get());

  // Create and return the Runner instance
  return std::make_unique<TextLLMRunner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::move(text_prefiller),
      std::move(text_token_generator),
      std::move(stats),
      temperature);
}

} // namespace executorch::extension::llm
