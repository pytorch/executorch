/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llama/runner/runner.h>

#include <algorithm>
#include <ctime>

#include <executorch/extension/llm/runner/util.h>

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
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
} // namespace

Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    std::optional<const std::string> data_path)
    // NOTE: we observed ~2x loading performance increase on iPhone 15
    // and a ~5% improvement on Galaxy S22 by switching to
    // FileDataLoader instead of MmapDataLoader + UseMlockIgnoreErrors.
    : tokenizer_path_(tokenizer_path),
      metadata_({
          {kEnableDynamicShape, false},
          {kMaxSeqLen, 128},
          {kMaxContextLen, 128},
          {kUseKVCache, true},
          {kUseSDPAWithKVCache, false},
      }) {
  if (data_path.has_value()) {
    module_ = std::make_unique<Module>(
        model_path, data_path.value(), Module::LoadMode::File);
  } else {
    module_ = std::make_unique<Module>(model_path, Module::LoadMode::File);
  }
  ET_LOG(
      Info,
      "Creating LLaMa runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());
}

[[deprecated(
    "This constructor is deprecated. Use the constructor without temperature parameter instead.")]]
Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const float temperature,
    std::optional<const std::string> data_path)
    : Runner(model_path, tokenizer_path, std::move(data_path)) {
  temperature_ = temperature;
}

bool Runner::is_loaded() const {
  return module_->is_loaded() && tokenizer_ && text_decoder_runner_ &&
      text_prefiller_ && text_token_generator_;
}

Error Runner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method("forward"));
  // load tokenizer. Assuming tiktoken is the default tokenizer
  tokenizer_ = nullptr;
  tokenizer_ = get_tiktoken_for_llama();
  ::tokenizers::Error err = tokenizer_->load(tokenizer_path_);
  // Rely on tiktoken to throw error if the artifact is incompatible. Then we
  // fallback to BPE tokenizer.
  if (err != ::tokenizers::Error::Ok) {
    ET_LOG(
        Info,
        "Failed to load %s as a Tiktoken artifact, trying BPE tokenizer",
        tokenizer_path_.c_str());
    tokenizer_.reset();
    tokenizer_ = std::make_unique<::tokenizers::Llama2cTokenizer>();
    err = tokenizer_->load(tokenizer_path_);
    ET_CHECK_TK_OK_OR_RETURN_ERROR(
        err,
        "Failed to load %s as a llama2.c tokenizer artifact",
        tokenizer_path_.c_str());
  }

  ET_LOG(Info, "Reading metadata from model");

  metadata_[kBosId] = tokenizer_->bos_tok();
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      std::unordered_set<uint64_t>{tokenizer_->eos_tok()});
  metadata_[kVocabSize] = tokenizer_->vocab_size();

  const auto method_names =
      ET_UNWRAP(module_->method_names(), "Failed reading method names");

  for (auto& pair : metadata_) {
    const auto& method_name = pair.first;
    auto& value = pair.second;

    if (method_names.count(method_name)) {
      value = ET_UNWRAP(module_->get(method_name))
                  .toScalar()
                  .to<decltype(metadata_)::mapped_type>();
    } else {
      ET_LOG(
          Info,
          "Methond %s not found, using the default value %" PRId64,
          method_name.c_str(),
          value);
    }
    ET_LOG(Info, "Metadata: %s = %" PRId64, method_name.c_str(), value);
  }
  if (method_names.count(kEosIds)) {
    eos_ids->clear();
    for (const auto& eos_id : ET_UNWRAP(module_->execute(kEosIds))) {
      auto value = eos_id.toScalar().to<int64_t>();
      eos_ids->emplace(value);
      ET_LOG(Info, "eos_id = %" PRId64, value);
    }
  }
  // @lint-ignore CLANGTIDY facebook-hte-Deprecated
  text_decoder_runner_ = std::make_unique<llm::TextDecoderRunner>(
      module_.get(), metadata_.at(kUseKVCache));
  text_prefiller_ = std::make_unique<llm::TextPrefiller>(
      text_decoder_runner_.get(),
      metadata_.at(kUseKVCache),
      metadata_.at(kEnableDynamicShape),
      metadata_.at(kMaxSeqLen));

  text_token_generator_ = std::make_unique<llm::TextTokenGenerator>(
      tokenizer_.get(),
      text_decoder_runner_.get(),
      metadata_.at(kUseKVCache),
      std::move(eos_ids),
      &stats_);

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
    stats_.model_load_start_ms = llm::time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = llm::time_in_ms();
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

  stats_.inference_start_ms = llm::time_in_ms();
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
  stats_.first_token_ms = llm::time_in_ms();
  stats_.prompt_eval_end_ms = llm::time_in_ms();

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

  stats_.inference_end_ms = llm::time_in_ms();
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

  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = num_generated_tokens;

  if (config.warming) {
    ET_LOG(Info, "Warmup run finished!");
  } else {
    // Do not print report during warmup
    ::executorch::llm::print_report(stats_);
  }
  if (stats_callback) {
    stats_callback(stats_);
  }

  return Error::Ok;
}

Error Runner::warmup(const std::string& prompt, int32_t max_new_tokens) {
  // Create a GenerationConfig for warmup
  llm::GenerationConfig config{
      .echo = false, .max_new_tokens = max_new_tokens, .warming = true};

  // Call generate with the warmup config
  Error err = generate(prompt, config);

  // Reset stats after warmup
  stats_.reset();
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
