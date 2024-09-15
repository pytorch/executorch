/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llama2/runner/runner.h>

#include <ctime>

#include <executorch/extension/llm/runner/util.h>

#include <executorch/examples/models/llama2/tokenizer/llama_tiktoken.h>
#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>

namespace example {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace llm = ::executorch::extension::llm;

namespace {
static constexpr auto kAppendEosToPrompt = "append_eos_to_prompt";
static constexpr auto kEnableDynamicShape = "enable_dynamic_shape";
static constexpr auto kBosId = "get_bos_id";
static constexpr auto kEosIds = "get_eos_ids";
static constexpr auto kMaxSeqLen = "get_max_seq_len";
static constexpr auto kNBos = "get_n_bos";
static constexpr auto kNEos = "get_n_eos";
static constexpr auto kVocabSize = "get_vocab_size";
static constexpr auto kUseKVCache = "use_kv_cache";
static constexpr auto kUseSDPAWithKVCache = "use_sdpa_with_kv_cache";
} // namespace

Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const float temperature)
    // NOTE: we observed ~2x loading performance increase on iPhone 15
    // and a ~5% improvement on Galaxy S22 by switching to
    // FileDataLoader instead of MmapDataLoader + UseMlockIgnoreErrors.
    : temperature_(temperature),
      module_(std::make_unique<Module>(model_path, Module::LoadMode::File)),
      tokenizer_path_(tokenizer_path),
      metadata_({
          {kAppendEosToPrompt, false},
          {kEnableDynamicShape, false},
          {kMaxSeqLen, 128},
          {kNBos, 1},
          {kNEos, 1},
          {kUseKVCache, true},
          {kUseSDPAWithKVCache, false},
      }) {
  ET_LOG(
      Info,
      "Creating LLaMa runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());
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
  Error err = tokenizer_->load(tokenizer_path_);
  // Rely on tiktoken to throw error if the artifact is incompatible. Then we
  // fallback to BPE tokenizer.
  if (err == Error::InvalidArgument) {
    ET_LOG(
        Info,
        "Failed to load %s as a Tiktoken artifact, trying BPE tokenizer",
        tokenizer_path_.c_str());
    tokenizer_.reset();
    tokenizer_ = std::make_unique<llm::BPETokenizer>();
    tokenizer_->load(tokenizer_path_);
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
  text_decoder_runner_ = std::make_unique<llm::TextDecoderRunner>(
      module_.get(),
      metadata_.at(kUseKVCache),
      metadata_.at(kVocabSize),
      temperature_);
  text_prefiller_ = std::make_unique<llm::TextPrefiller>(
      text_decoder_runner_.get(),
      metadata_.at(kUseKVCache),
      metadata_.at(kEnableDynamicShape));

  text_token_generator_ = std::make_unique<llm::TextTokenGenerator>(
      tokenizer_.get(),
      text_decoder_runner_.get(),
      metadata_.at(kUseKVCache),
      std::move(eos_ids),
      &stats_);

  return Error::Ok;
}

Error Runner::generate(
    const std::string& prompt,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const llm::Stats&)> stats_callback,
    bool echo) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    stats_.model_load_start_ms = llm::time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = llm::time_in_ms();
  }

  ET_LOG(
      Info,
      "RSS after loading model: %f MiB (0 if unsupported)",
      llm::get_rss_bytes() / 1024.0 / 1024.0);

  // Wrap the token_callback with print function
  std::function<void(const std::string&)> wrapped_callback =
      [token_callback](const std::string& piece) {
        llm::safe_printf(piece.c_str());
        fflush(stdout);
        if (token_callback) {
          token_callback(piece);
        }
      };
  // First token time only measures the time it takes to encode the prompt and
  // return a response token.

  stats_.inference_start_ms = llm::time_in_ms();
  shouldStop_ = false;

  // Set the sequence length to the max seq length if not provided
  seq_len = (seq_len > 0 && seq_len <= metadata_.at(kMaxSeqLen))
      ? seq_len
      : metadata_.at(kMaxSeqLen);

  Result<std::vector<uint64_t>> encode_res = tokenizer_->encode(
      prompt,
      metadata_.at(kNBos),
      metadata_.at(kAppendEosToPrompt) ? metadata_.at(kNEos) : 0);

  ET_CHECK_OK_OR_RETURN_ERROR(
      encode_res.error(), "Failed to encode prompt %s", prompt.c_str());

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();

  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      num_prompt_tokens < metadata_.at(kMaxSeqLen),
      "num_prompt_tokens %d >= max_seq_len_ %" PRId64
      ", Max seq length exceeded - please increase max seq len value in .../llama2/model.py",
      num_prompt_tokens,
      metadata_.at(kMaxSeqLen));
  ET_CHECK_MSG(
      num_prompt_tokens < seq_len,
      "num_prompt_tokens %d >= seq_len %d, Sequence length exceeded - please increase the seq_len value passed to generate()",
      num_prompt_tokens,
      seq_len);

  // Prefill first
  // Here feed all tokens to the model and get the next predicted token
  // after the prompt. After that we will enter generate loop.

  // print prompts
  if (echo) {
    wrapped_callback(prompt);
  }
  int64_t pos = 0;
  auto prefill_res = text_prefiller_->prefill(prompt_tokens, pos);
  stats_.first_token_ms = llm::time_in_ms();
  stats_.prompt_eval_end_ms = llm::time_in_ms();
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  uint64_t cur_token = prefill_res.get();

  // print the first token from prefill. No prev_token so use cur_token for it.
  wrapped_callback(ET_UNWRAP(tokenizer_->decode(cur_token, cur_token)));
  ET_LOG(
      Info,
      "RSS after prompt prefill: %f MiB (0 if unsupported)",
      llm::get_rss_bytes() / 1024.0 / 1024.0);

  // start the main loop
  prompt_tokens.push_back(cur_token);
  int64_t num_generated_tokens = ET_UNWRAP(text_token_generator_->generate(
      prompt_tokens, num_prompt_tokens, seq_len, wrapped_callback));

  stats_.inference_end_ms = llm::time_in_ms();
  printf("\n");
  ET_LOG(
      Info,
      "RSS after finishing text generation: %f MiB (0 if unsupported)",
      llm::get_rss_bytes() / 1024.0 / 1024.0);

  if (num_prompt_tokens + num_generated_tokens == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = num_generated_tokens;
  ::executorch::llm::print_report(stats_);
  if (stats_callback) {
    stats_callback(stats_);
  }

  return Error::Ok;
}

void Runner::stop() {
  if (is_loaded()) {
    text_token_generator_->stop();
  } else {
    ET_LOG(Error, "Token generator is not loaded, cannot stop");
  }
}
} // namespace example
