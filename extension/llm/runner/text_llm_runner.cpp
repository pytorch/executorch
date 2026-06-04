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

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/tiktoken.h>

namespace executorch::extension::llm {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

TextLLMRunner::TextLLMRunner(
    std::unordered_map<std::string, int64_t> metadata,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<::executorch::extension::Module> module,
    std::unique_ptr<TextDecoderRunner> text_decoder_runner,
    std::unique_ptr<TextPrefiller> text_prefiller,
    std::unique_ptr<IOManager> io_manager,
    std::unique_ptr<TextTokenGenerator> text_token_generator,
    std::unique_ptr<Stats> stats,
    float temperature)
    : tokenizer_(std::move(tokenizer)),
      metadata_(std::move(metadata)),
      module_(std::move(module)),
      text_decoder_runner_(std::move(text_decoder_runner)),
      text_prefiller_(std::move(text_prefiller)),
      io_manager_(std::move(io_manager)),
      text_token_generator_(std::move(text_token_generator)),
      stats_(std::move(stats)),
      temperature_(temperature),
      pos_(0) {
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
  ET_CHECK_OK_OR_RETURN_ERROR(io_manager_->load());
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

  // Get max_seq_len for single prefill chunk limit
  int64_t max_seq_len = metadata_.at(kMaxSeqLen);
  int64_t max_context_len = metadata_.at(kMaxContextLen);

  // Resolve sampling temperature once: the first token is sampled during
  // prefill and the rest in the token generator, so both must use the same
  // temperature.
  const float resolved_temp =
      temperature_ == -1.0f ? config.temperature : temperature_;

  uint64_t cur_token = 0;
  int num_prompt_tokens = 0;
  std::vector<uint64_t> prompt_tokens;

  if (!prompt.empty()) {
    ::tokenizers::Result<std::vector<uint64_t>> encode_res = tokenizer_->encode(
        prompt, /*bos=*/config.num_bos, /*eos=*/config.num_eos);

    if (!encode_res.ok()) {
      ET_LOG(
          Error,
          "Failed to encode prompt %s. Tokenizers error code %d",
          prompt.c_str(),
          static_cast<uint32_t>(encode_res.error()));
      return Error::InvalidArgument;
    }

    // encode the (string) prompt into tokens sequence
    prompt_tokens = encode_res.get();
    num_prompt_tokens = prompt_tokens.size();

    ET_CHECK_OR_RETURN_ERROR(
        num_prompt_tokens >= 1,
        InvalidArgument,
        "Expected at least 1 prompt token");
    // Note: We intentionally do NOT enforce num_prompt_tokens <= max_seq_len
    // here. TextPrefiller::prefill() supports chunked prefill: when
    // num_prompt_tokens > max_seq_len it splits the prompt into max_seq_len
    // chunks and prefills them sequentially. Models that were exported with
    // max_seq_len < max_context_len (e.g. a 1024 prefill chunk over a 4096 KV
    // cache) rely on this behavior.
    // Ensure the prompt fits within total KV cache capacity. For
    // sliding-window models (where max_seq_len < max_context_len) the model
    // handles position wrapping internally, so pos_ doesn't represent
    // consumed capacity and we only need a per-call bound.
    if (max_seq_len >= max_context_len) {
      ET_CHECK_OR_RETURN_ERROR(
          pos_ + num_prompt_tokens < max_context_len,
          InvalidArgument,
          "pos_ %" PRId64 " + num_prompt_tokens %d >= max_context_len %" PRId64
          ", Max seq length exceeded - please increase max seq len value in "
          "your export script",
          pos_,
          num_prompt_tokens,
          max_context_len);
    } else {
      ET_CHECK_OR_RETURN_ERROR(
          num_prompt_tokens < max_context_len,
          InvalidArgument,
          "num_prompt_tokens %d >= max_context_len %" PRId64
          ", Prompt exceeds KV cache capacity - please reduce prompt size or "
          "increase max_context_len in your export script",
          num_prompt_tokens,
          max_context_len);
    }

    // print prompts
    if (config.echo) {
      wrapped_callback(prompt);
    }

    // Prefill first
    // Here feed all tokens to the model and get the next predicted token
    // after the prompt. After that we will enter generate loop.
    auto prefill_res =
        text_prefiller_->prefill(prompt_tokens, pos_, resolved_temp);
    ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
    cur_token = prefill_res.get();
    prefill_next_token_.reset();
  } else {
    // Empty prompt: consume token from a prior prefill() call
    ET_CHECK_OR_RETURN_ERROR(
        prefill_next_token_.has_value(),
        InvalidState,
        "Empty prompt requires a prior prefill() call");
    cur_token = prefill_next_token_.value();
    prefill_next_token_.reset();
  }

  // For sliding window models, the ring buffer recycles space — pos_ doesn't
  // represent consumed capacity, so pass 0 to get the full budget.
  int64_t effective_pos = (max_seq_len < max_context_len) ? 0 : pos_;
  int max_new_tokens =
      config.resolve_max_new_tokens(max_context_len, effective_pos);

  ET_LOG(
      Info,
      "Max new tokens resolved: %d, given pos_ %" PRId64
      ", num_prompt_tokens %d, max_context_len %" PRId64,
      max_new_tokens,
      pos_,
      num_prompt_tokens,
      max_context_len);
  ET_CHECK_OR_RETURN_ERROR(
      max_new_tokens > 0,
      InvalidArgument,
      "Max new tokens %d is less than or equal to 0",
      max_new_tokens);

  stats_->first_token_ms = time_in_ms();
  stats_->prompt_eval_end_ms = time_in_ms();

  // print the first token from prefill. No prev_token so use cur_token for it.
  auto decode_result = tokenizer_->decode(cur_token, cur_token);
  if (!decode_result.ok()) {
    ET_LOG(
        Error,
        "Tokenizers error code %d",
        static_cast<uint32_t>(decode_result.error()));
    return ::executorch::runtime::Error::InvalidArgument;
  }
  wrapped_callback(std::move(*decode_result));
  RUNNER_ET_LOG(
      config.warming,
      "RSS after prompt prefill: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  // start the main loop
  prompt_tokens.push_back(cur_token);

  // Set ignore_eos based on config
  text_token_generator_->set_ignore_eos(config.ignore_eos);

  // Generate max_new_tokens - 1 because prefill already generated 1 token.
  auto generate_result = text_token_generator_->generate(
      prompt_tokens, pos_, max_new_tokens - 1, resolved_temp, wrapped_callback);

  if (!generate_result.ok()) {
    return generate_result.error();
  }
  int64_t num_generated_tokens = generate_result.get();

  pos_ += num_generated_tokens;

  stats_->inference_end_ms = time_in_ms();
  if (!config.warming) {
    printf("\n");
  }
  RUNNER_ET_LOG(
      config.warming,
      "RSS after finishing text generation: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  if (num_generated_tokens == max_new_tokens - 1) {
    RUNNER_ET_LOG(config.warming, "Max new tokens %i reached!", max_new_tokens);
  }

  // The prefill step produced and emitted one token (cur_token) before the
  // token generator ran, so the total generated count is that token plus the
  // generator's. For an empty prompt (continuation/prefix-reuse path) the
  // prompt length is everything resident before this turn's generation (pos_
  // already includes the generated tokens, so subtract them).
  stats_->num_prompt_tokens = prompt.empty()
      ? static_cast<int64_t>(pos_) - num_generated_tokens
      : num_prompt_tokens;
  stats_->num_generated_tokens = num_generated_tokens + 1;

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

Result<uint64_t> TextLLMRunner::prefill(
    const std::vector<MultimodalInput>& inputs,
    int32_t num_bos,
    int32_t num_eos) {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  for (const auto& input : inputs) {
    if (input.is_text()) {
      auto encode_res = tokenizer_->encode(
          input.get_text(), /*bos=*/num_bos, /*eos=*/num_eos);
      ET_CHECK_TK_OK_OR_RETURN_ERROR(
          encode_res.error(),
          "Failed to encode prompt %s",
          input.get_text().c_str());
      std::vector<uint64_t> tokens = encode_res.get();
      auto prefill_res = text_prefiller_->prefill(tokens, pos_);
      ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
      prefill_next_token_ = prefill_res.get();
      num_bos = 0;
      num_eos = 0;
    }
    // Skip non-text inputs — text-only runner
  }

  if (!prefill_next_token_.has_value()) {
    return Error::InvalidArgument;
  }
  return prefill_next_token_.value();
}

Result<uint64_t> TextLLMRunner::prefill(
    const std::string& prompt,
    int32_t num_bos,
    int32_t num_eos) {
  std::vector<MultimodalInput> inputs;
  inputs.emplace_back(MultimodalInput(prompt));
  return prefill(inputs, num_bos, num_eos);
}

Result<uint64_t> TextLLMRunner::prefill(
    const std::string& prompt,
    const GenerationConfig& config) {
  return prefill(prompt, config.num_bos, config.num_eos);
}

Error TextLLMRunner::warmup(const std::string& prompt, int32_t max_new_tokens) {
  // Create a GenerationConfig for warmup
  GenerationConfig config;
  config.echo = false;
  config.max_new_tokens = max_new_tokens;
  config.warming = true;

  // Call generate with the warmup config
  Error err = generate(prompt, config);

  // Reset stats after warmup, not resetting the std::unique_ptr!
  reset();
  return err;
}

void TextLLMRunner::stop() {
  if (is_loaded()) {
    text_token_generator_->stop();
  } else {
    ET_LOG(Error, "Token generator is not loaded, cannot stop");
  }
}

void TextLLMRunner::reset() {
  stats_->reset();
  pos_ = 0;
  prefill_next_token_.reset();
  prev_decode_token_.reset();
}

::executorch::runtime::Error TextLLMRunner::seek(int64_t pos) {
  // Token-step primitives require a KV cache (a non-KV model has no resident KV
  // to rewind); fail closed.
  if (metadata_.at(kUseKVCache) == 0) {
    ET_LOG(Error, "seek() requires a KV-cache model (use_kv_cache=true)");
    return ::executorch::runtime::Error::NotSupported;
  }
  // Sliding-window models (max_seq_len < max_context_len) recycle KV space, so
  // pos_ is not an absolute position and the prefix [0, pos) may have slid out
  // of the window; rewinding would attend to stale KV. Refuse so the caller
  // falls back to reset() + full re-prefill.
  if (metadata_.at(kMaxSeqLen) < metadata_.at(kMaxContextLen)) {
    ET_LOG(
        Error,
        "seek() is unsupported for sliding-window models "
        "(max_seq_len %" PRId64 " < max_context_len %" PRId64 ")",
        metadata_.at(kMaxSeqLen),
        metadata_.at(kMaxContextLen));
    return ::executorch::runtime::Error::NotSupported;
  }
  if (pos < 0 || pos > pos_) {
    ET_LOG(Error, "seek(%" PRId64 ") out of range [0, %" PRId64 "]", pos, pos_);
    return ::executorch::runtime::Error::InvalidArgument;
  }
  pos_ = pos;
  prefill_next_token_.reset();
  prev_decode_token_.reset();
  return ::executorch::runtime::Error::Ok;
}

::executorch::runtime::Result<uint64_t> TextLLMRunner::prefill_tokens(
    std::vector<uint64_t> tokens,
    float temperature) {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }
  // The token-step primitives assume KV-cached decode (each step forwards only
  // the new token at pos_). A non-KV model needs full-sequence re-forwarding,
  // which this path does not implement — fail closed rather than decode wrong.
  ET_CHECK_OR_RETURN_ERROR(
      metadata_.at(kUseKVCache) != 0,
      NotSupported,
      "prefill_tokens/decode_one require a KV-cache model (use_kv_cache=true)");
  if (tokens.empty()) {
    ET_LOG(Error, "prefill_tokens called with empty tokens");
    return ::executorch::runtime::Error::InvalidArgument;
  }
  // Same context-capacity guard as generate(): a caller that seek()s then
  // prefill_tokens()es a suffix must not push pos_ past the KV cache. This is
  // the only place the bound is enforced for prefill_tokens() (the public
  // prefix-cache primitive), since it doesn't go through generate(prompt).
  const int64_t max_seq_len = metadata_.at(kMaxSeqLen);
  const int64_t max_context_len = metadata_.at(kMaxContextLen);
  const int num_tokens = static_cast<int>(tokens.size());
  if (max_seq_len >= max_context_len) {
    ET_CHECK_OR_RETURN_ERROR(
        pos_ + num_tokens < max_context_len,
        InvalidArgument,
        "pos_ %" PRId64 " + num_tokens %d >= max_context_len %" PRId64
        ", prefill_tokens would exceed KV cache capacity",
        pos_,
        num_tokens,
        max_context_len);
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        num_tokens < max_context_len,
        InvalidArgument,
        "num_tokens %d >= max_context_len %" PRId64
        ", prefill_tokens exceeds KV cache capacity",
        num_tokens,
        max_context_len);
  }
  // Resolve temperature like decode_one() so the first token (sampled here in
  // prefill) honors the request instead of defaulting to greedy.
  const float temp = (temperature < 0.0f)
      ? (temperature_ == -1.0f ? 0.0f : temperature_)
      : temperature;
  auto prefill_res = text_prefiller_->prefill(tokens, pos_, temp);
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  prefill_next_token_ = prefill_res.get();
  prev_decode_token_.reset();
  return prefill_next_token_.value();
}

::executorch::runtime::Result<DecodeResult> TextLLMRunner::decode_one(
    float temperature) {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }
  // See prefill_tokens(): single-token KV stepping is invalid without a KV
  // cache.
  ET_CHECK_OR_RETURN_ERROR(
      metadata_.at(kUseKVCache) != 0,
      NotSupported,
      "decode_one requires a KV-cache model (use_kv_cache=true)");
  ET_CHECK_OR_RETURN_ERROR(
      prefill_next_token_.has_value(),
      InvalidState,
      "decode_one requires a pending token; call prefill()/prefill_tokens() first");

  // The pending token is the one we emit this step.
  const uint64_t token = prefill_next_token_.value();
  const bool is_eos = text_token_generator_->is_eos(token);

  // Decode the text piece with BPE context (previous token), like generate().
  // Surface tokenizer errors rather than hiding them as an empty piece (matches
  // generate(), which logs and returns InvalidArgument).
  const uint64_t prev = prev_decode_token_.value_or(token);
  auto decode_res = tokenizer_->decode(prev, token);
  if (!decode_res.ok()) {
    ET_LOG(
        Error,
        "Tokenizers error code %d",
        static_cast<uint32_t>(decode_res.error()));
    return ::executorch::runtime::Error::InvalidArgument;
  }
  std::string text_piece = std::move(*decode_res);

  // Stop at EOS WITHOUT forwarding it, like generate() (which breaks before the
  // next step()): the EOS token is not made resident and pos_ does not advance,
  // so position()/prefix reuse stay correct. No pending token remains, so a
  // subsequent decode_one() correctly errors (generation is complete).
  if (is_eos) {
    prefill_next_token_.reset();
    return DecodeResult{token, std::move(text_piece), true};
  }

  // Only a NON-EOS token is forwarded (made resident at pos_), so the capacity
  // check belongs here — after the EOS short-circuit. This lets the final EOS
  // be emitted even when the KV cache is exactly full.
  if (metadata_.at(kMaxSeqLen) >= metadata_.at(kMaxContextLen)) {
    ET_CHECK_OR_RETURN_ERROR(
        pos_ < metadata_.at(kMaxContextLen),
        InvalidArgument,
        "decode_one would exceed KV cache capacity: pos_ %" PRId64
        " >= max_context_len %" PRId64,
        pos_,
        metadata_.at(kMaxContextLen));
  }

  // Forward `token` at pos_ to predict the next pending token.
  std::vector<uint64_t> tok_data = {token};
  std::vector<::executorch::aten::SizesType> shape = {1, 1};
  auto tok_tensor =
      from_blob(tok_data.data(), shape, ::executorch::aten::ScalarType::Long);
  auto logits_res = text_decoder_runner_->step(tok_tensor, pos_);
  ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
  // Apply the same logit processors generate() does (grammar/tool masks,
  // penalties, top-k/top-p) so the session decode path can't diverge from it.
  ET_CHECK_OK_OR_RETURN_ERROR(
      text_token_generator_->apply_logit_processors(logits_res.get()));
  const float temp = (temperature < 0.0f)
      ? (temperature_ == -1.0f ? 0.0f : temperature_)
      : temperature;
  prefill_next_token_ = static_cast<uint64_t>(
      text_decoder_runner_->logits_to_token(logits_res.get(), temp));
  prev_decode_token_ = token;
  pos_ += 1;

  return DecodeResult{token, std::move(text_piece), false};
}

} // namespace executorch::extension::llm
