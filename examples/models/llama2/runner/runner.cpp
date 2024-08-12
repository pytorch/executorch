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
#if ET_USE_TIKTOKEN
#include <executorch/examples/models/llama2/tokenizer/llama_tiktoken.h>
#else /* BPE */
#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>
#endif /* ET_USE_TIKTOKEN*/
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/extension/module/metadata_util.h>
#include <executorch/extension/runner_util/managed_tensor.h>

#include <ctime>
#include <memory>
#include <sstream>

#ifdef USE_ATEN_LIB
#include <torch/torch.h>
#endif

#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

namespace torch::executor {

Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const float temperature)
    // NOTE: we observed ~2x loading performance increase on iPhone 15
    // and a ~5% improvement on Galaxy S22 by switching to
    // FileDataLoader instead of MmapDataLoader + UseMlockIgnoreErrors.
    : temperature_(temperature),
      module_(std::make_unique<Module>(model_path, Module::LoadMode::File)),
      tokenizer_path_(tokenizer_path) {
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

  // Read out metadata: vocab_size (expected by the model), BOS, EOS, n_BOS,
  // n_EOS max_seq_len from the model
  ET_LOG(Info, "Reading metadata from model");
  const auto method_names = module_->method_names();
  ET_CHECK_MSG(method_names.ok(), "Failed to read method names from model");
  model_methods_ = method_names.get();
  n_bos_ = get_module_metadata<int64_t>(module_.get(), "get_n_bos", 1);
  n_eos_ = get_module_metadata<int64_t>(module_.get(), "get_n_eos", 1);
  max_seq_len_ =
      get_module_metadata<int64_t>(module_.get(), "get_max_seq_len", 128);
  use_kv_cache_ = get_module_metadata(module_.get(), "use_kv_cache", true);
  use_sdpa_with_kv_cache_ =
      get_module_metadata(module_.get(), "use_sdpa_with_kv_cache", false);
  append_eos_ =
      get_module_metadata(module_.get(), "append_eos_to_prompt", false);
  enable_parallel_prefill_ =
      get_module_metadata(module_.get(), "enable_dynamic_shape", false);

  // Load tokenizer
#if ET_USE_TIKTOKEN
  tokenizer_ = get_tiktoken_for_llama();
#else
  tokenizer_ = std::make_unique<BPETokenizer>();
#endif
  tokenizer_->load(tokenizer_path_);

  vocab_size_ = get_module_metadata<int64_t>(
      module_.get(), "get_vocab_size", tokenizer_->vocab_size());
  bos_id_ = get_module_metadata<int64_t>(
      module_.get(), "get_bos_id", tokenizer_->bos_tok());
  eos_id_ = get_module_metadata<int64_t>(
      module_.get(), "get_eos_id", tokenizer_->eos_tok());

  // Create text decoder runner and prefiller
  text_decoder_runner_ = std::make_unique<TextDecoderRunner>(
      module_.get(), use_kv_cache_, vocab_size_, temperature_);

  text_prefiller_ = std::make_unique<TextPrefiller>(
      tokenizer_.get(),
      text_decoder_runner_.get(),
      use_kv_cache_,
      enable_parallel_prefill_);

  text_token_generator_ = std::make_unique<TextTokenGenerator>(
      tokenizer_.get(),
      text_decoder_runner_.get(),
      use_kv_cache_,
      eos_id_,
      &stats_);

  return Error::Ok;
}

Error Runner::generate(
    const std::string& prompt,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    stats_.model_load_start_ms = util::time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = util::time_in_ms();
  }

  // Wrap the token_callback with print function
  std::function<void(const std::string&)> wrapped_callback =
      [token_callback](const std::string& piece) {
        util::safe_printf(piece.c_str());
        fflush(stdout);
        if (token_callback) {
          token_callback(piece);
        }
      };
  // First token time only measures the time it takes to encode the prompt and
  // return a response token.

  stats_.inference_start_ms = util::time_in_ms();
  shouldStop_ = false;

  // Set the sequence length to the max seq length if not provided
  seq_len = (seq_len > 0 && seq_len <= max_seq_len_) ? seq_len : max_seq_len_;

  Result<std::vector<uint64_t>> encode_res =
      tokenizer_->encode(prompt, n_bos_, append_eos_ ? n_eos_ : 0);

  ET_CHECK_OK_OR_RETURN_ERROR(
      encode_res.error(), "Failed to encode prompt %s", prompt.c_str());

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();

  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      num_prompt_tokens < max_seq_len_,
      "num_prompt_tokens %d >= max_seq_len_ %d, Max seq length exceeded - please increase max seq len value in .../llama2/model.py",
      num_prompt_tokens,
      max_seq_len_);

  ET_CHECK_MSG(
      num_prompt_tokens < seq_len,
      "num_prompt_tokens %d >= seq_len %d, Sequence length exceeded - please increase the seq_len value passed to generate()",
      num_prompt_tokens,
      seq_len);

  // Prefill first
  // Here feed all tokens to the model and get the next predicted token
  // after the prompt. After that we will enter generate loop.
  auto prefill_res =
      text_prefiller_->prefill(prompt_tokens, 0, wrapped_callback);
  stats_.first_token_ms = util::time_in_ms();
  stats_.prompt_eval_end_ms = util::time_in_ms();
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  uint64_t cur_token = prefill_res.get();

  // print the first token from prefill. No prev_token so use cur_token for it.
  wrapped_callback(ET_UNWRAP(tokenizer_->decode(cur_token, cur_token)));

  // start the main loop
  prompt_tokens.push_back(cur_token);
  int64_t num_generated_tokens = ET_UNWRAP(text_token_generator_->generate(
      prompt_tokens, num_prompt_tokens, seq_len, wrapped_callback));

  stats_.inference_end_ms = util::time_in_ms();
  printf("\n");

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
} // namespace torch::executor
