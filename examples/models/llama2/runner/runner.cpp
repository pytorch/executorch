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

#include <executorch/examples/models/llama2/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

/**
 * Use tokenizer to decode token, print it, then execute callback.
 */
#define _DECODE_PRINT_CALLBACK(prev, cur, callback) \
  ({                                                \
    auto piece_res = tokenizer_->decode(prev, cur); \
    ET_CHECK_OK_OR_RETURN_ERROR(piece_res.error()); \
    const char* piece = piece_res.get().c_str();    \
    util::safe_printf(piece);                       \
    fflush(stdout);                                 \
    if (token_callback) {                           \
      token_callback(piece);                        \
    }                                               \
  })

namespace torch::executor {

Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const float temperature)
    // NOTE: we observed ~2x loading performance increase on iPhone 15
    // and a ~5% improvement on Galaxy S22 by switching to
    // FileDataLoader instead of MmapDataLoader + UseMlockIgnoreErrors.
    : module_(std::make_unique<Module>(model_path, Module::LoadMode::File)),
      tokenizer_path_(tokenizer_path),
      temperature_(temperature) {
  ET_LOG(
      Info,
      "Creating LLaMa runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());
}

bool Runner::is_loaded() const {
  return module_->is_loaded() && tokenizer_ && sampler_;
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

  // Create sampler
  sampler_ = std::make_unique<Sampler>(
      vocab_size_,
      temperature_,
      ::executorch::llm::kTopp,
      static_cast<unsigned long long>(std::time(nullptr)));

  return Error::Ok;
}

int32_t Runner::logitsToToken(const exec_aten::Tensor& logits_tensor) {
  ET_CHECK_MSG(logits_tensor.dim() == 3, "Logits tensor must be 3D");
  auto num_tokens = logits_tensor.size(1);

  switch (logits_tensor.scalar_type()) {
    case ScalarType::Float: {
      float* logits = logits_tensor.mutable_data_ptr<float>();
      float* logits_last = logits;
      logits_last += (num_tokens - 1) * tokenizer_->vocab_size();
      return sampler_->sample(logits_last);
    }
    case ScalarType::Half: {
      exec_aten::Half* logits =
          logits_tensor.mutable_data_ptr<exec_aten::Half>();
      exec_aten::Half* logits_last = logits;
      logits_last += (num_tokens - 1) * tokenizer_->vocab_size();
      return sampler_->sample(logits_last);
    }
    default:
      ET_CHECK_MSG(
          false,
          "Unsupported dtype output %hhd",
          static_cast<int8_t>(logits_tensor.scalar_type()));
  }
}

Result<int64_t> Runner::prefill(
    const std::vector<uint64_t>& prompt_tokens,
    int64_t start_pos,
    std::function<void(const std::string&)> token_callback) {
  // enable_parallel_prefill_ maybe set even when not using kv cache
  // When kv cache is not used, start pos is ignored
  std::vector<int64_t> tokens;
  for (uint64_t tok : prompt_tokens) {
    tokens.push_back(tok);
  }
  int32_t num_prompt_tokens = prompt_tokens.size();

  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      num_prompt_tokens < max_seq_len_,
      "Max seq length exceeded - please increase max seq len value");

  // store the token
  int64_t cur_token;
  if (enable_parallel_prefill_ || !use_kv_cache_) {
    // initialize tensor wrappers
    ManagedTensor managed_tokens(
        tokens.data(), {1, num_prompt_tokens}, ScalarType::Long);

    ManagedTensor managed_start_pos(&start_pos, {1}, ScalarType::Long);

    Result<torch::executor::Tensor> outputs_res =
        run_model_step(managed_tokens, managed_start_pos);

    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_LOG(
        Info, "Prefill token result numel(): %zu", outputs_res.get().numel());
    ET_CHECK_MSG(
        outputs_res.get().size(1) == num_prompt_tokens,
        "Expected number of output tokens %d does not match returned value %zu.",
        num_prompt_tokens,
        outputs_res.get().size(1));
    // insert new token into prompt_tokens
    uint64_t prev = prompt_tokens[0];
    uint64_t cur;
    for (int i = 1; i < prompt_tokens.size(); i++) {
      cur = prompt_tokens[i];
      _DECODE_PRINT_CALLBACK(prev, cur, token_callback);
      prev = cur;
    }
    cur_token = logitsToToken(outputs_res.get());
  } else { // sequential prefill
    int64_t pos = 0; // position in the sequence
    int64_t prev_token;
    // token & pos
    int64_t pos_data = 0;
    cur_token = prompt_tokens[0];
    std::vector<int64_t> token_vec = {
        cur_token}; // allocate space for the tokens

    // initialize tensor wrappers
    ManagedTensor managed_tokens(token_vec.data(), {1, 1}, ScalarType::Long);

    ManagedTensor managed_start_pos(&pos_data, {1}, ScalarType::Long);

    while (pos < num_prompt_tokens) {
      // Run the model
      pos_data = start_pos + pos;

      Result<torch::executor::Tensor> logits_res =
          run_model_step(managed_tokens, managed_start_pos);

      ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
      prev_token = cur_token;

      long sample_start_time_ms = util::time_in_ms();
      stats_.aggregate_sampling_time_ms +=
          util::time_in_ms() - sample_start_time_ms;

      pos++;

      cur_token = pos == num_prompt_tokens ? logitsToToken(logits_res.get())
                                           : prompt_tokens[pos];

      token_vec[0] = cur_token;

      // print the token as string, decode it with the Tokenizer object
      _DECODE_PRINT_CALLBACK(prev_token, cur_token, token_callback);
    }
  }
  // Return the next token
  stats_.first_token_ms = util::time_in_ms();
  stats_.prompt_eval_end_ms = util::time_in_ms();
  return cur_token;
}

// Given an input token. Set up the inputs for the model and execute a single
// step. Returning the logits tensor.
Result<torch::executor::Tensor> Runner::run_model_step(
    ManagedTensor& managed_tokens,
    ManagedTensor& managed_start_pos) {
  // ET_LOG(Info, "Input token %" PRIu64, input_token);
  auto tokens = managed_tokens.get_aliasing_tensor();
  if (use_kv_cache_) {
    auto start_pos = managed_start_pos.get_aliasing_tensor();

    Result<std::vector<EValue>> outputs_res =
        module_->forward({tokens, start_pos});
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More then one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    // Return the logits tensor
    return outputs_res.get()[0].toTensor();
  } else { // no kv cache
    std::vector<EValue> inputs;
    (void)managed_start_pos; // unused

    Result<std::vector<EValue>> outputs_res = module_->forward(inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res.get().size() == 1,
        "More then one output returned from executing LLM.");
    ET_CHECK_MSG(
        outputs_res.get()[0].isTensor(),
        "Non Tensor Output returned from executing LLM");

    // Return the logits tensor
    return outputs_res.get()[0].toTensor();
  }
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
      "Max seq length exceeded - please increase max seq len value in .../llama2/model.py");

  ET_CHECK_MSG(
      num_prompt_tokens < seq_len,
      "Sequence length exceeded - please increase the seq_len value passed to generate()");

  // Prefill first
  // Here feed all tokens to the model and get the next predicted token
  // after the prompt. After that we will enter generate loop.
  auto prefill_res = prefill(prompt_tokens, 0, token_callback);
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  int64_t cur_token = prefill_res.get();

  // print the first token from prefill. No prev_token so use cur_token for it.
  _DECODE_PRINT_CALLBACK(cur_token, cur_token, token_callback);

  // start the main loop
  int64_t pos = num_prompt_tokens; // position in the sequence

  // Generate the rest of the sequence
  std::vector<int64_t> token_data; // allocate space for the tokens
  std::vector<exec_aten::SizesType> token_shape;

  if (use_kv_cache_) {
    // hard code these to size 1 as kv cache is locked to static size right now.
    token_data = {cur_token};
    token_shape = {1, 1};
  } else {
    for (auto tok : prompt_tokens) {
      token_data.push_back(tok);
    }
    token_data.push_back(cur_token);
    token_shape = {1, num_prompt_tokens + 1};
  }

  // initialize tensor wrappers
  ManagedTensor tokens_managed(
      token_data.data(), token_shape, ScalarType::Long);

  ManagedTensor start_pos_managed(&pos, {1}, ScalarType::Long);

  int64_t prev_token;

  // Generate our tokens
  while (pos < seq_len - 1) {
    // Run the model
    Result<torch::executor::Tensor> logits_res =
        run_model_step(tokens_managed, start_pos_managed);

    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    exec_aten::Tensor& logits_tensor = logits_res.get();

    prev_token = cur_token;

    long sample_start_time_ms = util::time_in_ms();
    cur_token = logitsToToken(logits_tensor);
    stats_.aggregate_sampling_time_ms +=
        util::time_in_ms() - sample_start_time_ms;

    pos++;

    if (use_kv_cache_) {
      // update the token tensor. token_data will not be empty.
      // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
      token_data[0] = cur_token;
    } else {
      // push it to the back
      token_data.push_back(cur_token);
      tokens_managed.resize({1, static_cast<int>(token_data.size())});
    }

    // print the token as string, decode it with the Tokenizer object
    _DECODE_PRINT_CALLBACK(prev_token, cur_token, token_callback);

    if (shouldStop_) {
      break;
    }

    // data-dependent terminating condition: we have n_eos_ number of EOS
    if (pos >= num_prompt_tokens && cur_token == eos_id_) {
      printf("\n");
      ET_LOG(Info, "\nReached to the end of generation");
      break;
    }
  }
  stats_.inference_end_ms = util::time_in_ms();
  printf("\n");

  if (pos == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = pos - num_prompt_tokens;
  ::executorch::llm::print_report(stats_);
  if (stats_callback) {
    stats_callback(stats_);
  }

  return Error::Ok;
}

void Runner::stop() {
  shouldStop_ = true;
}
} // namespace torch::executor
