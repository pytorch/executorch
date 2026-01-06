/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/apple/coreml/llama/runner/static_llm_runner.h>

#include <fstream>
#include <nlohmann/json.hpp>

#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/tiktoken.h>

namespace example {

using namespace ::executorch::extension;
using namespace ::executorch::runtime;

namespace {

std::unique_ptr<::tokenizers::Tokenizer> load_tokenizer(
    const std::string& tokenizer_path) {
  auto hf_tokenizer = std::make_unique<::tokenizers::HFTokenizer>();
  if (hf_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded HuggingFace tokenizer");
    return hf_tokenizer;
  }

  auto tiktoken = std::make_unique<::tokenizers::Tiktoken>();
  if (tiktoken->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded TikToken tokenizer");
    return tiktoken;
  }

  auto sp_tokenizer = std::make_unique<::tokenizers::SPTokenizer>();
  if (sp_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded SentencePiece tokenizer");
    return sp_tokenizer;
  }

  auto bpe_tokenizer = std::make_unique<::tokenizers::Llama2cTokenizer>();
  if (bpe_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded BPE tokenizer");
    return bpe_tokenizer;
  }

  ET_LOG(Error, "Failed to load tokenizer from %s", tokenizer_path.c_str());
  return nullptr;
}

} // namespace

// ============================================================================
// StaticLLMIOManager implementation
// ============================================================================

StaticLLMIOManager::StaticLLMIOManager(
    Module& module,
    const StaticLLMConfig& config)
    : IOManager(module), module_(module), config_(config) {
  compute_rope_frequencies();

  input_buffer_.resize(config_.input_len, 0);

  size_t logits_size = config_.input_len * config_.vocab_size;
  logits_output_.resize(logits_size, static_cast<LogitT>(0));

  size_t update_size =
      config_.n_kv_heads * config_.input_len * config_.head_dim;
  k_update_buffers_.resize(config_.n_layers);
  v_update_buffers_.resize(config_.n_layers);
  for (size_t i = 0; i < config_.n_layers; i++) {
    k_update_buffers_[i].resize(update_size, static_cast<CacheT>(0));
    v_update_buffers_[i].resize(update_size, static_cast<CacheT>(0));
  }
}

void StaticLLMIOManager::compute_rope_frequencies() {
  size_t max_seq_len = config_.cache_len + config_.input_len;
  size_t rope_dim = config_.head_dim / 2;

  freqs_cos_.resize(max_seq_len * rope_dim);
  freqs_sin_.resize(max_seq_len * rope_dim);

  for (size_t pos = 0; pos < max_seq_len; pos++) {
    for (size_t i = 0; i < rope_dim; i++) {
      float freq =
          1.0f /
          std::pow(
              config_.rope_base, static_cast<float>(2 * i) / config_.head_dim);
      float angle = static_cast<float>(pos) * freq;
      freqs_cos_[pos * rope_dim + i] = static_cast<RopeT>(std::cos(angle));
      freqs_sin_[pos * rope_dim + i] = static_cast<RopeT>(std::sin(angle));
    }
  }
}

Error StaticLLMIOManager::load(
    const std::string& prefill_method,
    const std::string& decode_method) {
  (void)prefill_method;
  (void)decode_method;

  // Build input/output indices for StaticAttentionIOManager
  std::vector<size_t> k_cache_input_indices(config_.n_layers);
  std::vector<size_t> k_cache_output_indices(config_.n_layers);
  std::vector<size_t> v_cache_input_indices(config_.n_layers);
  std::vector<size_t> v_cache_output_indices(config_.n_layers);

  for (size_t i = 0; i < config_.n_layers; i++) {
    k_cache_input_indices[i] = 4 + i;
    k_cache_output_indices[i] = 1 + i;
    v_cache_input_indices[i] = 4 + config_.n_layers + i;
    v_cache_output_indices[i] = 1 + config_.n_layers + i;
  }

  typename StaticAttentionIOManager<CacheT, MaskT, RopeT>::StaticAttentionIOConfig
      io_config;
  io_config.n_caches = config_.n_layers;
  io_config.cache_lengths =
      std::vector<size_t>(config_.n_layers, config_.cache_len);
  io_config.head_dim = config_.head_dim;
  io_config.max_input_len = config_.input_len;
  io_config.n_heads_per_cache = config_.n_kv_heads;
  io_config.cache_len_to_mask_idx = {{config_.cache_len, 1}};
  io_config.rope_freqs_cos_input_index = 2;
  io_config.rope_freqs_sin_input_index = 3;
  io_config.k_cache_input_indices = k_cache_input_indices;
  io_config.k_cache_output_indices = k_cache_output_indices;
  io_config.v_cache_input_indices = v_cache_input_indices;
  io_config.v_cache_output_indices = v_cache_output_indices;
  io_config.max_context_len = config_.cache_len + config_.input_len;
  io_config.rope_freqs_cos = freqs_cos_.data();
  io_config.rope_freqs_sin = freqs_sin_.data();
  io_config.style = StaticAttentionUpdateStyle::SMART_MASK;
  io_config.generate_full_logits = config_.generate_full_logits;
  io_config.last_valid_token_pos_index = std::nullopt;

  static_io_manager_ =
      std::make_unique<StaticAttentionIOManager<CacheT, MaskT, RopeT>>(
          std::move(io_config));

  MaskT zero_val = static_cast<MaskT>(0.0f);
  MaskT mask_val = static_cast<MaskT>(-65504.0f);
  static_io_manager_->add_mask(config_.input_len, zero_val, mask_val);

  return Error::Ok;
}

Error StaticLLMIOManager::reset(
    const std::string& prefill_method,
    const std::string& decode_method) {
  (void)prefill_method;
  (void)decode_method;
  if (static_io_manager_) {
    static_io_manager_->reset();
  }
  output_buffers_set_ = false;
  return Error::Ok;
}

void StaticLLMIOManager::setup_output_buffers(Method& method) {
  if (output_buffers_set_) {
    return;
  }

  auto method_meta = method.method_meta();

  auto logits_meta = method_meta.output_tensor_meta(0);
  ET_CHECK_MSG(logits_meta.ok(), "Failed to get logits output meta");
  ET_CHECK(
      method.set_output_data_ptr(
          logits_output_.data(), logits_meta->nbytes(), 0) == Error::Ok);

  for (size_t i = 0; i < config_.n_layers; i++) {
    auto k_out_meta = method_meta.output_tensor_meta(1 + i);
    ET_CHECK_MSG(
        k_out_meta.ok(), "Failed to get k_cache output meta for layer %zu", i);
    ET_CHECK(
        method.set_output_data_ptr(
            k_update_buffers_[i].data(), k_out_meta->nbytes(), 1 + i) ==
        Error::Ok);
  }

  for (size_t i = 0; i < config_.n_layers; i++) {
    auto v_out_meta = method_meta.output_tensor_meta(1 + config_.n_layers + i);
    ET_CHECK_MSG(
        v_out_meta.ok(), "Failed to get v_cache output meta for layer %zu", i);
    ET_CHECK(
        method.set_output_data_ptr(
            v_update_buffers_[i].data(),
            v_out_meta->nbytes(),
            1 + config_.n_layers + i) == Error::Ok);
  }

  output_buffers_set_ = true;
}

Result<std::vector<EValue>> StaticLLMIOManager::prepare_prefill(
    const TensorPtr& input,
    const TensorPtr& start_pos,
    const std::string& prefill_method) {
  (void)start_pos;
  (void)prefill_method;

  // Copy tokens to input buffer
  const int64_t* input_data = input->const_data_ptr<int64_t>();
  actual_input_len_ = input->numel();
  for (size_t i = 0; i < config_.input_len; i++) {
    input_buffer_[i] = (i < actual_input_len_)
        ? static_cast<TokenT>(input_data[i])
        : 0;
  }

  // Return empty - inputs are set via Method::set_input by StaticAttentionIOManager
  return std::vector<EValue>{};
}

Result<std::vector<EValue>> StaticLLMIOManager::prepare_decode(
    const TensorPtr& input,
    const TensorPtr& start_pos,
    const std::string& decode_method) {
  (void)start_pos;
  (void)decode_method;

  const int64_t* input_data = input->const_data_ptr<int64_t>();
  actual_input_len_ = 1;
  input_buffer_[0] = static_cast<TokenT>(input_data[0]);
  for (size_t i = 1; i < config_.input_len; i++) {
    input_buffer_[i] = 0;
  }

  return std::vector<EValue>{};
}

Error StaticLLMIOManager::update_prefill(
    const std::vector<EValue>& model_outputs,
    const std::string& prefill_method) {
  (void)model_outputs;
  (void)prefill_method;
  // KV cache update is handled by StaticAttentionIOManager
  return Error::Ok;
}

Error StaticLLMIOManager::update_decode(
    const std::vector<EValue>& model_outputs,
    const std::string& decode_method) {
  (void)model_outputs;
  (void)decode_method;
  return Error::Ok;
}

// ============================================================================
// StaticLLMTextDecoderRunner implementation
// ============================================================================

StaticLLMTextDecoderRunner::StaticLLMTextDecoderRunner(
    Module* module,
    StaticLLMIOManager* io_manager)
    : TextDecoderRunner(module, io_manager), static_io_manager_(io_manager) {}

Error StaticLLMTextDecoderRunner::load() {
  ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method("forward"));
  ET_CHECK_OK_OR_RETURN_ERROR(io_manager_->load());
  return Error::Ok;
}

Result<executorch::aten::Tensor> StaticLLMTextDecoderRunner::step(
    TensorPtr& input,
    int64_t start_pos) {
  (void)start_pos;

  auto method_result = module_->method("forward");
  if (!method_result.ok()) {
    return method_result.error();
  }
  Method* method = method_result.get();

  // Set up output buffers for CoreML
  static_io_manager_->setup_output_buffers(*method);

  // Get the underlying StaticAttentionIOManager
  auto* static_io = static_io_manager_->get_static_io_manager();

  // Set token input
  auto input_meta = method->method_meta().input_tensor_meta(0);
  ET_CHECK_MSG(input_meta.ok(), "Failed to get input tensor meta");

  const int64_t* input_data = input->const_data_ptr<int64_t>();
  std::vector<StaticLLMIOManager::TokenT> tokens(
      static_io_manager_->config().input_len, 0);
  tokens[0] = static_cast<StaticLLMIOManager::TokenT>(input_data[0]);

  auto input_impl = ::executorch::runtime::etensor::TensorImpl(
      input_meta->scalar_type(),
      input_meta->sizes().size(),
      const_cast<executorch::aten::TensorImpl::SizesType*>(
          input_meta->sizes().data()),
      tokens.data(),
      const_cast<executorch::aten::TensorImpl::DimOrderType*>(
          input_meta->dim_order().data()));
  executorch::runtime::etensor::Tensor input_tensor(&input_impl);
  ET_CHECK(method->set_input(input_tensor, 0) == Error::Ok);

  // Set up mask and RoPE via StaticAttentionIOManager
  auto& masks = static_io->get_mask(static_io_manager_->config().input_len);
  for (auto& pair : masks) {
    auto& mask = *pair.second;
    mask.set_causal_mask();

    auto mask_meta = method->method_meta().input_tensor_meta(1);
    ET_CHECK_MSG(mask_meta.ok(), "Failed to get mask tensor meta");
    auto mask_impl = ::executorch::runtime::etensor::TensorImpl(
        mask_meta->scalar_type(),
        mask_meta->sizes().size(),
        const_cast<executorch::aten::TensorImpl::SizesType*>(
            mask_meta->sizes().data()),
        mask.get(),
        const_cast<executorch::aten::TensorImpl::DimOrderType*>(
            mask_meta->dim_order().data()));
    executorch::runtime::etensor::Tensor mask_tensor(&mask_impl);
    ET_CHECK(method->set_input(mask_tensor, 1) == Error::Ok);
  }

  static_io->prepare(*method);

  auto exec_result = method->execute();
  if (exec_result != Error::Ok) {
    return exec_result;
  }

  // Update KV caches
  const auto& config = static_io_manager_->config();
  std::vector<size_t> k_out_indices(config.n_layers);
  std::vector<size_t> v_out_indices(config.n_layers);
  for (size_t i = 0; i < config.n_layers; i++) {
    k_out_indices[i] = 1 + i;
    v_out_indices[i] = 1 + config.n_layers + i;
  }
  static_io->update(*method, k_out_indices, v_out_indices, 1);

  return method->get_output(0).toTensor();
}

// ============================================================================
// StaticLLMRunner implementation
// ============================================================================

StaticLLMRunner::StaticLLMRunner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const StaticLLMConfig& config)
    : model_path_(model_path),
      tokenizer_path_(tokenizer_path),
      config_(config) {
  runtime_init();
}

Error StaticLLMRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }

  stats_.model_load_start_ms = llm::time_in_ms();

  ET_LOG(Info, "Loading model from %s", model_path_.c_str());
  module_ = std::make_unique<Module>(model_path_, Module::LoadMode::File);

  ET_LOG(Info, "Loading tokenizer from %s", tokenizer_path_.c_str());
  tokenizer_ = load_tokenizer(tokenizer_path_);
  if (!tokenizer_) {
    return Error::InvalidArgument;
  }

  eos_ids_.insert(tokenizer_->eos_tok());
  eos_ids_.insert(128001);
  eos_ids_.insert(128009);

  io_manager_ = std::make_unique<StaticLLMIOManager>(*module_, config_);
  decoder_runner_ =
      std::make_unique<StaticLLMTextDecoderRunner>(module_.get(), io_manager_.get());

  ET_CHECK_OK_OR_RETURN_ERROR(decoder_runner_->load());

  stats_.model_load_end_ms = llm::time_in_ms();
  ET_LOG(
      Info,
      "Model loaded in %.2f seconds",
      (stats_.model_load_end_ms - stats_.model_load_start_ms) / 1000.0);

  return Error::Ok;
}

bool StaticLLMRunner::is_loaded() const {
  return decoder_runner_ && decoder_runner_->is_method_loaded();
}

void StaticLLMRunner::reset() {
  if (io_manager_) {
    io_manager_->reset("forward", "forward");
  }
  stats_.reset();
}

StaticLLMRunner::TokenT StaticLLMRunner::sample_token(
    Method& method,
    size_t pos) {
  auto logits_tensor = method.get_output(0).toTensor();
  size_t vocab_size = logits_tensor.size(logits_tensor.dim() - 1);
  const LogitT* logits_data = logits_tensor.const_data_ptr<LogitT>();

  size_t offset = pos * vocab_size;

  LogitT max_val = logits_data[offset];
  TokenT max_idx = 0;
  for (size_t i = 1; i < vocab_size; i++) {
    if (logits_data[offset + i] > max_val) {
      max_val = logits_data[offset + i];
      max_idx = static_cast<TokenT>(i);
    }
  }
  return max_idx;
}

std::vector<StaticLLMRunner::TokenT> StaticLLMRunner::sample_all_tokens(
    Method& method) {
  auto logits_tensor = method.get_output(0).toTensor();
  size_t vocab_size = logits_tensor.size(logits_tensor.dim() - 1);
  size_t seq_len = logits_tensor.size(logits_tensor.dim() - 2);
  const LogitT* logits_data = logits_tensor.const_data_ptr<LogitT>();

  std::vector<TokenT> tokens(seq_len);
  for (size_t pos = 0; pos < seq_len; pos++) {
    size_t offset = pos * vocab_size;
    LogitT max_val = logits_data[offset];
    TokenT max_idx = 0;
    for (size_t i = 1; i < vocab_size; i++) {
      if (logits_data[offset + i] > max_val) {
        max_val = logits_data[offset + i];
        max_idx = static_cast<TokenT>(i);
      }
    }
    tokens[pos] = max_idx;
  }
  return tokens;
}

Error StaticLLMRunner::generate(
    const std::string& prompt,
    int32_t max_new_tokens,
    float temperature,
    std::function<void(const std::string&)> token_callback) {
  (void)temperature;

  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  reset();
  stats_.inference_start_ms = llm::time_in_ms();

  auto encode_result = tokenizer_->encode(prompt, 1, 0);
  if (!encode_result.ok()) {
    return Error::InvalidArgument;
  }

  std::vector<uint64_t> prompt_tokens_u64 = encode_result.get();
  std::vector<TokenT> prompt_tokens(
      prompt_tokens_u64.begin(), prompt_tokens_u64.end());
  size_t num_prompt_tokens = prompt_tokens.size();

  ET_LOG(Info, "Prompt: %s", prompt.c_str());
  ET_LOG(Info, "Prompt tokens: %zu", num_prompt_tokens);

  auto method_result = module_->method("forward");
  if (!method_result.ok()) {
    return method_result.error();
  }
  Method* method = method_result.get();

  io_manager_->setup_output_buffers(*method);

  auto* static_io = io_manager_->get_static_io_manager();
  std::vector<TokenT> input_buffer(config_.input_len, 0);

  Span<TokenT> prompt_span(prompt_tokens.data(), prompt_tokens.size());
  Span<TokenT> input_span(input_buffer.data(), input_buffer.size());

  auto input_meta = method->method_meta().input_tensor_meta(0);
  ET_CHECK_MSG(input_meta.ok(), "Failed to get input tensor meta");
  auto input_impl = ::executorch::runtime::etensor::TensorImpl(
      input_meta->scalar_type(),
      input_meta->sizes().size(),
      const_cast<executorch::aten::TensorImpl::SizesType*>(
          input_meta->sizes().data()),
      input_buffer.data(),
      const_cast<executorch::aten::TensorImpl::DimOrderType*>(
          input_meta->dim_order().data()));
  executorch::runtime::etensor::Tensor input_tensor(&input_impl);
  ET_CHECK(method->set_input(input_tensor, 0) == Error::Ok);

  size_t last_logit_pos = static_io->prefill<TokenT, LogitT>(
      prompt_span, input_span, *method, nullptr);

  TokenT cur_token = sample_token(*method, last_logit_pos);

  stats_.first_token_ms = llm::time_in_ms();
  stats_.prompt_eval_end_ms = llm::time_in_ms();

  auto decode_result = tokenizer_->decode(cur_token, cur_token);
  if (decode_result.ok() && token_callback) {
    token_callback(*decode_result);
  }

  TokenT prev_token = cur_token;
  std::function<TokenT(Method&)> sample_fn = [this](Method& m) -> TokenT {
    return sample_token(m, 0);
  };

  int32_t num_generated = 1;
  std::function<bool(TokenT)> token_cb = [&](TokenT tok) -> bool {
    num_generated++;
    if (num_generated > max_new_tokens) {
      return false;
    }
    if (eos_ids_.find(tok) != eos_ids_.end()) {
      return false;
    }
    auto decode = tokenizer_->decode(prev_token, tok);
    if (decode.ok() && token_callback) {
      token_callback(*decode);
    }
    prev_token = tok;
    return true;
  };

  static_io->decode<TokenT>(prev_token, input_span, *method, sample_fn, token_cb);

  stats_.inference_end_ms = llm::time_in_ms();
  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = num_generated;

  double prefill_time_s =
      (stats_.first_token_ms - stats_.inference_start_ms) / 1000.0;
  double decode_time_s =
      (stats_.inference_end_ms - stats_.first_token_ms) / 1000.0;
  double tokens_per_sec =
      decode_time_s > 0 ? (num_generated - 1) / decode_time_s : 0;

  ET_LOG(
      Info,
      "\nPrefill: %zu tokens in %.2f s",
      num_prompt_tokens,
      prefill_time_s);
  ET_LOG(
      Info,
      "Decode: %d tokens in %.2f s (%.2f tok/s)",
      num_generated,
      decode_time_s,
      tokens_per_sec);

  return Error::Ok;
}

Error StaticLLMRunner::generate_with_lookahead(
    const std::string& prompt,
    int32_t max_new_tokens,
    const LookaheadConfig& lookahead_config,
    std::function<void(const std::string&)> token_callback) {
  if (!config_.generate_full_logits) {
    ET_LOG(
        Error,
        "Lookahead decoding requires generate_full_logits=true, but model "
        "outputs only last token logits");
    return Error::InvalidArgument;
  }

  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  reset();
  stats_.inference_start_ms = llm::time_in_ms();

  size_t ngram_size = lookahead_config.ngram_size;
  size_t window_size = lookahead_config.window_size;
  size_t n_verifications = lookahead_config.n_verifications;

  ET_LOG(
      Info,
      "Using lookahead decoding: ngram=%zu, window=%zu, verifications=%zu",
      ngram_size,
      window_size,
      n_verifications);

  auto encode_result = tokenizer_->encode(prompt, 1, 0);
  if (!encode_result.ok()) {
    return Error::InvalidArgument;
  }

  std::vector<uint64_t> prompt_tokens_u64 = encode_result.get();
  std::vector<TokenT> prompt_tokens(
      prompt_tokens_u64.begin(), prompt_tokens_u64.end());
  size_t num_prompt_tokens = prompt_tokens.size();

  ET_LOG(Info, "Prompt: %s", prompt.c_str());
  ET_LOG(Info, "Prompt tokens: %zu", num_prompt_tokens);

  auto method_result = module_->method("forward");
  if (!method_result.ok()) {
    return method_result.error();
  }
  Method* method = method_result.get();

  io_manager_->setup_output_buffers(*method);

  auto* static_io = io_manager_->get_static_io_manager();
  std::vector<TokenT> input_buffer(config_.input_len, 0);

  Span<TokenT> prompt_span(prompt_tokens.data(), prompt_tokens.size());
  Span<TokenT> input_span(input_buffer.data(), input_buffer.size());

  auto input_meta = method->method_meta().input_tensor_meta(0);
  ET_CHECK_MSG(input_meta.ok(), "Failed to get input tensor meta");
  auto input_impl = ::executorch::runtime::etensor::TensorImpl(
      input_meta->scalar_type(),
      input_meta->sizes().size(),
      const_cast<executorch::aten::TensorImpl::SizesType*>(
          input_meta->sizes().data()),
      input_buffer.data(),
      const_cast<executorch::aten::TensorImpl::DimOrderType*>(
          input_meta->dim_order().data()));
  executorch::runtime::etensor::Tensor input_tensor(&input_impl);
  ET_CHECK(method->set_input(input_tensor, 0) == Error::Ok);

  size_t last_logit_pos = static_io->prefill<TokenT, LogitT>(
      prompt_span, input_span, *method, nullptr);

  TokenT cur_token = sample_token(*method, last_logit_pos);

  stats_.first_token_ms = llm::time_in_ms();
  stats_.prompt_eval_end_ms = llm::time_in_ms();

  auto decode_result = tokenizer_->decode(cur_token, cur_token);
  if (decode_result.ok() && token_callback) {
    token_callback(*decode_result);
  }

  std::unordered_map<TokenT, SuffixCache<TokenT>> suffix_caches;
  SuffixCache<TokenT>::seed_suffix_caches(
      suffix_caches, prompt_span, ngram_size, n_verifications);

  TokenT prev_token = cur_token;
  std::function<std::vector<TokenT>(Method&)> sample_all_fn =
      [this](Method& m) -> std::vector<TokenT> { return sample_all_tokens(m); };

  int32_t num_generated = 1;
  std::function<bool(TokenT)> token_cb = [&](TokenT tok) -> bool {
    num_generated++;
    if (num_generated > max_new_tokens) {
      return false;
    }
    if (eos_ids_.find(tok) != eos_ids_.end()) {
      return false;
    }
    auto decode = tokenizer_->decode(prev_token, tok);
    if (decode.ok() && token_callback) {
      token_callback(*decode);
    }
    prev_token = tok;
    return true;
  };

  static_io->lookahead_decode<TokenT>(
      prev_token,
      input_span,
      *method,
      sample_all_fn,
      token_cb,
      ngram_size,
      window_size,
      n_verifications,
      std::move(suffix_caches));

  stats_.inference_end_ms = llm::time_in_ms();
  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = num_generated;

  double prefill_time_s =
      (stats_.first_token_ms - stats_.inference_start_ms) / 1000.0;
  double decode_time_s =
      (stats_.inference_end_ms - stats_.first_token_ms) / 1000.0;
  double tokens_per_sec =
      decode_time_s > 0 ? (num_generated - 1) / decode_time_s : 0;

  ET_LOG(
      Info,
      "\nPrefill: %zu tokens in %.2f s",
      num_prompt_tokens,
      prefill_time_s);
  ET_LOG(
      Info,
      "Decode: %d tokens in %.2f s (%.2f tok/s)",
      num_generated,
      decode_time_s,
      tokens_per_sec);

  return Error::Ok;
}

std::unique_ptr<StaticLLMRunner> create_static_llm_runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& params_path) {
  // Load model to extract metadata
  Module module(model_path, Module::LoadMode::File);
  auto load_result = module.load_method("forward");
  if (load_result != Error::Ok) {
    ET_LOG(Error, "Failed to load model method: %s", model_path.c_str());
    return nullptr;
  }

  auto method_meta = module.method_meta("forward");
  if (!method_meta.ok()) {
    ET_LOG(Error, "Failed to get method metadata");
    return nullptr;
  }

  // Extract input_len and cache_len from mask tensor shape
  // Mask shape is (1, input_len, cache_len + input_len)
  auto mask_meta = method_meta->input_tensor_meta(1);
  if (!mask_meta.ok()) {
    ET_LOG(Error, "Failed to get mask tensor metadata");
    return nullptr;
  }

  auto mask_sizes = mask_meta->sizes();
  if (mask_sizes.size() != 3) {
    ET_LOG(
        Error,
        "Expected mask tensor to have 3 dimensions, got %zu",
        mask_sizes.size());
    return nullptr;
  }

  size_t input_len = mask_sizes[1];
  size_t total_len = mask_sizes[2];
  size_t cache_len = total_len - input_len;

  // Extract n_layers from number of k_cache inputs
  // Inputs: tokens(0), mask(1), freqs_cos(2), freqs_sin(3), k_caches..., v_caches...
  size_t num_inputs = method_meta->num_inputs();
  size_t n_layers = (num_inputs - 4) / 2;

  // Extract n_kv_heads and head_dim from k_cache shape
  // k_cache shape is (1, n_kv_heads, cache_len, head_dim)
  auto k_cache_meta = method_meta->input_tensor_meta(4);
  if (!k_cache_meta.ok()) {
    ET_LOG(Error, "Failed to get k_cache tensor metadata");
    return nullptr;
  }

  auto k_cache_sizes = k_cache_meta->sizes();
  if (k_cache_sizes.size() != 4) {
    ET_LOG(
        Error,
        "Expected k_cache tensor to have 4 dimensions, got %zu",
        k_cache_sizes.size());
    return nullptr;
  }

  size_t n_kv_heads = k_cache_sizes[1];
  size_t head_dim = k_cache_sizes[3];

  // Extract vocab_size and generate_full_logits from logits output shape
  // Full logits shape: (1, input_len, vocab_size)
  // Last token only: (1, 1, vocab_size)
  auto logits_meta = method_meta->output_tensor_meta(0);
  if (!logits_meta.ok()) {
    ET_LOG(Error, "Failed to get logits tensor metadata");
    return nullptr;
  }

  auto logits_sizes = logits_meta->sizes();
  size_t vocab_size = logits_sizes[logits_sizes.size() - 1];
  size_t logits_seq_len = logits_sizes[logits_sizes.size() - 2];
  bool generate_full_logits = (logits_seq_len == input_len);

  // Read params.json for any additional config (rope_base, etc.)
  float rope_base = 500000.0f;
  std::ifstream params_file(params_path);
  if (params_file.is_open()) {
    try {
      nlohmann::json params;
      params_file >> params;
      rope_base = params.value("rope_theta", 500000.0f);
    } catch (const std::exception& e) {
      ET_LOG(Info, "Could not parse params.json, using defaults: %s", e.what());
    }
  }

  StaticLLMConfig config;
  config.n_layers = n_layers;
  config.n_kv_heads = n_kv_heads;
  config.head_dim = head_dim;
  config.vocab_size = vocab_size;
  config.input_len = input_len;
  config.cache_len = cache_len;
  config.generate_full_logits = generate_full_logits;
  config.rope_base = rope_base;

  ET_LOG(
      Info,
      "Config from model metadata: n_layers=%zu, n_kv_heads=%zu, head_dim=%zu, "
      "input_len=%zu, cache_len=%zu, vocab_size=%zu, full_logits=%s",
      config.n_layers,
      config.n_kv_heads,
      config.head_dim,
      config.input_len,
      config.cache_len,
      config.vocab_size,
      config.generate_full_logits ? "true" : "false");

  return std::make_unique<StaticLLMRunner>(model_path, tokenizer_path, config);
}

} // namespace example
