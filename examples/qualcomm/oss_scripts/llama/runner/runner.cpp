/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A llama 3.2 runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llama/runner/runner.h>
#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/client_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/lhd_token_generator.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <algorithm>
#include <fstream>

using executorch::extension::Module;
using executorch::extension::llm::get_rss_bytes;
using executorch::extension::llm::print_report;
using executorch::extension::llm::Stats;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
namespace llm = ::executorch::extension::llm;

namespace example {
namespace {
void print_performance_report(
    const Stats& stats,
    const std::string& performance_output_path) {
  // For now, we just print the total inference time for CI, can save more info
  // in future if needed.
  std::ofstream outfile(performance_output_path.c_str());
  if (outfile.is_open()) {
    double num_tok = 0;
    if (stats.num_generated_tokens == 0) {
      // For cases like evaluate perplexity where prompt_len == cache_len
      num_tok = ((stats.num_prompt_tokens)) /
          (double)(stats.prompt_eval_end_ms - stats.inference_start_ms) *
          stats.SCALING_FACTOR_UNITS_PER_SECOND;
    } else {
      num_tok = (stats.num_generated_tokens) /
          (double)(stats.inference_end_ms - stats.inference_start_ms) *
          stats.SCALING_FACTOR_UNITS_PER_SECOND;
    }

    outfile << num_tok;
    outfile.close();
  } else {
    ET_LOG(Error, "Error saving the inference speed file");
  }
}

void save_logits(
    const std::string& dump_logits_path,
    const std::vector<uint16_t>& prefill_logits,
    const std::vector<uint16_t>& decode_logits) {
  std::ofstream outFile(dump_logits_path.c_str(), std::ios::binary);
  if (outFile.is_open()) {
    outFile.write(
        reinterpret_cast<const char*>(prefill_logits.data()),
        prefill_logits.size() * sizeof(uint16_t));

    outFile.write(
        reinterpret_cast<const char*>(decode_logits.data()),
        decode_logits.size() * sizeof(uint16_t));
    outFile.close();
  } else {
    ET_CHECK_MSG(false, "Error saving the dump logits file");
  }
}

} // namespace

template <typename T>
Runner<T>::Runner(
    std::unique_ptr<executorch::extension::Module> module,
    const std::string& decoder_model_version,
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& dump_logits_path,
    const std::string& performance_output_path,
    const float temperature,
    const int eval_mode,
    const bool shared_buffer,
    const int ngram,
    const int window,
    const int gcap,
    std::unique_ptr<tokenizers::Tokenizer> tokenizer)
    : module_(std::move(module)),
      ngram_(ngram),
      window_(window),
      gcap_(gcap),
      tokenizer_path_(tokenizer_path),
      performance_output_path_(performance_output_path),
      dump_logits_path_(dump_logits_path),
      temperature_(temperature),
      eval_mode_(static_cast<EvalMode>(eval_mode)),
      shared_buffer_(shared_buffer),
      tokenizer_(std::move(tokenizer)) {
  stats_.reset();

  if (decoder_model_version == "llama2") {
    decoder_model_version_ = DecoderModelVersion::kLlama2;
  } else if (decoder_model_version == "llama3") {
    decoder_model_version_ = DecoderModelVersion::kLlama3;
  } else if (decoder_model_version == "gemma") {
    decoder_model_version_ = DecoderModelVersion::kGemma;
  } else if (decoder_model_version == "gemma2") {
    decoder_model_version_ = DecoderModelVersion::kGemma2;
    cache_mode_ = CacheMode::HybridCache;
  } else if (decoder_model_version == "gemma3") {
    decoder_model_version_ = DecoderModelVersion::kGemma3;
    cache_mode_ = CacheMode::HybridCache;
  } else if (decoder_model_version == "granite") {
    decoder_model_version_ = DecoderModelVersion::kGranite;
  } else if (decoder_model_version == "phi_4_mini") {
    decoder_model_version_ = DecoderModelVersion::kPhi4;
  } else if (decoder_model_version == "qwen2_5") {
    decoder_model_version_ = DecoderModelVersion::kQwen2_5;
  } else if (decoder_model_version == "qwen3") {
    decoder_model_version_ = DecoderModelVersion::kQwen3;
  } else if (decoder_model_version == "smollm2_135m") {
    decoder_model_version_ = DecoderModelVersion::kSmollm2_135m;
  } else if (decoder_model_version == "smollm3") {
    decoder_model_version_ = DecoderModelVersion::kSmollm3;
  } else if (decoder_model_version == "codegen") {
    decoder_model_version_ = DecoderModelVersion::kCodegen;
  } else if (decoder_model_version == "glm") {
    decoder_model_version_ = DecoderModelVersion::kGlm;
  } else {
    ET_CHECK_MSG(false, "Unsupported Decoder Model");
  }

  ET_LOG(Info, "creating module: model_path=%s", model_path.c_str());
  ET_LOG(Info, "creating runner: tokenizer_path=%s", tokenizer_path_.c_str());
  ET_LOG(Info, "eval mode=%d", eval_mode_);
}

template <typename T>
bool Runner<T>::is_loaded() const {
  return module_->is_loaded() && tokenizer_ && decoder_runner_ &&
      prompt_processor_ && token_generator_ && kv_manager_ && buffer_manager_;
}

template <typename T>
Error Runner<T>::load() {
  if (is_loaded()) {
    return Error::Ok;
  }

  std::string token_generator_method_name, prompt_processor_method_name;
  std::vector<std::string> method_names;
  switch (eval_mode_) {
    case EvalMode::kKVCached:
      prompt_processor_method_name = "forward";
      token_generator_method_name = "forward";
      method_names.emplace_back(token_generator_method_name);
      break;
    case EvalMode::kHybrid:
    case EvalMode::kLookaheadDecoding:
      prompt_processor_method_name = "prefill_forward";
      token_generator_method_name = "kv_forward";
      method_names.emplace_back(prompt_processor_method_name);
      method_names.emplace_back(token_generator_method_name);
      break;
    case EvalMode::kUnsupported:
      ET_CHECK_MSG(false, "Unsupported llama evaluation mode");
      break;
  }
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>();
  if (tokenizer_ != nullptr) {
    eos_ids->insert(tokenizer_->encode("<|eot_id|>", 0, 0).get()[0]);
    eos_ids->insert(tokenizer_->encode("<|eot|>", 0, 0).get()[0]);
    eos_ids->insert(tokenizer_->encode("<|end_of_text|>", 0, 0).get()[0]);
  } else {
    tokenizer_ = llm::load_tokenizer(tokenizer_path_);
    if (tokenizer_ == nullptr) {
      ET_LOG(
          Error, "Failed to load tokenizer with %s", tokenizer_path_.c_str());
      return Error::Internal;
    }
    eos_ids->insert(tokenizer_->eos_tok());
  }
  if (decoder_model_version_ == DecoderModelVersion::kLlama3) {
    eos_ids->insert(tokenizer_->encode("<|eot_id|>", 0, 0).get()[0]);
  } else if (decoder_model_version_ == DecoderModelVersion::kPhi4) {
    eos_ids->insert(tokenizer_->encode("<|end|>", 0, 0).get()[0]);
  } else if (
      decoder_model_version_ == DecoderModelVersion::kQwen3 ||
      decoder_model_version_ == DecoderModelVersion::kSmollm2_135m ||
      decoder_model_version_ == DecoderModelVersion::kSmollm3) {
    eos_ids->insert(tokenizer_->encode("<|im_end|>", 0, 0).get()[0]);
  } else if (
      decoder_model_version_ == DecoderModelVersion::kGemma ||
      decoder_model_version_ == DecoderModelVersion::kGemma2 ||
      decoder_model_version_ == DecoderModelVersion::kGemma3) {
    eos_ids->insert(tokenizer_->encode("<end_of_turn>", 0, 0).get()[0]);
  } else if (decoder_model_version_ == DecoderModelVersion::kCodegen) {
    eos_ids->insert(tokenizer_->encode("<|endoftext|>", 0, 0).get()[0]);
  } else if (decoder_model_version_ == DecoderModelVersion::kGlm) {
    eos_ids->insert(tokenizer_->encode("<|user|>", 0, 0).get()[0]);
  }

  // Try avoid getMetadataHelper as it is time consuming.
  Result<MethodMeta> method_meta =
      module_->method_meta(token_generator_method_name);

  // For some tokenizer.json, runtime vocab_size might be different, use output
  // shape to get vocab size.
  int32_t vocab_size = method_meta->output_tensor_meta(0)->sizes()[2];
  decoder_runner_ =
      std::make_unique<DecoderRunner>(module_.get(), vocab_size, temperature_);

  ET_CHECK_OK_OR_RETURN_ERROR(decoder_runner_->load(method_names));

  ET_LOG(Info, "Reading metadata from model");
  // retrieve any method meta, can be either prefill or kv
  int64_t num_layers =
      ET_UNWRAP(module_->get("get_n_layers")).toScalar().to<int64_t>();

  ET_CHECK_MSG(num_layers != -1, "Could not retrieve num layers");
  // k_cache: [1, n_heads, head_dim, seq_len]
  auto k_cache_shape = method_meta->output_tensor_meta(1)->sizes();
  int64_t num_heads = k_cache_shape[1];
  int64_t head_dim = k_cache_shape[2];
  bool use_int64_token = method_meta->input_tensor_meta(0)->scalar_type() ==
      executorch::aten::ScalarType::Long;

  // Use attention mask length to retrieve AR length and context length
  // Cache len equals to context_len - ar_len
  int32_t prompt_processor_ar_len = 0;
  int32_t token_generator_ar_len = 0;
  int32_t max_cache_len = 0;
  int32_t max_ar_len = 0;
  // atten mask: [1, AR-N, CL]
  auto atten_mask_meta_token = method_meta->input_tensor_meta(1);
  token_generator_ar_len = atten_mask_meta_token->sizes()[1];
  context_len_ = atten_mask_meta_token->sizes()[2];
  if (eval_mode_ == EvalMode::kKVCached) {
    prompt_processor_ar_len = token_generator_ar_len;
  } else if (
      eval_mode_ == EvalMode::kHybrid ||
      eval_mode_ == EvalMode::kLookaheadDecoding) {
    auto atten_mask_meta_prompt =
        module_->method_meta(prompt_processor_method_name)
            ->input_tensor_meta(1);
    prompt_processor_ar_len = atten_mask_meta_prompt->sizes()[1];
  }
  if (prompt_processor_ar_len == context_len_)
    max_cache_len = context_len_;
  else
    max_cache_len = context_len_ -
        std::min(token_generator_ar_len, prompt_processor_ar_len);
  max_ar_len = std::max(token_generator_ar_len, prompt_processor_ar_len);

  // Load the sliding window size if the model supports it.
  // This is used to configure the attention mask for models with window
  // attention
  int32_t sliding_window = context_len_;
  if (module_->method_names()->count("get_sliding_window") > 0) {
    sliding_window = ET_UNWRAP(module_->get("get_sliding_window")).toInt();
  }
  kv_manager_ = std::make_unique<KVManager<T>>(typename KVManager<T>::Metadata{
      context_len_,
      head_dim,
      max_ar_len,
      max_cache_len,
      num_heads,
      num_layers});

  prompt_processor_ = std::make_unique<PromptProcessor<T>>(
      decoder_runner_.get(),
      kv_manager_.get(),
      prompt_processor_method_name,
      typename PromptProcessor<T>::Metadata{
          context_len_,
          num_heads,
          num_layers,
          prompt_processor_ar_len,
          vocab_size,
          use_int64_token,
          sliding_window,
          cache_mode_});
  if (eval_mode_ == EvalMode::kLookaheadDecoding) {
    token_generator_ = std::make_unique<LhdTokenGenerator<T>>(
        tokenizer_.get(),
        decoder_runner_.get(),
        kv_manager_.get(),
        token_generator_method_name,
        std::move(eos_ids),
        typename LhdTokenGenerator<T>::Metadata{
            context_len_,
            num_heads,
            num_layers,
            token_generator_ar_len,
            vocab_size,
            use_int64_token,
            ngram_,
            window_,
            gcap_,
            sliding_window,
            cache_mode_},
        &stats_);
  } else {
    token_generator_ = std::make_unique<TokenGenerator<T>>(
        tokenizer_.get(),
        decoder_runner_.get(),
        kv_manager_.get(),
        token_generator_method_name,
        std::move(eos_ids),
        typename TokenGenerator<T>::Metadata{
            context_len_,
            num_heads,
            num_layers,
            token_generator_ar_len,
            vocab_size,
            use_int64_token,
            sliding_window,
            cache_mode_},
        &stats_);
  }

  buffer_manager_ = std::make_unique<ClientMem>();
  if (shared_buffer_) {
    buffer_manager_ = std::make_unique<RpcMem>(
        kv_manager_->total_cache_size_in_bytes(),
        prompt_processor_->total_prompt_processor_io_size_in_bytes(),
        token_generator_->total_token_generator_io_size_in_bytes());
  }
  ET_LOG(Info, "creating io_memory");
  // prepare io
  kv_manager_->init_cache(buffer_manager_.get(), prompt_processor_ar_len);
  prompt_processor_->init_io(
      buffer_manager_.get(),
      module_->method_meta(prompt_processor_method_name));
  token_generator_->init_io(
      buffer_manager_.get(), module_->method_meta(token_generator_method_name));
  return Error::Ok;
}

template <typename T>
Error Runner<T>::generate(
    const std::string& prompt,
    const llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  return generate_from_prompt_or_file(
      prompt, false, config, token_callback, stats_callback);
}

template <typename T>
Error Runner<T>::generate_from_prompt_or_file(
    const std::string& prompt,
    bool tokenized_prompt,
    const llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  ET_CHECK_MSG(!prompt.empty(), "prompt cannot be null");
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  stats_.inference_start_ms = time_in_ms();

  int32_t seq_len = config.seq_len;
  if (seq_len > context_len_) {
    ET_LOG(
        Info,
        "Warning: Requested seq_len (%d) exceeds compiled max_seq_len (%d). Clamping to %d.",
        seq_len,
        context_len_,
        context_len_);
    seq_len = context_len_;
  } else if (seq_len <= 0) {
    ET_LOG(
        Info,
        "Warning: Invalid seq_len (%d). Using compiled max_seq_len (%d).",
        seq_len,
        context_len_);
    seq_len = context_len_;
  }
  int32_t n_bos = (cur_pos_ == 0) ? 1 : 0;

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens;
  if (tokenized_prompt) {
    std::ifstream inFile(prompt, std::ios::binary);
    if (inFile.is_open()) {
      // Get file size
      inFile.seekg(0, std::ios::end);
      size_t fileSize = inFile.tellg();
      inFile.seekg(0, std::ios::beg);

      // Resize vector and read raw data
      prompt_tokens.resize(fileSize / sizeof(uint64_t));

      inFile.read(reinterpret_cast<char*>(prompt_tokens.data()), fileSize);
      inFile.close();
    } else {
      ET_CHECK_MSG(
          false,
          "Unable to read tokenized prompt from file: %s",
          prompt.c_str());
    }
  } else {
    tokenizers::Result<std::vector<uint64_t>> encode_res =
        tokenizer_->encode(prompt, n_bos, 0);
    ET_CHECK_TK_OK_OR_RETURN_ERROR(
        encode_res.error(), "failed to encode prompt %s", prompt.c_str());
    prompt_tokens = encode_res.get();
  }
  int num_prompt_tokens = prompt_tokens.size();
  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      cur_pos_ + num_prompt_tokens < seq_len,
      "sequence length exceeded - please increase the seq_len value");

  // Prompt Processor first
  if (token_callback && config.echo) {
    token_callback(prompt);
  }
  bool dump_logits = dump_logits_path_.empty() ? false : true;
  auto prefill_res =
      prompt_processor_->prefill(prompt_tokens, cur_pos_, dump_logits);
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  uint64_t cur_token = prefill_res.get();
  cur_pos_ += num_prompt_tokens;
  stats_.first_token_ms = time_in_ms();
  stats_.prompt_eval_end_ms = time_in_ms();

  // print the first token from prefill. No prev_token so use cur_token for
  // it.
  if (token_callback) {
    token_callback(
        ET_UNWRAP_TOKENIZER(tokenizer_->decode(cur_token, cur_token)));
  }
  ET_LOG(
      Info,
      "RSS after prompt prefill: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  // start the main loop
  prompt_tokens.push_back(cur_token);
  int64_t num_generated_tokens = ET_UNWRAP(token_generator_->generate(
      prompt_tokens, cur_pos_, seq_len, token_callback, dump_logits));
  stats_.inference_end_ms = time_in_ms();
  ET_LOG(
      Info,
      "RSS after finishing text generation: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);
  cur_pos_ += num_generated_tokens;
  if (cur_pos_ == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  stats_.num_prompt_tokens = num_prompt_tokens;
  stats_.num_generated_tokens = num_generated_tokens;
  print_report(stats_);
  print_performance_report(stats_, performance_output_path_);
  if (dump_logits) {
    save_logits(
        dump_logits_path_,
        prompt_processor_->get_all_logits(),
        token_generator_->get_all_logits());
  }
  if (stats_callback) {
    stats_callback(stats_);
  }
  return Error::Ok;
}

template <typename T>
Result<DecoderModelVersion> Runner<T>::get_decoder_model_version() {
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  return decoder_model_version_;
}

// Explicit instantiations
template class Runner<uint16_t>;
template class Runner<uint8_t>;

} // namespace example
