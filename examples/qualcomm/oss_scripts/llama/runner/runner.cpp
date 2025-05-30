/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A llama 3.2 runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/client_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
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

namespace example {
namespace {
void print_performance_report(
    const Stats& stats,
    const std::string& performance_output_path) {
  // For now, we just print the total inference time for CI, can save more info
  // in future if needed.
  std::ofstream outfile(performance_output_path.c_str());
  if (outfile.is_open()) {
    double num_tok = (stats.num_generated_tokens) /
        (double)(stats.inference_end_ms - stats.inference_start_ms) *
        stats.SCALING_FACTOR_UNITS_PER_SECOND;
    outfile << num_tok;
    outfile.close();
  } else {
    ET_CHECK_MSG(false, "Error saving the inference speed file");
  }
}
} // namespace

Runner::Runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& performance_output_path,
    const float temperature,
    const int eval_mode,
    const std::string& kv_updater)
    : tokenizer_path_(tokenizer_path),
      performance_output_path_(performance_output_path),
      temperature_(temperature),
      eval_mode_(static_cast<EvalMode>(eval_mode)) {
  module_ = std::make_unique<Module>(
      model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  if (kv_updater == "SmartMask") {
    kv_updater_ = KVManagerMode::SMART_MASK;
  } else if (kv_updater == "ShiftPointer") {
    kv_updater_ = KVManagerMode::SHIFT_POINTER;
  } else {
    ET_CHECK_MSG(false, "kv updater (%s) not found", kv_updater.c_str());
  }
  ET_LOG(Info, "creating module: model_path=%s", model_path.c_str());
  ET_LOG(Info, "creating runner: tokenizer_path=%s", tokenizer_path_.c_str());
  ET_LOG(Info, "eval mode=%d", eval_mode_);
  ET_LOG(Info, "kv updater=%s", kv_updater.c_str());
}

bool Runner::is_loaded() const {
  return module_->is_loaded() && tokenizer_ && decoder_runner_ &&
      prompt_processor_ && token_generator_ && kv_manager_ && buffer_manager_;
}

Error Runner::load() {
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
      prompt_processor_method_name = "prefill_forward";
      token_generator_method_name = "kv_forward";
      method_names.emplace_back(prompt_processor_method_name);
      method_names.emplace_back(token_generator_method_name);
      break;
    case EvalMode::kUnsupported:
      ET_CHECK_MSG(false, "Unsupported llama evaluation mode");
      break;
  }

  // load tokenizer. Assuming tiktoken is the default tokenizer
  tokenizer_ = get_tiktoken_for_llama();
  auto err = tokenizer_->load(tokenizer_path_);
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>();
  // Rely on tiktoken to throw error if the artifact is incompatible. Then we
  // fallback to BPE tokenizer.
  if (err != tokenizers::Error::Ok) {
    ET_LOG(
        Info,
        "Failed to load %s as a Tiktoken artifact, trying BPE tokenizer",
        tokenizer_path_.c_str());
    tokenizer_.reset();
    tokenizer_ = std::make_unique<tokenizers::Llama2cTokenizer>();
    err = tokenizer_->load(tokenizer_path_);
    llama_version_ = LlamaVersion::kLlama2;
    ET_CHECK_MSG(
        err == tokenizers::Error::Ok,
        "failed to load tokenizer %s",
        tokenizer_path_.c_str());
  } else {
    eos_ids->insert(tokenizer_->encode("<|eot_id|>", 0, 0).get()[0]);
    llama_version_ = LlamaVersion::kLlama3;
  }
  eos_ids->insert(tokenizer_->eos_tok());
  int32_t vocab_size = tokenizer_->vocab_size();
  decoder_runner_ =
      std::make_unique<DecoderRunner>(module_.get(), vocab_size, temperature_);

  ET_CHECK_OK_OR_RETURN_ERROR(decoder_runner_->load(method_names));

  ET_LOG(Info, "Reading metadata from model");
  // Try avoid getMetadataHelper as it is time consuming.
  Result<MethodMeta> method_meta =
      module_->method_meta(token_generator_method_name);
  // retrieve any method meta, can be either prefill or kv
  int64_t num_layers =
      ET_UNWRAP(module_->get("get_n_layers")).toScalar().to<int64_t>();
  ET_CHECK_MSG(num_layers != -1, "Could not retrieve num layers");
  // k_cache: [1, head_dim, seq_len]
  int64_t head_dim = method_meta->output_tensor_meta(1)->sizes()[1];
  int64_t num_heads = (method_meta->num_outputs() - 1) / (num_layers * 2);
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
  } else if (eval_mode_ == EvalMode::kHybrid) {
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

  kv_manager_ = std::make_unique<KVManager>(
      kv_updater_,
      KVManager::Metadata{
          context_len_,
          head_dim,
          max_ar_len,
          max_cache_len,
          num_heads,
          num_layers});

  prompt_processor_ = std::make_unique<PromptProcessor>(
      decoder_runner_.get(),
      kv_manager_.get(),
      prompt_processor_method_name,
      PromptProcessor::Metadata{
          context_len_,
          num_heads,
          num_layers,
          prompt_processor_ar_len,
          vocab_size,
          use_int64_token});
  token_generator_ = std::make_unique<TokenGenerator>(
      tokenizer_.get(),
      decoder_runner_.get(),
      kv_manager_.get(),
      token_generator_method_name,
      std::move(eos_ids),
      TokenGenerator::Metadata{
          context_len_,
          num_heads,
          num_layers,
          token_generator_ar_len,
          vocab_size,
          use_int64_token,
      },
      &stats_);

  buffer_manager_ = std::make_unique<ClientMem>();
  if (kv_updater_ == KVManagerMode::SMART_MASK) {
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

Error Runner::generate(
    const std::string& prompt,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback,
    bool echo,
    bool warming) {
  ET_CHECK_MSG(!prompt.empty(), "prompt cannot be null");
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  stats_.inference_start_ms = time_in_ms();

  seq_len = (seq_len > 0 && seq_len <= context_len_) ? seq_len : context_len_;
  int32_t n_bos = (cur_pos_ == 0) ? 1 : 0;
  tokenizers::Result<std::vector<uint64_t>> encode_res =
      tokenizer_->encode(prompt, n_bos, 0);
  ET_CHECK_TK_OK_OR_RETURN_ERROR(
      encode_res.error(), "failed to encode prompt %s", prompt.c_str());

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();
  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      cur_pos_ + num_prompt_tokens < seq_len,
      "sequence length exceeded - please increase the seq_len value");

  // Prompt Processor first
  if (token_callback) {
    token_callback(prompt);
  }

  auto prefill_res = prompt_processor_->prefill(prompt_tokens, cur_pos_);
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  uint64_t cur_token = prefill_res.get();
  cur_pos_ += num_prompt_tokens;
  stats_.first_token_ms = time_in_ms();
  stats_.prompt_eval_end_ms = time_in_ms();

  // print the first token from prefill. No prev_token so use cur_token for it.
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
      prompt_tokens, cur_pos_, seq_len, token_callback));
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
  if (stats_callback) {
    stats_callback(stats_);
  }
  return Error::Ok;
}

Result<LlamaVersion> Runner::get_llama_version() {
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  return llama_version_;
}

} // namespace example
