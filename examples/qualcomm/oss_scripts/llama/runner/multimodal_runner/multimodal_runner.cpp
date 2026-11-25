/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A llama 3.2 runner that includes preprocessing and post processing
// logic. The text_decoder takes in a string as input and emits a string as
// output.

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/client_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_embedding_merger.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_lhd_token_generator.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>

#if defined(__aarch64__)
#include "arm_neon.h"
#endif

using executorch::extension::Module;
using executorch::extension::llm::Audio;
using executorch::extension::llm::get_rss_bytes;
using executorch::extension::llm::Image;
using executorch::extension::llm::MultimodalInput;
using executorch::extension::llm::print_report;
using executorch::extension::llm::Stats;
using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::EValue;
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
    const std::vector<std::byte>& prefill_logits,
    const std::vector<std::byte>& decode_logits) {
  std::ofstream outFile(dump_logits_path.c_str(), std::ios::binary);
  if (outFile.is_open()) {
    outFile.write(
        reinterpret_cast<const char*>(prefill_logits.data()),
        prefill_logits.size());

    outFile.write(
        reinterpret_cast<const char*>(decode_logits.data()),
        decode_logits.size());
    outFile.close();
  } else {
    ET_CHECK_MSG(false, "Error saving the dump logits file");
  }
}

Result<int32_t> speech_constant(Module& module, const char* name) {
  auto result = module.get(name);
  ET_CHECK_OR_RETURN_ERROR(
      result.ok(),
      InvalidArgument,
      "Missing speech metadata %s; re-export all three speech components",
      name);
  const auto& value = result.get();
  ET_CHECK_OR_RETURN_ERROR(
      value.isInt() && value.toInt() >= 0 &&
          value.toInt() <= std::numeric_limits<int32_t>::max(),
      InvalidArgument,
      "Invalid speech metadata %s",
      name);
  return static_cast<int32_t>(value.toInt());
}

Error validate_speech_io(
    Module& module,
    const std::vector<
        std::pair<executorch::aten::ScalarType, std::vector<int32_t>>>& inputs,
    const std::vector<int64_t>& output_shape) {
  ET_CHECK_OK_OR_RETURN_ERROR(module.load_forward());
  ET_ASSIGN_OR_RETURN(meta, module.method_meta("forward"));
  ET_CHECK_OR_RETURN_ERROR(
      meta.num_inputs() == inputs.size() && meta.num_outputs() == 1,
      InvalidArgument,
      "Unexpected speech component input/output count");
  for (size_t i = 0; i < inputs.size(); ++i) {
    ET_ASSIGN_OR_RETURN(info, meta.input_tensor_meta(i));
    const auto sizes = info.sizes();
    ET_CHECK_OR_RETURN_ERROR(
        info.scalar_type() == inputs[i].first &&
            sizes.size() == inputs[i].second.size() &&
            std::equal(sizes.begin(), sizes.end(), inputs[i].second.begin()),
        InvalidArgument,
        "Speech component input %zu has an incompatible shape or dtype",
        i);
  }
  ET_ASSIGN_OR_RETURN(output, meta.output_tensor_meta(0));
  const auto sizes = output.sizes();
  ET_CHECK_OR_RETURN_ERROR(
      output.scalar_type() == executorch::aten::ScalarType::Float &&
          sizes.size() == output_shape.size() &&
          std::equal(sizes.begin(), sizes.end(), output_shape.begin()),
      InvalidArgument,
      "Speech component output has an incompatible shape or dtype");
  return Error::Ok;
}

} // namespace

QNNMultimodalRunner::QNNMultimodalRunner(
    std::unique_ptr<executorch::extension::Module> encoder,
    std::unique_ptr<executorch::extension::Module> tok_embedding,
    std::unique_ptr<executorch::extension::Module> text_decoder,
    const std::string& model_version,
    const std::string& tokenizer_path,
    const std::string& dump_logits_path,
    const std::string& performance_output_path,
    const float temperature,
    const int eval_mode,
    const bool shared_buffer,
    const int ngram,
    const int window,
    const int gcap)
    // TODO: The arguments for the MultimodalRunner base class are currently
    // unused in this runner implementation. QNNMultimodalRunner will be
    // refactored in the future to align with MultimodalRunner's usage.
    : MultimodalRunner(
          {},
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          std::make_unique<Stats>()),
      encoder_(std::move(encoder)),
      tok_embedding_(std::move(tok_embedding)),
      text_decoder_(std::move(text_decoder)),
      ngram_(ngram),
      window_(window),
      gcap_(gcap),
      tokenizer_path_(tokenizer_path),
      performance_output_path_(performance_output_path),
      dump_logits_path_(dump_logits_path),
      temperature_(temperature),
      eval_mode_(static_cast<EvalMode>(eval_mode)),
      shared_buffer_(shared_buffer) {
  stats_.reset();

  if (model_version == "smolvlm") {
    model_version_ = VisionLanguageModel::kSmolvlm;
  } else if (model_version == "internvl3") {
    model_version_ = VisionLanguageModel::kInternvl3;
  } else if (model_version == "lfm3") {
    model_version_ = VisionLanguageModel::kLfm3Omni;
    cache_mode_ = CacheMode::HybridCache;
  } else if (model_version == "granite_speech") {
    model_version_ = AudioLanguageModel::kGraniteSpeech;
  } else if (model_version == "lfm3_speech") {
    model_version_ = AudioLanguageModel::kLfm3OmniSpeech;
    cache_mode_ = CacheMode::HybridCache;
  } else {
    ET_CHECK_MSG(false, "Unsupported Decoder Model");
  }

  ET_LOG(Info, "creating runner: tokenizer_path=%s", tokenizer_path_.c_str());
  ET_LOG(Info, "eval mode=%d", eval_mode_);
}

bool QNNMultimodalRunner::is_loaded() const {
  return encoder_->is_loaded() && tok_embedding_->is_loaded() &&
      text_decoder_->is_loaded() && embedding_merger_ && tokenizer_ &&
      decoder_runner_ && prompt_processor_ && token_generator_ && kv_manager_ &&
      buffer_manager_;
}

Error QNNMultimodalRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }
  std::string prompt_embedding_method_name, tok_embedding_method_name;
  std::string token_generator_method_name, prompt_processor_method_name;
  std::vector<std::string> method_names;
  switch (eval_mode_) {
    case EvalMode::kKVCached:
      prompt_embedding_method_name = "tok_embedding_kv_forward";
      tok_embedding_method_name = "tok_embedding_kv_forward";
      prompt_processor_method_name = "kv_forward";
      token_generator_method_name = "kv_forward";
      method_names.emplace_back(prompt_processor_method_name);
      method_names.emplace_back(token_generator_method_name);
      break;
    case EvalMode::kHybrid:
    case EvalMode::kLookaheadDecoding:
      prompt_embedding_method_name = "tok_embedding_prefill_forward";
      tok_embedding_method_name = "tok_embedding_kv_forward";
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
  if (const auto* vlm = std::get_if<VisionLanguageModel>(&model_version_)) {
    if (*vlm == VisionLanguageModel::kSmolvlm) {
      eos_ids->insert(tokenizer_->encode("<end_of_utterance>", 0, 0).get()[0]);
    } else if (
        *vlm == VisionLanguageModel::kInternvl3 ||
        *vlm == VisionLanguageModel::kLfm3Omni) {
      eos_ids->insert(tokenizer_->encode("<|im_end|>", 0, 0).get()[0]);
    }
  } else if (
      const auto* alm = std::get_if<AudioLanguageModel>(&model_version_)) {
    if (*alm == AudioLanguageModel::kGraniteSpeech) {
      eos_ids->insert(tokenizer_->encode("<|end_of_text|>", 0, 0).get()[0]);
    } else if (*alm == AudioLanguageModel::kLfm3OmniSpeech) {
      eos_ids->insert(tokenizer_->encode("<|im_end|>", 0, 0).get()[0]);
    }
  }

  Result<MethodMeta> method_meta =
      text_decoder_->method_meta(token_generator_method_name);

  // For some tokenizer.json, runtime vocab_size might be different, use output
  // shape to get vocab size.
  int32_t vocab_size = method_meta->output_tensor_meta(0)->sizes()[2];
  decoder_runner_ = std::make_unique<DecoderRunner>(
      text_decoder_.get(), vocab_size, temperature_);

  ET_CHECK_OK_OR_RETURN_ERROR(decoder_runner_->load(method_names));

  ET_LOG(Info, "Reading metadata from model");
  // retrieve any method meta, can be either prefill or kv
  ET_ASSIGN_OR_RETURN(num_layers_evalue__, text_decoder_->get("get_n_layers"));
  int64_t num_layers = num_layers_evalue__.toScalar().to<int64_t>();

  ET_CHECK_MSG(num_layers != -1, "Could not retrieve num layers");
  // Decoders that also return their final hidden state put it at output 1.
  const bool has_hidden =
      text_decoder_->method_names()->count("get_output_hidden") > 0;
  // k_cache: [1, n_heads, head_dim, seq_len]
  auto k_cache_shape =
      method_meta->output_tensor_meta(has_hidden ? 2 : 1)->sizes();
  int64_t num_heads = k_cache_shape[1];

  // hidden_state: [1, ar_len, dim]
  int64_t dim = tok_embedding_->method_meta(tok_embedding_method_name)
                    ->output_tensor_meta(0)
                    ->sizes()[2];
  embedding_dim_ = static_cast<int32_t>(dim);
  const int32_t hidden_dim = has_hidden ? static_cast<int32_t>(dim) : 0;
  bool use_int64_token =
      tok_embedding_->method_meta(tok_embedding_method_name)
          ->input_tensor_meta(0)
          ->scalar_type() == executorch::aten::ScalarType::Long;

  // Use attention mask length to retrieve AR length and context length
  // Cache len equals to context_len - ar_len
  int32_t prompt_processor_ar_len = 0;
  int32_t token_generator_ar_len = 0;
  int32_t max_cache_len = 0;
  int32_t max_ar_len = 0;
  // atten mask: [1, 1, AR-N, CL]
  auto atten_mask_meta_token = method_meta->input_tensor_meta(1);
  token_generator_ar_len = atten_mask_meta_token->sizes()[2];
  context_len_ = atten_mask_meta_token->sizes()[3];
  if (eval_mode_ == EvalMode::kKVCached) {
    prompt_processor_ar_len = token_generator_ar_len;
  } else if (
      eval_mode_ == EvalMode::kHybrid ||
      eval_mode_ == EvalMode::kLookaheadDecoding) {
    auto atten_mask_meta_prompt =
        text_decoder_->method_meta(prompt_processor_method_name)
            ->input_tensor_meta(1);
    prompt_processor_ar_len = atten_mask_meta_prompt->sizes()[2];
  }
  if (prompt_processor_ar_len == context_len_)
    max_cache_len = context_len_;
  else
    max_cache_len = context_len_ -
        std::min(token_generator_ar_len, prompt_processor_ar_len);
  max_ar_len = std::max(token_generator_ar_len, prompt_processor_ar_len);

  // Initialize Encoder
  encoder_runner_ = std::make_unique<EncoderRunner>(encoder_.get());
  ET_CHECK_OK_OR_RETURN_ERROR(encoder_runner_->load());

  // Initialize TokenEmbeddingProcessor
  tok_embedding_runner_ =
      std::make_unique<TokenEmbeddingRunner>(tok_embedding_.get());
  ET_CHECK_OK_OR_RETURN_ERROR(tok_embedding_runner_->load(
      {prompt_embedding_method_name, tok_embedding_method_name}));
  tok_embedding_processor_ = std::make_unique<TokenEmbeddingProcessor>(
      tok_embedding_runner_.get(),
      prompt_embedding_method_name,
      TokenEmbeddingProcessor::Metadata{
          context_len_,
          prompt_processor_ar_len,
          vocab_size,
          use_int64_token,
          static_cast<int32_t>(dim)});

  // Load the sliding window size if the model supports it.
  // This is used to configure the attention mask for models with window
  // attention
  int32_t sliding_window = context_len_;
  if (text_decoder_->method_names()->count("get_sliding_window") > 0) {
    ET_ASSIGN_OR_RETURN(
        sliding_window_evalue__, text_decoder_->get("get_sliding_window"));
    sliding_window = sliding_window_evalue__.toInt();
  }
  // Hybrid models (LFM3): only attention layers have KV I/O; conv layers carry
  // a rolling conv state instead.
  if (text_decoder_->method_names()->count("get_n_self_layers") > 0) {
    ET_ASSIGN_OR_RETURN(
        n_self_evalue__, text_decoder_->get("get_n_self_layers"));
    num_layers = n_self_evalue__.toScalar().to<int64_t>();
  }
  auto conv_state_manager = ConvStateManager::from_module(
      *text_decoder_, method_meta.get(), max_ar_len);
  ET_CHECK_OK_OR_RETURN_ERROR(conv_state_manager.error());
  conv_state_manager_ = std::move(*conv_state_manager);
  if (conv_state_manager_ != nullptr) {
    ET_CHECK_MSG(
        eval_mode_ != EvalMode::kLookaheadDecoding,
        "Conv-state models support kv and hybrid modes only");
  }
  kv_manager_ = std::make_unique<KVManager>(
      KVManager::Metadata{
          context_len_,
          max_ar_len,
          max_cache_len,
          num_heads,
          num_layers,
          has_hidden ? 2 : 1},
      std::make_unique<MethodMeta>(std::move(
          text_decoder_->method_meta(token_generator_method_name).get())));

  prompt_processor_ = std::make_unique<MultimodalPromptProcessor>(
      decoder_runner_.get(),
      kv_manager_.get(),
      prompt_processor_method_name,
      MultimodalPromptProcessor::Metadata{
          context_len_,
          num_heads,
          num_layers,
          prompt_processor_ar_len,
          vocab_size,
          use_int64_token,
          sliding_window,
          cache_mode_,
          static_cast<int32_t>(dim),
          hidden_dim},
      std::make_unique<MethodMeta>(std::move(
          text_decoder_->method_meta(prompt_processor_method_name).get())));

  // Initialize EmbeddingGenerator
  tok_embedding_generator_ = std::make_unique<TokenEmbeddingProcessor>(
      tok_embedding_runner_.get(),
      tok_embedding_method_name,
      TokenEmbeddingProcessor::Metadata{
          context_len_,
          token_generator_ar_len,
          vocab_size,
          use_int64_token,
          static_cast<int32_t>(dim)});
  if (eval_mode_ == EvalMode::kLookaheadDecoding) {
    // Initialize TokenGenerator
    token_generator_ = std::make_unique<MultimodalLhdTokenGenerator>(
        tokenizer_.get(),
        tok_embedding_generator_.get(),
        decoder_runner_.get(),
        kv_manager_.get(),
        token_generator_method_name,
        std::move(eos_ids),
        MultimodalLhdTokenGenerator::Metadata{
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
            cache_mode_,
            static_cast<int32_t>(dim)},
        &stats_,
        std::make_unique<MethodMeta>(std::move(
            text_decoder_->method_meta(token_generator_method_name).get())));
  } else {
    MultimodalTokenGenerator::Metadata generator_metadata{
        context_len_,
        num_heads,
        num_layers,
        token_generator_ar_len,
        vocab_size,
        use_int64_token,
        sliding_window,
        cache_mode_,
        static_cast<int32_t>(dim),
        hidden_dim};
    if (depth_ != nullptr) {
      ET_CHECK_OK_OR_RETURN_ERROR(load_speech_components(hidden_dim));
      ET_CHECK_MSG(has_hidden, "Speech output needs a decoder hidden state");
      ET_ASSIGN_OR_RETURN(
          scale_evalue__, text_decoder_->get("get_hidden_scale"));
      ET_ASSIGN_OR_RETURN(
          zero_point_evalue__, text_decoder_->get("get_hidden_zero_point"));
      ET_ASSIGN_OR_RETURN(
          logits_scale_evalue__, text_decoder_->get("get_logits_scale"));
      ET_ASSIGN_OR_RETURN(
          logits_zero_point_evalue__,
          text_decoder_->get("get_logits_zero_point"));
      auto speech_generator = std::make_unique<Lfm3SpeechTokenGenerator>(
          tokenizer_.get(),
          tok_embedding_generator_.get(),
          decoder_runner_.get(),
          kv_manager_.get(),
          token_generator_method_name,
          std::move(eos_ids),
          generator_metadata,
          &stats_,
          std::make_unique<MethodMeta>(std::move(
              text_decoder_->method_meta(token_generator_method_name).get())),
          depth_.get(),
          embed_frame_.get(),
          Lfm3SpeechTokenGenerator::SpeechConfig{
              static_cast<float>(scale_evalue__.toDouble()),
              static_cast<int32_t>(zero_point_evalue__.toInt()),
              audio_temperature_,
              audio_repetition_penalty_,
              speech_seed_,
              codec_config_.codebook_size,
              codec_config_.audio_eos_code,
              static_cast<float>(logits_scale_evalue__.toDouble()),
              static_cast<int32_t>(logits_zero_point_evalue__.toInt())});
      ET_CHECK_OK_OR_RETURN_ERROR(speech_generator->load_speech());
      speech_generator_ = speech_generator.get();
      token_generator_ = std::move(speech_generator);
    } else {
      token_generator_ = std::make_unique<MultimodalTokenGenerator>(
          tokenizer_.get(),
          tok_embedding_generator_.get(),
          decoder_runner_.get(),
          kv_manager_.get(),
          token_generator_method_name,
          std::move(eos_ids),
          generator_metadata,
          &stats_,
          std::make_unique<MethodMeta>(std::move(
              text_decoder_->method_meta(token_generator_method_name).get())));
    }
  }

  buffer_manager_ = std::make_unique<ClientMem>();
  if (shared_buffer_) {
    buffer_manager_ = std::make_unique<RpcMem>(
        kv_manager_->total_cache_size_in_bytes() +
            (conv_state_manager_ ? conv_state_manager_->total_size_in_bytes()
                                 : 0),
        prompt_processor_->total_prompt_processor_io_size_in_bytes(),
        token_generator_->total_token_generator_io_size_in_bytes(),
        tok_embedding_processor_->total_embedding_processor_io_size_in_bytes(),
        tok_embedding_generator_->total_embedding_processor_io_size_in_bytes());
  }

  ET_LOG(Info, "creating io_memory");
  // prepare io
  kv_manager_->init_cache(buffer_manager_.get(), prompt_processor_ar_len);
  if (conv_state_manager_ != nullptr) {
    conv_state_manager_->init(buffer_manager_.get());
    prompt_processor_->set_conv_state_manager(conv_state_manager_.get());
    token_generator_->set_conv_state_manager(conv_state_manager_.get());
  }
  prompt_processor_->init_io(
      buffer_manager_.get(),
      text_decoder_->method_meta(prompt_processor_method_name));
  token_generator_->init_io(
      buffer_manager_.get(),
      text_decoder_->method_meta(token_generator_method_name));
  // Prepare io for embedding
  tok_embedding_processor_->init_io(
      buffer_manager_.get(),
      tok_embedding_->method_meta(prompt_embedding_method_name));
  tok_embedding_generator_->init_io(
      buffer_manager_.get(),
      tok_embedding_->method_meta(tok_embedding_method_name));

  // Initialize embedding merger
  embedding_merger_ =
      std::make_unique<MultimodalEmbeddingMerger>(static_cast<int32_t>(dim));

  return Error::Ok;
}

executorch::runtime::Error QNNMultimodalRunner::generate(
    const std::vector<MultimodalInput>& inputs,
    const llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  ET_CHECK_MSG(!inputs.empty(), "inputs cannot be empty");
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  const auto* alm = std::get_if<AudioLanguageModel>(&model_version_);
  const auto* vlm = std::get_if<VisionLanguageModel>(&model_version_);
  const bool lfm3 = (alm && *alm == AudioLanguageModel::kLfm3OmniSpeech) ||
      (vlm && *vlm == VisionLanguageModel::kLfm3Omni);
  ET_CHECK_OR_RETURN_ERROR(
      !lfm3 || cur_pos_ == 0,
      NotSupported,
      "LFM3 multimodal generation is single-turn; create a new runner for another turn");
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
  // For multimodal, we will disable n_bos
  int32_t n_bos = 0;

  std::string prompt;
  std::vector<uint64_t> prompt_tokens;
  bool dump_logits = !dump_logits_path_.empty();

  // Reset merger for new generation
  embedding_merger_->reset();

  // Process each input and add embeddings to merger
  for (const MultimodalInput& input : inputs) {
    if (input.is_text()) {
      std::string text = input.get_text();
      prompt += text;

      tokenizers::Result<std::vector<uint64_t>> encode_res =
          tokenizer_->encode(text, n_bos, 0);
      ET_CHECK_TK_OK_OR_RETURN_ERROR(
          encode_res.error(), "failed to encode prompt %s", text.c_str());

      std::vector<uint64_t> tokens = encode_res.get();
      tok_embedding_processor_->prefill(tokens);
      const TensorStruct<float>& text_embeddings =
          tok_embedding_processor_->get_prompt_embeddings();

      // Add text embeddings to merger
      embedding_merger_->add_embeddings(text_embeddings);

      prompt_tokens.insert(prompt_tokens.end(), tokens.begin(), tokens.end());

    } else if (input.is_audio()) {
      ET_CHECK_OR_RETURN_ERROR(
          alm != nullptr,
          InvalidArgument,
          "Audio input requires an audio-language model");
      const Audio& audio = input.get_audio();
      auto audio_tensor_res = audio.toTensor();
      ET_CHECK_OK_OR_RETURN_ERROR(audio_tensor_res.error());
      executorch::extension::TensorPtr audio_tensor_ptr =
          audio_tensor_res.get();
      const bool lfm3_audio = *alm == AudioLanguageModel::kLfm3OmniSpeech;
      if (lfm3_audio) {
        const auto& features = *audio_tensor_ptr;
        ET_CHECK_OR_RETURN_ERROR(
            features.scalar_type() == executorch::aten::ScalarType::Float &&
                features.dim() == 3 && features.size(0) == 1 &&
                features.size(1) == 129 && features.size(2) > 0,
            InvalidArgument,
            "LFM3 audio features must be float32 [1, 129, frames]");
      }
      auto encode_res = encoder_runner_->encode(audio_tensor_ptr);
      ET_CHECK_OK_OR_RETURN_ERROR(encode_res.error());
      executorch::aten::Tensor audio_embeddings_tensor = encode_res.get();

      if (lfm3_audio) {
        ET_CHECK_OR_RETURN_ERROR(
            audio_embeddings_tensor.scalar_type() ==
                    executorch::aten::ScalarType::Float &&
                audio_embeddings_tensor.dim() == 3 &&
                audio_embeddings_tensor.size(0) == 1 &&
                audio_embeddings_tensor.size(2) == embedding_dim_,
            InvalidArgument,
            "LFM3 audio embeddings must be float32 [1, tokens, decoder_dim]");
        const auto& features = *audio_tensor_ptr;
        const int64_t rows = features.size(1), max_frames = features.size(2);
        const float* valid =
            features.const_data_ptr<float>() + (rows - 1) * max_frames;
        const int64_t frames =
            std::count_if(valid, valid + max_frames, [](float value) {
              return value > 0.5f;
            });
        const int64_t tokens = std::min<int64_t>(
            (frames + 7) / 8, audio_embeddings_tensor.size(1));
        if (tokens == 0) {
          continue;
        }
        std::array<executorch::aten::TensorImpl::SizesType, 3> sizes = {
            1,
            static_cast<executorch::aten::TensorImpl::SizesType>(tokens),
            static_cast<executorch::aten::TensorImpl::SizesType>(
                audio_embeddings_tensor.size(2))};
        executorch::aten::TensorImpl valid_tokens(
            executorch::aten::ScalarType::Float,
            sizes.size(),
            sizes.data(),
            audio_embeddings_tensor.mutable_data_ptr<float>());
        embedding_merger_->add_embeddings(
            executorch::aten::Tensor(&valid_tokens));
      } else {
        // Add audio embeddings to merger
        embedding_merger_->add_embeddings(audio_embeddings_tensor);
      }
    } else if (input.is_image()) {
      const Image& image = input.get_image();
      auto image_tensor_res = image.toTensor(/*with_batch*/ true);
      executorch::extension::TensorPtr image_tensor_ptr =
          image_tensor_res.get();

      auto encode_res = encoder_runner_->encode(image_tensor_ptr);
      executorch::aten::Tensor image_embeddings_tensor = encode_res.get();

      // Add image embeddings to merger
      embedding_merger_->add_embeddings(image_embeddings_tensor);

    } else {
      ET_CHECK_MSG(false, "Unsupported input data type");
    }
  }

  TensorStruct<float> merged_embeddings =
      embedding_merger_->get_merged_embeddings();
  int num_prompt_tokens = embedding_merger_->get_total_tokens();

  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  ET_CHECK_MSG(
      cur_pos_ + num_prompt_tokens < seq_len,
      "sequence length exceeded - please increase the seq_len value");

  if (token_callback && config.echo) {
    token_callback(prompt);
  }

  auto prefill_res = prompt_processor_->prefill(
      merged_embeddings, cur_pos_, dump_logits, nullptr);
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  uint64_t cur_token = prefill_res.get();
  cur_pos_ += num_prompt_tokens;
  stats_.first_token_ms = time_in_ms();
  stats_.prompt_eval_end_ms = time_in_ms();

  // print the first token from prefill. No prev_token so use cur_token for
  // it. The speech generator prints its text tokens itself (in a speech phase
  // the prefill token is not used).
  if (token_callback && speech_generator_ == nullptr) {
    ET_ASSIGN_OR_RETURN_TOKENIZER(
        decoded_token__, tokenizer_->decode(cur_token, cur_token));
    token_callback(decoded_token__);
  }
  ET_LOG(
      Info,
      "RSS after prompt prefill: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);

  // start the main loop
  prompt_tokens.push_back(cur_token);
  if (speech_generator_ != nullptr) {
    speech_generator_->set_prompt_outputs(
        prompt_processor_->last_hidden(), prompt_processor_->last_logits());
  }

  ET_ASSIGN_OR_RETURN(
      num_generated_tokens,
      token_generator_->generate(
          prompt_tokens,
          cur_pos_,
          seq_len,
          token_callback,
          dump_logits,
          nullptr));
  stats_.inference_end_ms = time_in_ms();
  ET_LOG(
      Info,
      "RSS after finishing text generation: %f MiB (0 if unsupported)",
      get_rss_bytes() / 1024.0 / 1024.0);
  cur_pos_ += num_generated_tokens;
  if (cur_pos_ == seq_len) {
    ET_LOG(Info, "Sequence length (%i tokens) reached!", seq_len);
  }

  if (speech_generator_ != nullptr) {
    ET_CHECK_OK_OR_RETURN_ERROR(write_speech());
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

void QNNMultimodalRunner::set_speech(
    std::unique_ptr<Module> depth,
    std::unique_ptr<Module> embed_frame,
    std::unique_ptr<Module> codec,
    const std::string& wav_path,
    std::optional<float> audio_temperature,
    std::optional<float> audio_repetition_penalty,
    uint32_t seed) {
  depth_ = std::move(depth);
  embed_frame_ = std::move(embed_frame);
  codec_ = std::move(codec);
  wav_path_ = wav_path;
  audio_temperature_ = audio_temperature;
  audio_repetition_penalty_ = audio_repetition_penalty;
  speech_seed_ = seed;
}

Error QNNMultimodalRunner::load_speech_components(int32_t hidden_dim) {
  ET_CHECK_OR_RETURN_ERROR(
      depth_ && embed_frame_ && codec_ && !wav_path_.empty() &&
          hidden_dim > 0 && eval_mode_ != EvalMode::kLookaheadDecoding,
      InvalidArgument,
      "Speech requires all three components, a hidden output and an output path");
  ET_CHECK_OK_OR_RETURN_ERROR(depth_->load_forward());
  ET_CHECK_OK_OR_RETURN_ERROR(embed_frame_->load_forward());
  ET_CHECK_OK_OR_RETURN_ERROR(codec_->load_forward());
  ET_ASSIGN_OR_RETURN(n_q, speech_constant(*depth_, "get_num_codebooks"));
  ET_ASSIGN_OR_RETURN(codebook, speech_constant(*depth_, "get_codebook_size"));
  ET_ASSIGN_OR_RETURN(eos, speech_constant(*depth_, "get_audio_eos_code"));
  ET_CHECK_OR_RETURN_ERROR(
      n_q == Lfm3SpeechTokenGenerator::kNumCodebooks && codebook > 1 &&
          static_cast<int64_t>(n_q) * codebook <=
              std::numeric_limits<int32_t>::max() &&
          eos == codebook - 1,
      NotSupported,
      "This speech generator requires 12 codebooks and an explicit final EOS row");
  for (Module* module : {depth_.get(), embed_frame_.get(), codec_.get()}) {
    ET_ASSIGN_OR_RETURN(
        version, speech_constant(*module, "get_speech_contract_version"));
    ET_ASSIGN_OR_RETURN(
        module_n_q, speech_constant(*module, "get_num_codebooks"));
    ET_ASSIGN_OR_RETURN(
        module_codebook, speech_constant(*module, "get_codebook_size"));
    ET_ASSIGN_OR_RETURN(
        module_eos, speech_constant(*module, "get_audio_eos_code"));
    ET_CHECK_OR_RETURN_ERROR(
        version == 1 && module_n_q == n_q && module_codebook == codebook &&
            module_eos == eos,
        InvalidArgument,
        "Speech components have different or obsolete contracts; re-export all three");
  }
  ET_ASSIGN_OR_RETURN(window, speech_constant(*codec_, "get_codec_frames"));
  ET_ASSIGN_OR_RETURN(left, speech_constant(*codec_, "get_codec_left_context"));
  ET_ASSIGN_OR_RETURN(
      right, speech_constant(*codec_, "get_codec_right_context"));
  ET_ASSIGN_OR_RETURN(samples, speech_constant(*codec_, "get_frame_samples"));
  ET_ASSIGN_OR_RETURN(rate, speech_constant(*codec_, "get_sample_rate"));
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<int64_t>(window) > static_cast<int64_t>(left) + right &&
          static_cast<int64_t>(window) * n_q <=
              std::numeric_limits<int32_t>::max() &&
          samples > 0 && rate > 0 &&
          rate <= std::numeric_limits<int32_t>::max() / 2 &&
          static_cast<int64_t>(window) * samples <=
              std::numeric_limits<int32_t>::max(),
      InvalidArgument,
      "Invalid codec context, sample geometry or window");
  using executorch::aten::ScalarType;
  ET_CHECK_OK_OR_RETURN_ERROR(validate_speech_io(
      *depth_,
      {{ScalarType::Float, {1, hidden_dim}},
       {ScalarType::Int, {n_q}},
       {ScalarType::Float, {n_q}}},
      {n_q, codebook}));
  ET_CHECK_OK_OR_RETURN_ERROR(validate_speech_io(
      *embed_frame_, {{ScalarType::Int, {n_q}}}, {1, 1, hidden_dim}));
  ET_CHECK_OK_OR_RETURN_ERROR(validate_speech_io(
      *codec_,
      {{ScalarType::Int, {n_q, window}}, {ScalarType::Int, {1}}},
      {static_cast<int64_t>(window) * samples}));
  codec_config_ = {window, left, right, samples, rate, codebook, eos};
  return Error::Ok;
}

Error QNNMultimodalRunner::write_speech() {
  const auto& frames = speech_generator_->frames();
  if (frames.empty()) {
    return Error::Ok;
  }
  const auto& config = codec_config_;
  ET_CHECK_OR_RETURN_ERROR(
      config.window > 0 && config.frame_samples > 0 &&
          frames.size() <= (std::numeric_limits<uint32_t>::max() - 36) /
                  (2ULL * config.frame_samples),
      InvalidArgument,
      "Speech output exceeds the RIFF PCM size limit");
  const size_t total_samples = frames.size() * config.frame_samples;
  std::vector<int16_t> pcm;
  pcm.reserve(total_samples);
  const long t0 = time_in_ms();
  size_t emitted = 0;
  while (emitted < frames.size()) {
    const size_t start =
        emitted > static_cast<size_t>(config.left) ? emitted - config.left : 0;
    const size_t valid = std::min<size_t>(config.window, frames.size() - start);
    const size_t end =
        frames.size() - start <= static_cast<size_t>(config.window)
        ? frames.size()
        : start + config.window - config.right;
    std::vector<int32_t> codes(
        static_cast<size_t>(Lfm3SpeechTokenGenerator::kNumCodebooks) *
            config.window,
        config.audio_eos_code);
    for (size_t t = 0; t < valid; ++t) {
      for (int q = 0; q < Lfm3SpeechTokenGenerator::kNumCodebooks; ++q) {
        const int32_t code = frames[start + t][q];
        ET_CHECK_OR_RETURN_ERROR(
            code >= 0 && code < config.audio_eos_code,
            InvalidArgument,
            "Invalid generated codec code");
        codes[static_cast<size_t>(q) * config.window + t] = code;
      }
    }
    auto codes_t = executorch::extension::from_blob(
        codes.data(),
        {Lfm3SpeechTokenGenerator::kNumCodebooks, config.window},
        executorch::aten::ScalarType::Int);
    int32_t valid_frames = static_cast<int32_t>(valid);
    auto valid_t = executorch::extension::from_blob(
        &valid_frames, {1}, executorch::aten::ScalarType::Int);
    auto out = codec_->forward({EValue(codes_t), EValue(valid_t)});
    ET_CHECK_OK_OR_RETURN_ERROR(out.error());
    ET_CHECK_OR_RETURN_ERROR(
        out.get().size() == 1 && out.get()[0].isTensor(),
        InvalidArgument,
        "Codec must return one waveform tensor");
    const auto& audio = out.get()[0].toTensor();
    ET_CHECK_OR_RETURN_ERROR(
        audio.scalar_type() == executorch::aten::ScalarType::Float &&
            audio.dim() == 1 &&
            audio.size(0) ==
                static_cast<int64_t>(config.window) * config.frame_samples,
        InvalidArgument,
        "Codec waveform must be float32 [window * frame_samples]");
    const size_t offset = (emitted - start) * config.frame_samples;
    const size_t count = (end - emitted) * config.frame_samples;
    ET_CHECK_OR_RETURN_ERROR(
        end > emitted && offset + count <= static_cast<size_t>(audio.numel()),
        InvalidArgument,
        "Invalid codec boundary trimming");
    const float* samples = audio.const_data_ptr<float>() + offset;
    for (size_t i = 0; i < count; ++i) {
      ET_CHECK_OR_RETURN_ERROR(
          std::isfinite(samples[i]),
          InvalidArgument,
          "Non-finite codec sample");
      pcm.push_back(
          static_cast<int16_t>(std::clamp(samples[i], -1.0f, 1.0f) * 32767.0f));
    }
    emitted = end;
  }
  ET_CHECK_OR_RETURN_ERROR(
      pcm.size() == total_samples,
      Internal,
      "Codec did not emit every sample exactly once");
  ET_LOG(
      Info,
      "Codec: %zu frames -> %.2f s of audio in %ld ms",
      frames.size(),
      pcm.size() / static_cast<double>(config.sample_rate),
      time_in_ms() - t0);
  std::ofstream wav(wav_path_, std::ios::binary);
  ET_CHECK_OR_RETURN_ERROR(
      wav.is_open(), Internal, "Cannot open speech output file");
  auto put = [&](uint32_t v, int bytes) {
    for (int i = 0; i < bytes; ++i) {
      wav.put(static_cast<char>((v >> (8 * i)) & 0xff));
    }
  };
  const uint32_t data_bytes =
      static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
  wav.write("RIFF", 4);
  put(36 + data_bytes, 4);
  wav.write("WAVEfmt ", 8);
  put(16, 4);
  put(1, 2);
  put(1, 2);
  put(config.sample_rate, 4);
  put(config.sample_rate * 2, 4);
  put(2, 2);
  put(16, 2);
  wav.write("data", 4);
  put(data_bytes, 4);
  for (int16_t sample : pcm) {
    put(static_cast<uint16_t>(sample), 2);
  }
  wav.close();
  ET_CHECK_OR_RETURN_ERROR(
      !wav.fail(), Internal, "Writing speech output failed");
  return Error::Ok;
}

Result<ModelVersion> QNNMultimodalRunner::get_model_version() {
  if (!is_loaded()) {
    stats_.model_load_start_ms = time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_.model_load_end_ms = time_in_ms();
  }
  return model_version_;
}

Result<MethodMeta> QNNMultimodalRunner::get_encoder_method_meta() {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }
  return encoder_->method_meta(kEncoderForwardName);
}

} // namespace example
