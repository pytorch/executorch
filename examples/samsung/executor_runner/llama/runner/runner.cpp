/*
 * Copyright (c) Samsung Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/runtime.h>

#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/examples/samsung/executor_runner/llama/runner/decoder.h>
#include <executorch/examples/samsung/executor_runner/llama/runner/io_manager.h>
#include <executorch/examples/samsung/executor_runner/llama/runner/prefiller.h>
#include <executorch/examples/samsung/executor_runner/llama/runner/runner.h>
#include <executorch/examples/samsung/executor_runner/llama/runner/constants.h>


namespace example {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace llm = ::executorch::extension::llm;

/**
 * @brief Loads a tokenizer from the specified path
 *
 * This function creates and initializes a tokenizer from a file, with options
 * to customize special tokens and regex patterns.
 *
 * @param tokenizer_path Path to the tokenizer file
 * @param special_tokens Optional list of special tokens to add to the tokenizer
 * @param pattern Optional regex pattern for tokenization
 * @param bos_token_index Index of the beginning-of-sequence token
 * @param eos_token_index Index of the end-of-sequence token
 * @return std::unique_ptr<tokenizers::Tokenizer> Initialized tokenizer instance
 */
ET_EXPERIMENTAL std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(
    const std::string& tokenizer_path,
    std::unique_ptr<std::vector<std::string>> special_tokens = nullptr,
    std::optional<std::string> pattern = std::nullopt,
    size_t bos_token_index = 0,
    size_t eos_token_index = 1);

/**
 * @brief Creates a LLMRunner instance with the specified model and
 * tokenizer
 *
 * This factory function creates and initializes a LLMRunner with all
 * necessary components for text generation using the specified model and
 * tokenizer.
 *
 * @param model_path Path to the model file
 * @param tokenizer Initialized tokenizer instance
 * @param data_path Optional path to additional data required by the model
 * @param decode_only Optional flag to run with decode method only
 * @param temperature Optional temperature parameter for controlling randomness
 * (deprecated)
 * @return std::unique_ptr<LLMRunner> Initialized LLMRunner instance
 */
ET_EXPERIMENTAL std::unique_ptr<LLMRunner> create_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path = std::nullopt,
    bool decode_only = false,
    float temperature = -1.0f);

std::unique_ptr<::tokenizers::Tokenizer> load_llama_tokenizer(
    const std::string& tokenizer_path,
    Version version) {
  auto special_tokens = get_special_tokens(version);
  return load_tokenizer(tokenizer_path, std::move(special_tokens));
}

std::unique_ptr<llm::IRunner> create_llama_runner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    std::optional<const std::string> data_path,
    bool decode_only,
    float temperature) {
  ET_LOG(
      Info,
      "Creating LLaMa runner: model_path=%s, tokenizer_path=%s",
      model_path.c_str(),
      tokenizer_path.c_str());

  // Create and load tokenizer
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer =
      load_llama_tokenizer(tokenizer_path, Version::Default);
  if (tokenizer == nullptr) {
    ET_LOG(
        Info,
        "Failed to load %s as a Tiktoken, Sentencepiece or Llama2.c tokenizer, make sure the artifact is one of these types",
        tokenizer_path.c_str());
    return nullptr;
  }
  ET_LOG(
      Info,
      "[Create_test_llm_runner] %s %s",
      model_path.c_str(),
      data_path->c_str());
  return create_llm_runner(model_path, std::move(tokenizer), data_path, decode_only);
}

static const std::unordered_map<std::string, std::string> kMetadataMap{
    {llm::kEnableDynamicShape, "enable_dynamic_shape"},
    {llm::kBosId, "get_bos_id"},
    {llm::kEosIds, "get_eos_ids"},
    {llm::kMaxSeqLen, "get_max_seq_len"},
    {llm::kMaxContextLen, "get_max_context_len"},
    {llm::kVocabSize, "get_vocab_size"},
    {llm::kUseKVCache, "use_kv_cache"},
    {llm::kUseSDPAWithKVCache, "use_sdpa_with_kv_cache"},
    {example::kPrefillArLen, "get_prefill_ar_len"},
    {example::kNumOfHiddenDim, "get_hidden_dim"},
    {example::kNumOfHeads, "get_n_heads"},
    {example::kNumOfKVHeads, "get_n_kv_heads"},
    {example::kNumOfLayers, "get_n_layers"}
};

LLMRunner::LLMRunner(
    std::unordered_map<std::string, int64_t> metadata,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<::executorch::extension::Module> module,
    std::unique_ptr<llm::TextDecoderRunner> text_decoder_runner,
    std::unique_ptr<llm::TextPrefiller> text_prefiller,
    std::unique_ptr<example::TokenGenerator> text_token_generator,
    std::unique_ptr<llm::Stats> stats,
    std::unique_ptr<llm::IOManager> io_manager,
    float temperature)
    : tokenizer_(std::move(tokenizer)),
      metadata_(std::move(metadata)),
      module_(std::move(module)),
      text_decoder_runner_(std::move(text_decoder_runner)),
      text_prefiller_(std::move(text_prefiller)),
      text_token_generator_(std::move(text_token_generator)),
      io_manager_(std::move(io_manager)),
      stats_(std::move(stats)),
      temperature_(temperature) {
  // Note: This constructor assumes that text_prefiller and text_token_generator
  // already have references to the Module and TextDecoderRunner they need
}

bool LLMRunner::is_loaded() const {
  return text_prefiller_->is_loaded() && text_token_generator_->is_loaded();
}

Error LLMRunner::load() {
  ET_CHECK_OK_OR_RETURN_ERROR(text_decoder_runner_->load());
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

Error LLMRunner::generate_from_pos(
    const std::string& prompt,
    int64_t start_pos,
    const llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const llm::Stats&)> stats_callback) {
  // Prepare the inputs.
  // Use ones-initialized inputs.
  ET_CHECK_MSG(!prompt.empty(), "Prompt cannot be null");
  if (!is_loaded()) {
    stats_->model_load_start_ms = llm::time_in_ms();
    ET_CHECK_OK_OR_RETURN_ERROR(load());
    stats_->model_load_end_ms = llm::time_in_ms();
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

  stats_->inference_start_ms = llm::time_in_ms();
  shouldStop_ = false;

  ET_LOG(
      Info,
      "################ input prompt  ################\n %s\n",
      prompt.c_str());

  ::tokenizers::Result<std::vector<uint64_t>> encode_res = tokenizer_->encode(
      prompt,
      /*bos=*/config.num_bos,
      /*eos=*/config.num_eos);

  ET_CHECK_TK_OK_OR_RETURN_ERROR(
      encode_res.error(), "Failed to encode prompt %s", prompt.c_str());

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();

  // Reduce max_context_len by start_pos
  int64_t max_context_len = metadata_.at(llm::kMaxContextLen) - start_pos;
  ET_CHECK_OR_RETURN_ERROR(
      num_prompt_tokens >= 1,
      InvalidArgument,
      "Expected at least 1 prompt token");
  ET_CHECK_OR_RETURN_ERROR(
      num_prompt_tokens < max_context_len,
      InvalidArgument,
      "num_prompt_tokens %d >= max_context_len %" PRId64
      ", Max seq length exceeded - please increase max seq len value in your export script",
      num_prompt_tokens,
      max_context_len);

  // Determine max_new_tokens using the GenerationConfig's resolve method,
  // then subtract start_pos for max_new_tokens.
  int max_new_tokens =
      config.resolve_max_new_tokens(max_context_len, num_prompt_tokens);

  ET_LOG(
      Info,
      "Max new tokens resolved: %d, given start_pos %" PRId64
      ", num_prompt_tokens %zu, max_context_len %" PRId64,
      max_new_tokens,
      start_pos,
      prompt_tokens.size(),
      max_context_len);
  ET_CHECK_OR_RETURN_ERROR(
      max_new_tokens > 0,
      InvalidArgument,
      "Max new tokens %d is less than or equal to 0",
      max_new_tokens);
  // Prefill first
  // Here feed all tokens to the model and get the next predicted token
  // after the prompt. After that we will enter generate loop.

  // print prompts
  if (config.echo) {
    wrapped_callback(prompt);
  }
  int64_t pos = start_pos;
  auto prefill_res = text_prefiller_->prefill(prompt_tokens, pos);
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  uint64_t cur_token = prefill_res.get();
  stats_->first_token_ms = llm::time_in_ms();
  stats_->prompt_eval_end_ms = llm::time_in_ms();

  RUNNER_ET_LOG(
      config.warming,
      "\nRSS after prompt prefill: %f MiB (0 if unsupported)",
      llm::get_rss_bytes() / 1024.0 / 1024.0);

  ::tokenizers::Result<std::vector<uint64_t>> newline_token =
      tokenizer_->encode("\n");

  ET_LOG(Info, "[ENN] first token %lu\n", cur_token);

  ET_LOG(
      Info,
      "################ The start of token generation ################\n");

  // print the first token from prefill. No prev_token so use cur_token for it.
  wrapped_callback(
      ET_UNWRAP_TOKENIZER(tokenizer_->decode(cur_token, cur_token)));

  // start the main loop
  prompt_tokens.push_back(cur_token);

  // Generate max_new_tokens - 1 because prefill already generated 1 token.
  int64_t num_generated_tokens = ET_UNWRAP(text_token_generator_->generate(
      prompt_tokens,
      num_prompt_tokens,
      max_new_tokens - 1,
      temperature_ == -1.0f ? config.temperature : temperature_,
      wrapped_callback));

  // A temporary new line to separate stdout mixed with results.
  wrapped_callback(ET_UNWRAP_TOKENIZER(
      tokenizer_->decode(newline_token.get()[0], newline_token.get()[0])));

  ET_LOG(
      Info, "################ The end of token generation  ################\n");

  stats_->inference_end_ms = llm::time_in_ms();
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

Error LLMRunner::generate(
    const std::string& prompt,
    const llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const llm::Stats&)> stats_callback) {
  return generate_from_pos(prompt, 0, config, token_callback, stats_callback);
}

Error LLMRunner::warmup(const std::string& prompt, int32_t max_new_tokens) {
  // Create a GenerationConfig for warmup
  llm::GenerationConfig config{
      .echo = false, .max_new_tokens = max_new_tokens, .warming = true};

  // Call generate with the warmup config
  Error err = generate(prompt, config);

  // Reset stats after warmup, not resetting the std::unique_ptr!
  stats_->reset();
  return err;
}

void LLMRunner::stop() {
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
  ::executorch::runtime::runtime_init();
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

  auto json_tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (json_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded json tokenizer");
    return json_tokenizer;
  }

  auto sp_tokenizer = std::make_unique<::tokenizers::SPTokenizer>();
  if (sp_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded Sentencepiece tokenizer");
    return sp_tokenizer;
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
      {kMetadataMap.at(llm::kEnableDynamicShape), false},
      {kMetadataMap.at(llm::kMaxSeqLen), 128},
      {kMetadataMap.at(llm::kMaxContextLen), 128},
      {kMetadataMap.at(llm::kUseKVCache), true},
      {kMetadataMap.at(llm::kUseSDPAWithKVCache), false},
      {kMetadataMap.at(example::kPrefillArLen), 0},
      //Params.json for llama3.2-1B
      {kMetadataMap.at(example::kNumOfHiddenDim), 2048},
      {kMetadataMap.at(example::kNumOfHeads), 32},
      {kMetadataMap.at(example::kNumOfKVHeads), 8},
      {kMetadataMap.at(example::kNumOfLayers), 16},
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

std::unique_ptr<LLMRunner> create_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path,
    bool decode_only,
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

  // Get methods
  executorch::runtime::Result<std::unordered_set<std::string>> methods_res =
      module->method_names();
  if (methods_res.error() != executorch::runtime::Error::Ok) {
    ET_CHECK_MSG(false, "Failed to get method names");
  }

  auto methods = methods_res.get();
  std::vector<std::string> register_methods;
  if (decode_only) {
    register_methods.emplace_back(default_method_name_decode);
  } else {
    for (auto method : methods) {
      if (kMetadataMap.find(method) != kMetadataMap.end()) {
        ET_LOG(Info, "It is not method(%s)", method.c_str());
        continue;
      }
      ET_LOG(Info, "Register method \"%s\" to decoder", method.c_str());
      register_methods.emplace_back(method);
    }
 }

  // Get metadata from Module
  ET_LOG(Info, "Reading metadata from model");
  auto metadata = get_llm_metadata(tokenizer.get(), module.get());

  // Create IO Manager
  example::LlmIOManager::Config conf;
  conf.n_layers = metadata.at(example::kNumOfLayers);
  conf.n_kv_heads = metadata.at(example::kNumOfKVHeads);
  conf.max_seq_len = metadata.at(llm::kMaxSeqLen);
  conf.prefill_seq_len = (metadata.at(example::kPrefillArLen) > 1) ?
                            (metadata.at(example::kPrefillArLen)) :
                            (metadata.at(llm::kMaxSeqLen));

  if (decode_only) {
    conf.prefill_seq_len = 1;
  }

  size_t dim = metadata.at(example::kNumOfHiddenDim);
  size_t n_heads = metadata.at(example::kNumOfHeads);
  conf.head_dim = dim / n_heads;

  auto io_manager = std::make_unique<example::LlmIOManager>(*module.get(), conf);

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      get_eos_ids(tokenizer.get(), module.get()));

  // Create text_decoder_runner. Use a shared_ptr so that it can be shared with
  // TextPrefiller and TextTokenGenerator
  auto text_decoder_runner =
      std::make_unique<example::Decoder>(module.get(), io_manager.get());

  // Register methods to cope with multi-methods
  for (auto method : register_methods) {
    text_decoder_runner->register_method(method);
  }

  // TODO(sicheon.oh) : Need to find proper flag to distinguish separaged prefill/decode opion.
  bool separated_prefill_decode = (conf.prefill_seq_len > 1);
  ET_LOG(Info, "separated_prefill_decode(%s)", (separated_prefill_decode)?"True":"False");

  // Create text_prefiller
  auto text_prefiller = std::make_unique<example::Prefiller>(
      text_decoder_runner.get(),
      metadata.at(llm::kUseKVCache),
      metadata.at(llm::kEnableDynamicShape) | (separated_prefill_decode),
      conf.prefill_seq_len);

  // Create text_token_generator with stats
  auto stats = std::make_unique<llm::Stats>();
  auto text_token_generator = std::make_unique<example::TokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(llm::kUseKVCache),
      std::move(eos_ids),
      stats.get());

  // Create and return the Runner instance
  std::unique_ptr<LLMRunner> runner = std::make_unique<LLMRunner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::move(text_prefiller),
      std::move(text_token_generator),
      std::move(stats),
      std::move(io_manager),
      temperature);

  runner->load();

  return runner;
}
} // namespace example
