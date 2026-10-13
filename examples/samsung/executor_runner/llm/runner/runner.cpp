/*
 * Copyright (c) Samsung Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple LLM runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#include <executorch/examples/samsung/executor_runner/llm/runner/config.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/runtime.h>

#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/config.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/decoder.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/io_manager.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/prefiller.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/runner.h>
#include <nlohmann/json.hpp>
#include <fstream>

namespace example {

namespace llm = ::executorch::extension::llm;
namespace runtime = ::executorch::runtime;
namespace aten = ::executorch::aten;
namespace extension = ::executorch::extension;

using json = nlohmann::json;

std::string LLMRunner::get_formatted_prompt(
    const std::string& prompt,
    const std::string& system_prompt,
    DecoderModelVersion version) {
  std::string formatted_prompt;
  switch (version) {
    case DecoderModelVersion::kLlama3:
      if (!system_prompt.empty()) {
        formatted_prompt.append(
            "<|start_header_id|>system<|end_header_id|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|eot_id|>");
      }
      formatted_prompt.append("<|start_header_id|>user<|end_header_id|>\n\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append(
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
      break;
    case DecoderModelVersion::kGemma3:
      formatted_prompt.append("<start_of_turn>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_turn>\n");
      formatted_prompt.append("<start_of_turn>model\n");
      if (!system_prompt.empty()) {
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<end_of_turn>\n");
      }
      break;
    case DecoderModelVersion::kGemma4:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|turn>system ");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<turn|>\n");
      }
      formatted_prompt.append("<|turn>user ");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<turn|>\n");
      formatted_prompt.append("<|turn>model\n");
      break;
    case DecoderModelVersion::kGeneral:
      formatted_prompt.append(prompt);
      break;
    default:
      formatted_prompt.append(prompt);
      break;
  }
  return formatted_prompt;
}

/**
 * This function creates and initializes a tokenizer from a file based on the
 * specified type. Supported types are: "bpe", "tiktoken", "hf", and "sp".
 */
ET_EXPERIMENTAL std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(
    const std::string& type,
    const std::string& tokenizer_path,
    Version version);

std::unique_ptr<llm::IRunner> create_llm_runner(
    const RunnerPaths& runner_paths,
    RunnerConfig runner_config) {
  ET_LOG(
      Info,
      "Creating LLM runner: model_path=%s, tokenizer_path=%s",
      runner_paths.model_path.c_str(),
      runner_paths.tokenizer_path.c_str());

  runner_config.load_from_json();

  std::unique_ptr<::tokenizers::Tokenizer> tokenizer = load_tokenizer(
      runner_config.tokenizer_type,
      runner_paths.tokenizer_path,
      Version::Default);
  if (tokenizer == nullptr) {
    ET_LOG(
        Info,
        "Failed to load tokenizer. Parsed tokenizer type is %s, located at %s.",
        runner_config.tokenizer_type.c_str(),
        runner_paths.tokenizer_path.c_str());
    return nullptr;
  }
  ET_LOG(
      Info,
      "[create_llm_runner] %s %s",
      runner_paths.model_path.c_str(),
      runner_paths.data_path.has_value()
          ? runner_paths.data_path.value().c_str()
          : "null");
  return create_llm_runner(
      runner_paths, std::move(runner_config), std::move(tokenizer));
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
    {llm::kHeadDim, "get_head_dim"},
    {example::kPrefillArLen, "get_prefill_ar_len"},
    {example::kNumOfHiddenDim, "get_hidden_dim"},
    {example::kNumOfHeads, "get_n_heads"},
    {example::kNumOfKVHeads, "get_n_kv_heads"},
    {example::kNumOfLayers, "get_n_layers"},
    {"get_sliding_window", "get_sliding_window"}};

LLMRunner::LLMRunner(
    std::unordered_map<std::string, int64_t> metadata,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<extension::Module> module,
    std::unique_ptr<llm::TextDecoderRunner> text_decoder_runner,
    std::unique_ptr<llm::TextPrefiller> text_prefiller,
    std::unique_ptr<example::TokenGenerator> text_token_generator,
    std::unique_ptr<llm::Stats> stats,
    std::unique_ptr<llm::IOManager> io_manager,
    RunnerConfig config,
    DecoderModelVersion model_version,
    float temperature)
    : config_(std::move(config)),
      tokenizer_(std::move(tokenizer)),
      metadata_(std::move(metadata)),
      module_(std::move(module)),
      text_decoder_runner_(std::move(text_decoder_runner)),
      text_prefiller_(std::move(text_prefiller)),
      text_token_generator_(std::move(text_token_generator)),
      io_manager_(std::move(io_manager)),
      stats_(std::move(stats)),
      temperature_(temperature),
      model_version_(model_version) {
  // Note: This constructor assumes that text_prefiller and text_token_generator
  // already have references to the Module and TextDecoderRunner they need
}

bool LLMRunner::is_loaded() const {
  return text_prefiller_->is_loaded() && text_token_generator_->is_loaded();
}

runtime::Error LLMRunner::load() {
  ET_CHECK_OK_OR_RETURN_ERROR(text_decoder_runner_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(text_prefiller_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(text_token_generator_->load());
  return runtime::Error::Ok;
}

// Don't print with the same priority during warmup
#define RUNNER_ET_LOG(is_warmup, format, ...) \
  if (is_warmup) {                            \
    ET_LOG(Debug, format, __VA_ARGS__);       \
  } else {                                    \
    ET_LOG(Info, format, __VA_ARGS__);        \
  }

void replace_all_newlines(std::string& str) {
  std::string from = "\\n";
  std::string to = "\n";

  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
}

runtime::Error LLMRunner::generate_from_pos(
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

  std::string input_prompt = get_formatted_prompt(prompt, "", model_version_);
  if (model_version_ == DecoderModelVersion::kGemma3) {
    replace_all_newlines(input_prompt);
  }

  ::tokenizers::Result<std::vector<uint64_t>> encode_res = tokenizer_->encode(
      input_prompt,
      /*bos=*/config.num_bos,
      /*eos=*/config.num_eos);

  ET_CHECK_TK_OK_OR_RETURN_ERROR(
      encode_res.error(), "Failed to encode prompt %s", input_prompt.c_str());

  // encode the (string) prompt into tokens sequence
  std::vector<uint64_t> prompt_tokens = encode_res.get();
  int num_prompt_tokens = prompt_tokens.size();

  // Reduce max_context_len by start_pos
  // TODO(brucekk.kim): Check where to fetch from: metatdata or runner_config
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
    wrapped_callback(input_prompt);
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

  return runtime::Error::Ok;
}

runtime::Error LLMRunner::generate(
    const std::string& prompt,
    const llm::GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const llm::Stats&)> stats_callback) {
  return generate_from_pos(prompt, 0, config, token_callback, stats_callback);
}

runtime::Error LLMRunner::warmup(
    const std::string& prompt,
    int32_t max_new_tokens) {
  // Create a GenerationConfig for warmup
  llm::GenerationConfig config{
      .echo = false, .max_new_tokens = max_new_tokens, .warming = true};

  // Call generate with the warmup config
  runtime::Error err = generate(prompt, config);

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
    const std::string& type,
    const std::string& tokenizer_path,
    Version version) {
  ::executorch::runtime::runtime_init();

  std::unique_ptr<::tokenizers::Tokenizer> tokenizer;
  if (type == "bpe") {
    tokenizer = std::make_unique<::tokenizers::Llama2cTokenizer>();
  } else if (type == "tiktoken") {
    tokenizer = example::get_tiktoken_for_llama(version);
  } else if (type == "hf") {
    tokenizer = std::make_unique<::tokenizers::HFTokenizer>();
  } else if (type == "sp") {
    tokenizer = std::make_unique<::tokenizers::SPTokenizer>();
  } else {
    return nullptr;
  }

  if (tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded %s tokenizer", type.c_str());
    return tokenizer;
  }

  return nullptr;
}

std::unordered_map<std::string, int64_t> get_llm_metadata(
    tokenizers::Tokenizer* tokenizer,
    extension::Module* module,
    RunnerConfig& config) {
  // Initialize metadata from JSON config values
  std::unordered_map<std::string, int64_t> metadata({
      {kMetadataMap.at(llm::kEnableDynamicShape), false},
      {kMetadataMap.at(llm::kMaxSeqLen),
       config.model_args ? config.model_args->get_max_seq_len() : 128},
      {kMetadataMap.at(llm::kMaxContextLen),
       config.model_args ? config.model_args->get_max_context_len() : 128},
      {kMetadataMap.at(llm::kUseKVCache), true},
      {kMetadataMap.at(llm::kUseSDPAWithKVCache), false},
      {kMetadataMap.at(example::kPrefillArLen), 0},
      {kMetadataMap.at(example::kNumOfHiddenDim),
       config.model_args ? config.model_args->get_hidden_dim() : 0},
      {kMetadataMap.at(example::kNumOfHeads),
       config.model_args ? config.model_args->get_n_heads() : 0},
      {kMetadataMap.at(example::kNumOfKVHeads),
       config.model_args ? config.model_args->get_n_kv_heads() : 0},
      {kMetadataMap.at(example::kNumOfLayers),
       config.model_args ? config.model_args->get_n_layers() : 0},
      {kMetadataMap.at(llm::kHeadDim),
       config.model_args ? config.model_args->get_head_dim() : 0},
  });

  auto method_names_result = module->method_names();
  if (method_names_result.error() != runtime::Error::Ok) {
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
          Debug,
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
    extension::Module* module) {
  std::unordered_set<uint64_t> eos_ids = {tokenizer->eos_tok()};
  // Get EOS IDs if available
  auto method_names_result = module->method_names();
  if (method_names_result.error() != runtime::Error::Ok) {
    ET_LOG(Error, "Failed reading method names");
    return eos_ids;
  }
  const auto method_names = method_names_result.get();

  if (method_names.count(llm::kEosIds)) {
    eos_ids.clear();
    auto execute_result = module->execute(llm::kEosIds);
    if (execute_result.error() != runtime::Error::Ok) {
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

std::unique_ptr<llm::IRunner> create_llm_runner(
    const RunnerPaths& runner_paths,
    RunnerConfig runner_config,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer) {
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer is null or not loaded");
    return nullptr;
  }

  DecoderModelVersion model_version = runner_config.model_version;
  float temperature = runner_config.temperature;
  bool is_decode_only = runner_config.is_decode_only;

  std::unique_ptr<extension::Module> module;
  if (runner_paths.data_path.has_value()) {
    module = std::make_unique<extension::Module>(
        runner_paths.model_path,
        runner_paths.data_path.value(),
        extension::Module::LoadMode::File);
  } else {
    module = std::make_unique<extension::Module>(
        runner_paths.model_path, extension::Module::LoadMode::File);
  }

  executorch::runtime::Result<std::unordered_set<std::string>> methods_res =
      module->method_names();
  if (methods_res.error() != executorch::runtime::Error::Ok) {
    ET_CHECK_MSG(false, "Failed to get method names");
  }

  auto methods = methods_res.get();
  std::vector<std::string> register_methods;
  if (is_decode_only) {
    register_methods.emplace_back(default_method_name_decode);
  } else {
    for (auto method : methods) {
      if (kMetadataMap.find(method) != kMetadataMap.end()) {
        continue;
      }
      ET_LOG(Info, "Register method: \"%s\"", method.c_str());
      register_methods.emplace_back(method);
    }
  }

  ET_LOG(Info, "Reading metadata from model");
  auto metadata =
      get_llm_metadata(tokenizer.get(), module.get(), runner_config);

  auto stats = std::make_unique<llm::Stats>();
  llm::Stats* stats_ptr = stats.get();

  ET_LOG(Info, "Instantiate LLMRunner with null placeholders");
  std::unique_ptr<LLMRunner> runner = std::make_unique<LLMRunner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      nullptr, // text_decoder_runner placeholder
      nullptr, // text_prefiller placeholder
      nullptr, // text_token_generator placeholder
      std::move(stats),
      nullptr, // io_manager placeholder
      std::move(runner_config),
      model_version,
      temperature);

  const auto& config = runner->get_config();

  ET_LOG(Info, "Instantiate IO Manager");
  auto io_manager =
      std::make_unique<example::LlmIOManager>(*runner->get_module(), config);

  ET_LOG(Info, "Start loading Rotary Embedding");
  if (io_manager->load_rotary_embedding(runner_paths.pos_embedding_path) !=
      executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to load or precompute pos_embedding.");
    return nullptr;
  }

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      get_eos_ids(runner->get_tokenizer(), runner->get_module()));

  auto text_decoder_runner = std::make_unique<example::Decoder>(
      runner->get_module(), io_manager.get());

  for (auto method : register_methods) {
    text_decoder_runner->set_method(method);
  }

  bool separated_prefill_decode =
      (config.model_args->get_prefill_seq_len() > 1);
  ET_LOG(
      Info,
      "separated_prefill_decode(%s)",
      (separated_prefill_decode) ? "True" : "False");

  auto text_prefiller = std::make_unique<example::Prefiller>(
      text_decoder_runner.get(),
      runner->get_metadata().at(llm::kUseKVCache),
      runner->get_metadata().at(llm::kEnableDynamicShape) |
          (separated_prefill_decode),
      config.model_args->get_prefill_seq_len());

  auto text_token_generator = std::make_unique<example::TokenGenerator>(
      runner->get_tokenizer(),
      text_decoder_runner.get(),
      runner->get_metadata().at(llm::kUseKVCache),
      std::move(eos_ids),
      stats_ptr);

  runner->set_io_manager(std::move(io_manager));
  runner->set_text_decoder_runner(std::move(text_decoder_runner));
  runner->set_text_prefiller(std::move(text_prefiller));
  runner->set_text_token_generator(std::move(text_token_generator));

  runner->load();

  return runner;
}
} // namespace example
