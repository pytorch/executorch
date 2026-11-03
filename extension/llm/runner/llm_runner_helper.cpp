/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
// Implementation of helper utilities for creating and configuring LLM runners

#include <executorch/extension/llm/runner/image_prefiller.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>

#include <executorch/extension/llm/runner/io_manager/kv_cache_io_manager.h>
#include <executorch/extension/llm/runner/multimodal_decoder_runner.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/memory_allocator/cpu_caching_malloc_allocator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/tekken.h>
#include <pytorch/tokenizers/tiktoken.h>

#include <limits>

namespace executorch::extension::llm {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;

std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(
    const std::string& tokenizer_path,
    std::unique_ptr<std::vector<std::string>> special_tokens,
    std::optional<std::string> pattern,
    size_t bos_token_index,
    size_t eos_token_index) {
  runtime::runtime_init();
  auto tekken_tokenizer = std::make_unique<tokenizers::Tekken>();
  // Prevent the case where tekken tokenizer accidentally successfully loads a
  // HuggingFace tokenizer, which is also .json.
  static constexpr std::string_view tekken_name = "tekken.json";
  if (tokenizer_path.size() >= tekken_name.size() &&
      tokenizer_path.rfind(tekken_name) ==
          tokenizer_path.size() - tekken_name.size()) {
    if (tekken_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
      ET_LOG(Info, "Loaded tekken tokenizer");
      return tekken_tokenizer;
    }
  }
  auto json_tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (json_tokenizer->load(tokenizer_path) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded json tokenizer");
    return json_tokenizer;
  }
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

::executorch::runtime::Result<std::unordered_map<std::string, int64_t>>
get_llm_metadata(tokenizers::Tokenizer* tokenizer, Module* module) {
  // Initialize metadata with default values
  std::unordered_map<std::string, int64_t> metadata({
      {llm::kEnableDynamicShape, false},
      {llm::kMaxSeqLen, 128},
      {llm::kMaxContextLen, 128},
      {llm::kUseKVCache, true},
      {llm::kUseSDPAWithKVCache, false},
  });

  // Read metadata from the model
  auto method_names_result = module->method_names();
  if (method_names_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed reading method names");
    return ::executorch::runtime::Error::InvalidArgument;
  }
  const auto& method_names = method_names_result.get();

  // Error out if the max seq len metadata method is not present, since
  // it is hard to figure out from just the .pte itself.
  if (!method_names.count(llm::kMaxSeqLen)) {
    ET_LOG(
        Error,
        "Required metadata method %s not found in model",
        llm::kMaxSeqLen);
    return ::executorch::runtime::Error::InvalidArgument;
  }

  for (auto& pair : metadata) {
    const auto& method_name = pair.first;
    auto& value = pair.second;

    if (method_names.count(method_name)) {
      auto get_result = module->get(method_name);
      if (!get_result.ok()) {
        return get_result.error();
      }
      if (!get_result->isInt() && !get_result->isBool()) {
        ET_LOG(
            Error,
            "Metadata method %s must return an integer or boolean.",
            method_name.c_str());
        return Error::InvalidArgument;
      }
      value = get_result->toScalar().to<decltype(metadata)::mapped_type>();
    } else {
      ET_LOG(
          Info,
          "Method %s not found, using the default value %" PRId64,
          method_name.c_str(),
          value);
    }
    ET_LOG(Info, "Metadata: %s = %" PRId64, method_name.c_str(), value);
  }

  // If kMaxContextLen method not found but kMaxSeqLen is
  // available, set kMaxContextLen to the value of kMaxSeqLen.
  if (!method_names.count(llm::kMaxContextLen) &&
      method_names.count(llm::kMaxSeqLen)) {
    metadata[llm::kMaxContextLen] = metadata[llm::kMaxSeqLen];
    ET_LOG(
        Info,
        "Setting kMaxContextLen to kMaxSeqLen value: %" PRId64,
        metadata[llm::kMaxContextLen]);
  }

  for (const auto* name : {kMaxSeqLen, kMaxContextLen}) {
    const auto value = metadata.at(name);
    ET_CHECK_OR_RETURN_ERROR(
        value > 0 && value <= std::numeric_limits<int32_t>::max(),
        InvalidArgument,
        "Metadata %s must be in [1, INT32_MAX].",
        name);
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
  const auto& method_names = method_names_result.get();

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

std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path,
    float temperature,
    const std::string& method_name,
    Module::LoadMode load_mode) {
  // An empty path is not a path. Callers that build the optional from a
  // language whose "no value" is an empty string would otherwise reach the
  // loader with "", which fails to open and takes the whole load down with it.
  if (data_path.has_value() && !data_path.value().empty()) {
    std::vector<std::string> data_files;
    data_files.push_back(data_path.value());
    return create_text_llm_runner(
        model_path,
        std::move(tokenizer),
        std::move(data_files),
        temperature,
        nullptr,
        method_name,
        load_mode);
  }
  return create_text_llm_runner(
      model_path,
      std::move(tokenizer),
      std::vector<std::string>(),
      temperature,
      nullptr,
      method_name,
      load_mode);
}

std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::vector<std::string> data_files,
    float temperature,
    std::unique_ptr<::executorch::runtime::EventTracer> event_tracer,
    const std::string& method_name,
    Module::LoadMode load_mode) {
  // Sanity check tokenizer
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer is null or not loaded");
    return nullptr;
  }

  // Create the Module
  std::unique_ptr<Module> module;
  uint32_t max_cached_memory_size_bytes_ = 1024 * 1024 * 10; // 10MB
  if (data_files.size() > 0) {
    module = std::make_unique<Module>(
        model_path,
        data_files,
        load_mode,
        std::move(event_tracer),
        nullptr, // memory allocator
        std::make_unique<
            executorch::extension::CPUCachingAllocator>( // temp memory
                                                         // allocator
            max_cached_memory_size_bytes_));
  } else {
    module = std::make_unique<Module>(
        model_path,
        load_mode,
        std::move(event_tracer), // event tracer
        nullptr, // memory allocator
        std::make_unique<
            executorch::extension::CPUCachingAllocator>( // temp memory
                                                         // allocator
            max_cached_memory_size_bytes_));
  }

  // Get metadata from Module
  ET_LOG(Info, "Reading metadata from model");
  auto metadata_result = llm::get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to get metadata from model");
    return nullptr;
  }
  auto metadata = metadata_result.get();

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      llm::get_eos_ids(tokenizer.get(), module.get()));

  // Create the IOManager for the I/O contract the program declares, if any.
  register_builtin_io_managers();
  std::string io_contract;
  if (auto names = module->method_names();
      names.ok() && names->count(kIoContract)) {
    auto contract = module->get(kIoContract);
    if (!contract.ok() || !contract->isString()) {
      ET_LOG(Error, "Failed to read the program's I/O contract label.");
      return nullptr;
    }
    io_contract = std::string(contract->toString());
    if (io_contract.empty()) {
      ET_LOG(Error, "The program's I/O contract label is empty.");
      return nullptr;
    }
    ET_LOG(Info, "I/O contract: %s", io_contract.c_str());
  }
  auto io_manager_result = make_io_manager(*module, io_contract);
  if (!io_manager_result.ok()) {
    ET_LOG(
        Error,
        "Failed to create the IOManager: 0x%x.",
        static_cast<unsigned>(io_manager_result.error()));
    return nullptr;
  }
  auto io_manager = std::move(*io_manager_result);

  // Read vocab_size for Sampler
  int32_t vocab_size = static_cast<int32_t>(metadata.at(kVocabSize));
  float init_temp = temperature == -1.0f ? 0.0f : temperature;
  auto sampler = std::make_unique<Sampler>(vocab_size, init_temp);

  auto stats = std::make_unique<Stats>();

  auto method_names_result = module->method_names();
  const std::unordered_set<std::string> method_names = method_names_result.ok()
      ? method_names_result.get()
      : std::unordered_set<std::string>{};

  // A two-method PTE exports kPrefillMethod/kDecodeMethod instead of a single
  // forward method, so fall back to kPrefillMethod when the requested method is
  // absent. An explicitly requested method that does exist always wins.
  std::string prefill_method_name = method_name;
  if (method_names.count(prefill_method_name) == 0 &&
      method_names.count(kPrefillMethod) > 0) {
    prefill_method_name = kPrefillMethod;
  }
  // An IOManager may drive methods of its own contract's names.
  auto [io_prefill, io_decode] = io_manager->method_names(method_names);
  if (!io_prefill.empty()) {
    prefill_method_name = io_prefill;
  }
  if (io_manager->load(
          prefill_method_name,
          io_decode.empty() ? prefill_method_name : io_decode) != Error::Ok) {
    ET_LOG(Error, "Failed to load the IOManager");
    return nullptr;
  }

  // Create text_decoder_runner
  ET_LOG(Info, "Using method: %s", prefill_method_name.c_str());
  auto text_decoder_runner = std::make_unique<TextDecoderRunner>(
      module.get(),
      io_manager.get(),
      prefill_method_name,
      std::move(sampler),
      stats.get());

  // The decode stage gets its own decoder runner when the PTE exports a
  // separate decode method; otherwise both stages share text_decoder_runner.
  std::unique_ptr<TextDecoderRunner> decode_text_decoder_runner;
  if (!io_decode.empty() && io_decode != prefill_method_name) {
    decode_text_decoder_runner = std::make_unique<TextDecoderRunner>(
        module.get(),
        io_manager.get(),
        io_decode,
        std::make_unique<Sampler>(vocab_size, init_temp),
        stats.get());
  } else if (
      io_decode.empty() && method_names.count(kDecodeMethod) > 0 &&
      prefill_method_name != kDecodeMethod) {
    ET_LOG(Info, "Using decode method: %s", kDecodeMethod);
    decode_text_decoder_runner = std::make_unique<TextDecoderRunner>(
        module.get(),
        io_manager.get(),
        kDecodeMethod,
        std::make_unique<Sampler>(vocab_size, init_temp),
        stats.get());
  }
  TextDecoderRunner* decode_runner = decode_text_decoder_runner
      ? decode_text_decoder_runner.get()
      : text_decoder_runner.get();

  // Create text_prefiller. A prompt method with a fixed step size takes the
  // prompt in chunks of that size.
  const int64_t step_tokens =
      io_manager->max_tokens_per_step(prefill_method_name);
  auto text_prefiller = std::make_unique<TextPrefiller>(
      text_decoder_runner.get(),
      metadata.at(kUseKVCache),
      step_tokens > 1 ? true
                      : static_cast<bool>(metadata.at(kEnableDynamicShape)),
      step_tokens > 1 ? step_tokens : metadata.at(kMaxSeqLen),
      step_tokens > 1 ? decode_runner : nullptr);

  // Create text_token_generator with stats
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      decode_runner,
      metadata.at(kUseKVCache),
      std::move(eos_ids),
      stats.get());

  // Create and return the Runner instance
  return std::make_unique<TextLLMRunner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::move(text_prefiller),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats),
      temperature,
      std::move(decode_text_decoder_runner));
}

std::unique_ptr<MultimodalRunner> create_multimodal_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path,
    Module::LoadMode load_mode) {
  // Sanity check tokenizer
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer is null or not loaded");
    return nullptr;
  }

  // Create the Module
  std::unique_ptr<Module> module;
  if (data_path.has_value() && !data_path.value().empty()) {
    module = std::make_unique<Module>(model_path, data_path.value(), load_mode);
  } else {
    module = std::make_unique<Module>(model_path, load_mode);
  }

  // Get metadata from Module
  ET_LOG(Info, "Reading metadata from model");
  auto metadata_result = get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to get metadata from model");
    return nullptr;
  }
  auto metadata = metadata_result.get();

  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      get_eos_ids(tokenizer.get(), module.get()));

  // Create IOManager
  std::unique_ptr<IOManager> io_manager = std::make_unique<IOManager>(*module);

  // Read vocab_size for Sampler
  int32_t vocab_size = static_cast<int32_t>(metadata.at(kVocabSize));
  auto sampler = std::make_unique<Sampler>(vocab_size, 0.0f); // Default temp

  // Create text_decoder_runner
  auto text_decoder_runner = std::make_unique<MultimodalDecoderRunner>(
      module.get(), io_manager.get(), "forward", std::move(sampler));

  // Create multimodal_prefiller
  auto multimodal_prefiller = std::make_unique<MultimodalPrefiller>(
      module.get(),
      text_decoder_runner.get(),
      tokenizer.get(),
      io_manager.get());

  // Create text_token_generator with stats
  auto stats = std::make_unique<Stats>();
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
      metadata.at(kUseKVCache),
      std::move(eos_ids),
      stats.get());

  // Create and return the MultimodalRunner instance
  return std::make_unique<MultimodalRunner>(
      std::move(metadata),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::move(multimodal_prefiller),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));
}

} // namespace executorch::extension::llm
