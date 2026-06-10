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
#include <executorch/extension/llm/runner/multimodal_decoder_runner.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/memory_allocator/cpu_caching_malloc_allocator.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/tekken.h>
#include <pytorch/tokenizers/tiktoken.h>

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

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

std::unique_ptr<tokenizers::Tokenizer> load_tokenizer_from_buffer(
    const void* data,
    size_t size,
    std::unique_ptr<std::vector<std::string>> special_tokens,
    std::optional<std::string> pattern,
    size_t bos_token_index,
    size_t eos_token_index) {
  runtime::runtime_init();
  auto tekken_tokenizer = std::make_unique<tokenizers::Tekken>();
  if (tekken_tokenizer->load_from_buffer(data, size) ==
      ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded tekken tokenizer from buffer");
    return tekken_tokenizer;
  }

  auto json_tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (json_tokenizer->load_from_buffer(data, size) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded json tokenizer from buffer");
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
  if (tiktoken_tokenizer->load_from_buffer(data, size) ==
      ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded TikToken tokenizer from buffer");
    return tiktoken_tokenizer;
  }

  auto sp_tokenizer = std::make_unique<::tokenizers::SPTokenizer>();
  if (sp_tokenizer->load_from_buffer(data, size) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded Sentencepiece tokenizer from buffer");
    return sp_tokenizer;
  }

  auto bpe_tokenizer = std::make_unique<::tokenizers::Llama2cTokenizer>();
  if (bpe_tokenizer->load_from_buffer(data, size) == ::tokenizers::Error::Ok) {
    ET_LOG(Info, "Loaded BPE tokenizer from buffer");
    return bpe_tokenizer;
  }

  return nullptr;
}

namespace {

constexpr const char* kTokenizerBackendId = "TokenizerBackend";
constexpr const char* kMaxContextLengthSpec = "max_context_length";
constexpr const char* kBosSpec = "bos";
constexpr const char* kEosSpec = "eos";

struct TokenizerDelegateHandle final {
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer;
  size_t max_context_length = 0;
  int8_t bos = 0;
  int8_t eos = 0;
};

Error parse_size_compile_spec(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>
        compile_specs,
    const char* key,
    bool required,
    size_t* out) {
  for (size_t i = 0; i < compile_specs.size(); ++i) {
    const auto& spec = compile_specs[i];
    if (std::strcmp(spec.key, key) != 0) {
      continue;
    }
    std::string value(
        static_cast<const char*>(spec.value.buffer), spec.value.nbytes);
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    ET_CHECK_OR_RETURN_ERROR(
        !value.empty() && value[0] != '-' && errno != ERANGE &&
            end == value.c_str() + value.size(),
        InvalidProgram,
        "Invalid TokenizerBackend compile spec %s=%s",
        key,
        value.c_str());
    ET_CHECK_OR_RETURN_ERROR(
        parsed <= std::numeric_limits<size_t>::max(),
        InvalidProgram,
        "TokenizerBackend compile spec %s is too large",
        key);
    *out = static_cast<size_t>(parsed);
    return Error::Ok;
  }
  ET_CHECK_OR_RETURN_ERROR(
      !required,
      InvalidProgram,
      "Missing TokenizerBackend compile spec %s",
      key);
  return Error::Ok;
}

Error parse_i8_compile_spec(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>
        compile_specs,
    const char* key,
    int8_t* out) {
  size_t parsed = static_cast<size_t>(*out);
  ET_CHECK_OK_OR_RETURN_ERROR(
      parse_size_compile_spec(compile_specs, key, /*required=*/false, &parsed));
  ET_CHECK_OR_RETURN_ERROR(
      parsed <= static_cast<size_t>(std::numeric_limits<int8_t>::max()),
      InvalidProgram,
      "TokenizerBackend compile spec %s is too large for int8_t",
      key);
  *out = static_cast<int8_t>(parsed);
  return Error::Ok;
}

class TokenizerBackend final : public executorch::runtime::BackendInterface {
 public:
  bool is_available() const override {
    return true;
  }

  executorch::runtime::Result<executorch::runtime::DelegateHandle*> init(
      executorch::runtime::BackendInitContext& context,
      executorch::runtime::FreeableBuffer* processed,
      executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>
          compile_specs) const override {
    ET_CHECK_OR_RETURN_ERROR(
        processed != nullptr && processed->data() != nullptr &&
            processed->size() > 0,
        InvalidProgram,
        "TokenizerBackend requires non-empty bundled tokenizer data");

    size_t max_context_length = 0;
    ET_CHECK_OK_OR_RETURN_ERROR(parse_size_compile_spec(
        compile_specs,
        kMaxContextLengthSpec,
        /*required=*/true,
        &max_context_length));
    ET_CHECK_OR_RETURN_ERROR(
        max_context_length > 0,
        InvalidProgram,
        "TokenizerBackend max_context_length must be positive");

    int8_t bos = 0;
    int8_t eos = 0;
    ET_CHECK_OK_OR_RETURN_ERROR(
        parse_i8_compile_spec(compile_specs, kBosSpec, &bos));
    ET_CHECK_OK_OR_RETURN_ERROR(
        parse_i8_compile_spec(compile_specs, kEosSpec, &eos));

    auto* handle = context.get_runtime_allocator()
                       ->allocateInstance<TokenizerDelegateHandle>();
    ET_CHECK_OR_RETURN_ERROR(
        handle != nullptr,
        MemoryAllocationFailed,
        "Failed to allocate TokenizerBackend handle");
    new (handle) TokenizerDelegateHandle();
    handle->max_context_length = max_context_length;
    handle->bos = bos;
    handle->eos = eos;
    handle->tokenizer = load_tokenizer_from_buffer(
        processed->data(), processed->size());
    if (handle->tokenizer == nullptr) {
      handle->~TokenizerDelegateHandle();
      ET_LOG(Error, "Failed to load bundled tokenizer");
      return Error::InvalidProgram;
    }
    return reinterpret_cast<executorch::runtime::DelegateHandle*>(handle);
  }

  Error execute(
      executorch::runtime::BackendExecutionContext&,
      executorch::runtime::DelegateHandle* handle,
      executorch::runtime::Span<executorch::runtime::EValue*> args)
      const override {
    ET_CHECK_OR_RETURN_ERROR(
        handle != nullptr,
        DelegateInvalidHandle,
        "TokenizerBackend handle is null");
    ET_CHECK_OR_RETURN_ERROR(
        args.size() == 2,
        InvalidProgram,
        "TokenizerBackend expects 2 arguments, got %zu",
        args.size());

    auto* tokenizer_handle =
        reinterpret_cast<TokenizerDelegateHandle*>(handle);
    auto* input = args[0];
    auto* output_value = args[1];
    ET_CHECK_OR_RETURN_ERROR(
        input != nullptr && input->isString(),
        InvalidArgument,
        "TokenizerBackend input must be a string");
    ET_CHECK_OR_RETURN_ERROR(
        output_value != nullptr && output_value->isTensor(),
        InvalidArgument,
        "TokenizerBackend output must be a tensor");

    const std::string_view prompt = input->toString();
    auto tokens_result = tokenizer_handle->tokenizer->encode(
        std::string(prompt), tokenizer_handle->bos, tokenizer_handle->eos);
    if (!tokens_result.ok()) {
      ET_LOG(Error, "Bundled tokenizer failed to encode input");
      return Error::InvalidArgument;
    }
    const auto& tokens = tokens_result.get();
    ET_CHECK_OR_RETURN_ERROR(
        tokens.size() <= tokenizer_handle->max_context_length,
        InvalidArgument,
        "Tokenizer output length %zu exceeds max context length %zu",
        tokens.size(),
        tokenizer_handle->max_context_length);

    auto& output = output_value->toTensor();
    ET_CHECK_OR_RETURN_ERROR(
        output.scalar_type() == executorch::aten::ScalarType::Long,
        InvalidArgument,
        "TokenizerBackend output tensor must be int64");
    ET_CHECK_OR_RETURN_ERROR(
        output.dim() == 1,
        InvalidArgument,
        "TokenizerBackend output tensor must be rank 1");
    executorch::aten::SizesType output_size =
        static_cast<executorch::aten::SizesType>(tokens.size());
    ET_CHECK_OK_OR_RETURN_ERROR(
        executorch::runtime::resize_tensor(
            output,
            executorch::aten::ArrayRef<executorch::aten::SizesType>(
                &output_size, 1)));
    auto* output_data = output.mutable_data_ptr<int64_t>();
    for (size_t i = 0; i < tokens.size(); ++i) {
      output_data[i] = static_cast<int64_t>(tokens[i]);
    }
    return Error::Ok;
  }

  void destroy(executorch::runtime::DelegateHandle* handle) const override {
    if (handle != nullptr) {
      reinterpret_cast<TokenizerDelegateHandle*>(handle)
          ->~TokenizerDelegateHandle();
    }
  }
};

TokenizerBackend tokenizer_backend;
executorch::runtime::Backend tokenizer_backend_registration{
    kTokenizerBackendId,
    &tokenizer_backend};
static auto tokenizer_backend_registration_status =
    executorch::runtime::register_backend(tokenizer_backend_registration);

} // namespace

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
  if (data_path.has_value()) {
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

  // Create IOManager
  std::unique_ptr<IOManager> io_manager = std::make_unique<IOManager>(*module);

  // Create text_decoder_runner
  ET_LOG(Info, "Using method: %s", method_name.c_str());
  auto text_decoder_runner = std::make_unique<TextDecoderRunner>(
      module.get(), io_manager.get(), method_name);

  // Create text_prefiller
  auto text_prefiller = std::make_unique<TextPrefiller>(
      text_decoder_runner.get(),
      metadata.at(kUseKVCache),
      metadata.at(kEnableDynamicShape),
      metadata.at(kMaxSeqLen));

  // Create text_token_generator with stats
  auto stats = std::make_unique<Stats>();
  auto text_token_generator = std::make_unique<TextTokenGenerator>(
      tokenizer.get(),
      text_decoder_runner.get(),
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
      temperature);
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
  if (data_path.has_value()) {
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

  // Create text_decoder_runner
  auto text_decoder_runner =
      std::make_unique<MultimodalDecoderRunner>(module.get(), io_manager.get());

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
