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
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/tekken.h>
#include <pytorch/tokenizers/tiktoken.h>

namespace executorch::extension::llm {

using ::executorch::extension::Module;
using ::executorch::extension::Program;
using ::executorch::runtime::Error;

// Assembles the per-Module components (decoder/prefiller/token generator/io
// manager/stats) into a TextLLMRunner. Shared by the path-based and the
// shared-Program (TextLLMEngine session) construction paths.
static std::unique_ptr<TextLLMRunner> assemble_text_llm_runner(
    std::unique_ptr<Module> module,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    float temperature,
    const std::string& method_name);

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

  return assemble_text_llm_runner(
      std::move(module), std::move(tokenizer), temperature, method_name);
}

static std::unique_ptr<TextLLMRunner> assemble_text_llm_runner(
    std::unique_ptr<Module> module,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    float temperature,
    const std::string& method_name) {
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

  // Read vocab_size for Sampler
  int32_t vocab_size = static_cast<int32_t>(metadata.at(kVocabSize));
  float init_temp = temperature == -1.0f ? 0.0f : temperature;
  auto sampler = std::make_unique<Sampler>(vocab_size, init_temp);

  // Create text_decoder_runner
  ET_LOG(Info, "Using method: %s", method_name.c_str());
  auto text_decoder_runner = std::make_unique<TextDecoderRunner>(
      module.get(), io_manager.get(), method_name, std::move(sampler));

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

std::unique_ptr<TextLLMRunner> create_text_llm_runner_from_program(
    std::shared_ptr<Program> program,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    float temperature,
    const std::string& method_name) {
  if (!tokenizer || !tokenizer->is_loaded()) {
    ET_LOG(Error, "Tokenizer is null or not loaded");
    return nullptr;
  }
  if (!program) {
    ET_LOG(Error, "Program is null");
    return nullptr;
  }
  // A Module over the already-loaded Program: it reuses that Program rather
  // than re-loading it, and its loaded method allocates its own planned (KV)
  // memory. Whether packed weights are physically shared vs. re-materialized
  // per method instance is backend-dependent (serving_capacity() is the
  // authority); on XNNPACK assume per-instance.
  constexpr uint32_t kMaxCachedMemoryBytes = 1024 * 1024 * 10; // 10MB
  auto module = std::make_unique<Module>(
      std::move(program),
      nullptr, // memory allocator
      std::make_unique<executorch::extension::CPUCachingAllocator>(
          kMaxCachedMemoryBytes));
  return assemble_text_llm_runner(
      std::move(module), std::move(tokenizer), temperature, method_name);
}

namespace {
// The TextLLM adapter: implements the model-agnostic LLMSession over a
// TextLLMRunner. TextLLMRunner is an implementation detail here — the engine
// and server depend only on LLMSession.
class TextLLMSession : public LLMSession {
 public:
  explicit TextLLMSession(std::unique_ptr<TextLLMRunner> runner)
      : runner_(std::move(runner)) {}

  Error prefill_tokens(std::vector<uint64_t> tokens) override {
    return runner_->prefill_tokens(std::move(tokens)).error();
  }
  ::executorch::runtime::Result<DecodeResult> decode_one(
      const SamplingConfig& sampling) override {
    // Only temperature is plumbed today; top_p/top_k/seed need a per-session
    // sampler (applied in a follow-up).
    return runner_->decode_one(sampling.temperature);
  }
  Error seek(int64_t pos) override {
    return runner_->seek(pos);
  }
  int64_t position() const override {
    return runner_->position();
  }
  Error reset() override {
    runner_->reset();
    return Error::Ok;
  }
  void stop() override {
    runner_->stop();
  }

 private:
  std::unique_ptr<TextLLMRunner> runner_;
};
} // namespace

TextLLMEngine::TextLLMEngine(
    std::unique_ptr<Module> loader_module,
    std::shared_ptr<Program> program,
    std::string tokenizer_path,
    float temperature,
    std::string method_name,
    std::unordered_map<std::string, int64_t> metadata)
    : loader_module_(std::move(loader_module)),
      program_(std::move(program)),
      tokenizer_path_(std::move(tokenizer_path)),
      temperature_(temperature),
      method_name_(std::move(method_name)),
      metadata_(std::move(metadata)) {}

std::unique_ptr<TextLLMEngine> TextLLMEngine::create(
    const std::string& model_path,
    const std::string& tokenizer_path,
    std::optional<const std::string> data_path,
    float temperature,
    const std::string& method_name,
    Module::LoadMode load_mode) {
  // External .ptd weights are not yet supported for shared sessions: each
  // session Module built from the shared Program would also need the
  // data_map_loader threaded into its load_method() to resolve external
  // weights (see Module::load_method merged_data_map_). Fail loudly rather than
  // silently produce sessions that error on first generate.
  if (data_path.has_value()) {
    ET_LOG(
        Error,
        "TextLLMEngine: external data_path (.ptd) is not yet supported for "
        "shared sessions; use a self-contained .pte for now.");
    return nullptr;
  }
  // Load the program ONCE; sessions reuse it (loaded a single time, per-session
  // KV). Physical weight sharing across sessions is backend-dependent — see
  // serving_capacity().
  auto loader_module = std::make_unique<Module>(model_path, load_mode);
  if (loader_module->load() != Error::Ok) {
    ET_LOG(
        Error,
        "TextLLMEngine: failed to load program from %s",
        model_path.c_str());
    return nullptr;
  }
  auto program = loader_module->program();
  if (!program) {
    ET_LOG(Error, "TextLLMEngine: program is null after load");
    return nullptr;
  }
  // Read model-level metadata once (shared by all sessions).
  auto meta_tokenizer = load_tokenizer(tokenizer_path);
  if (!meta_tokenizer) {
    ET_LOG(
        Error,
        "TextLLMEngine: failed to load tokenizer from %s",
        tokenizer_path.c_str());
    return nullptr;
  }
  auto metadata_result =
      get_llm_metadata(meta_tokenizer.get(), loader_module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "TextLLMEngine: failed to read metadata");
    return nullptr;
  }
  return std::unique_ptr<TextLLMEngine>(new TextLLMEngine(
      std::move(loader_module),
      std::move(program),
      tokenizer_path,
      temperature,
      method_name,
      metadata_result.get()));
}

::executorch::runtime::Result<std::unique_ptr<LLMSession>>
TextLLMEngine::create_session() {
  auto tokenizer = load_tokenizer(tokenizer_path_);
  if (!tokenizer) {
    ET_LOG(
        Error,
        "TextLLMEngine: failed to load tokenizer from %s",
        tokenizer_path_.c_str());
    return Error::InvalidState;
  }
  auto runner = create_text_llm_runner_from_program(
      program_, std::move(tokenizer), temperature_, method_name_);
  if (!runner) {
    ET_LOG(Error, "TextLLMEngine: failed to build session runner");
    return Error::InvalidState;
  }
  return std::unique_ptr<LLMSession>(
      std::make_unique<TextLLMSession>(std::move(runner)));
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
