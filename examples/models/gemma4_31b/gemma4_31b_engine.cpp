/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/gemma4_31b/gemma4_31b_engine.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <vector>

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#include <executorch/backends/cuda/runtime/cuda_mutable_state.h>
#include <nlohmann/json.hpp>
#endif

namespace executorch::extension::llm {

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Result;
using SizesType = executorch::aten::SizesType;

namespace {

Result<uint64_t> read_sampled_token(
    const executorch::aten::Tensor& output,
    float temperature) {
#ifdef EXECUTORCH_BUILD_CUDA
  (void)temperature;
  const void* ptr = output.const_data_ptr();
  cudaPointerAttributes attrs{};
  const bool on_device = cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice;
  float val = 0.0f;
  if (on_device) {
    if (cudaMemcpy(&val, ptr, sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      ET_LOG(Error, "read_sampled_token: cudaMemcpy D2H failed");
      return Error::Internal;
    }
  } else {
    std::memcpy(&val, ptr, sizeof(float));
  }
  return static_cast<uint64_t>(llrintf(val));
#else
  (void)output;
  (void)temperature;
  return Error::NotSupported;
#endif
}

Result<std::unique_ptr<Module>> build_gemma_module(
    const Gemma4_31BConfig& config) {
  std::vector<std::string> data_files;
  if (!config.data_path.empty()) {
    data_files.push_back(config.data_path);
  }
  auto module = std::make_unique<Module>(
      config.model_path,
      data_files,
      Module::LoadMode::MmapUseMlockIgnoreErrors,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/true);

#ifdef EXECUTORCH_BUILD_CUDA
  if (config.enable_cuda_graph) {
    executorch::runtime::BackendOptions<2> cuda_opts;
    ET_CHECK_OK_OR_RETURN_ERROR(
        cuda_opts.set_option("enable_cuda_graph_for_method", "decode"));
    ET_CHECK_OK_OR_RETURN_ERROR(
        executorch::runtime::set_option("CudaBackend", cuda_opts.view()));
  }
  {
    executorch::runtime::BackendOptions<1> backend_options;
    ET_CHECK_OK_OR_RETURN_ERROR(
        backend_options.set_option("weight_sharing_across_methods", true));
    ET_CHECK_OK_OR_RETURN_ERROR(
        executorch::runtime::set_option("CudaBackend", backend_options.view()));
  }
  ET_CHECK_OK_OR_RETURN_ERROR(module->load_method("prefill"));
  ET_CHECK_OK_OR_RETURN_ERROR(module->load_method("decode"));
#else
  (void)module;
  ET_LOG(Error, "Gemma4_31BEngine is implemented for CUDA only");
  return Error::NotSupported;
#endif
  return module;
}

void add_token_piece(
    ::tokenizers::Tokenizer* tokenizer,
    std::unordered_set<uint64_t>& ids,
    const char* piece) {
  if (auto id = tokenizer->piece_to_id(piece); id.ok()) {
    ids.insert(*id);
  }
}

#ifdef EXECUTORCH_BUILD_CUDA
Error register_mutable_fqns(Module* module, int mutable_ctx) {
  auto res = module->execute("get_mutable_buffer_metadata");
  if (res.error() != Error::Ok) {
    ET_LOG(
        Info, "Gemma4_31BEngine: no mutable-buffer metadata; capacity stays 1");
    return res.error();
  }
  const auto& outs = res.get();
  if (outs.empty() || !outs[0].isString()) {
    ET_LOG(Error, "get_mutable_buffer_metadata did not return a string");
    return Error::InvalidProgram;
  }
  std::string json_str(outs[0].toString());
  auto j = nlohmann::json::parse(json_str, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_object() || j.value("version", 0) != 1 ||
      !j.contains("mutable_buffers") || !j["mutable_buffers"].is_array()) {
    ET_LOG(Error, "get_mutable_buffer_metadata has invalid schema");
    return Error::InvalidProgram;
  }
  std::vector<std::string> fqns;
  for (const auto& f : j["mutable_buffers"]) {
    if (!f.is_string() || f.get<std::string>().empty()) {
      ET_LOG(Error, "mutable_buffers entries must be non-empty strings");
      return Error::InvalidProgram;
    }
    fqns.push_back(f.get<std::string>());
  }
  if (fqns.empty()) {
    ET_LOG(Error, "mutable_buffers must be non-empty for multi-session");
    return Error::InvalidProgram;
  }
  ::executorch::backends::cuda::mutable_state_register_fqns(mutable_ctx, fqns);
  return Error::Ok;
}
#endif

class Gemma4_31BSession : public LLMSession {
 public:
  Gemma4_31BSession(
      Module* module,
      std::mutex* exec_mutex,
      int mutable_ctx,
      int session_token,
      std::atomic<int>* live_sessions,
      ::tokenizers::Tokenizer* tokenizer,
      std::unordered_map<std::string, int64_t> metadata,
      std::unordered_set<uint64_t> eos_ids,
      int64_t max_prefill_chunk,
      int64_t min_prefill_chunk)
      : module_(module),
        exec_mutex_(exec_mutex),
        mutable_ctx_(mutable_ctx),
        session_token_(session_token),
        live_sessions_(live_sessions),
        tokenizer_(tokenizer),
        metadata_(std::move(metadata)),
        eos_ids_(std::move(eos_ids)),
        max_prefill_chunk_(max_prefill_chunk),
        min_prefill_chunk_(min_prefill_chunk) {
    decode_tokens_ = from_blob(
        decode_token_data_, {1, 1}, executorch::aten::ScalarType::Long);
    decode_pos_ =
        from_blob(decode_pos_data_, {1}, executorch::aten::ScalarType::Long);
#ifdef EXECUTORCH_BUILD_CUDA
    temp_tensor_ =
        from_blob(&temp_val_, {1}, executorch::aten::ScalarType::Float);
#endif
  }

  ~Gemma4_31BSession() override {
#ifdef EXECUTORCH_BUILD_CUDA
    if (session_token_ != ::executorch::backends::cuda::kNoMutableSession) {
      ::executorch::backends::cuda::mutable_state_destroy_session(
          mutable_ctx_, session_token_);
    }
#endif
    if (live_sessions_ != nullptr) {
      live_sessions_->fetch_sub(1);
    }
  }

  Error prefill_tokens(
      std::vector<uint64_t> tokens,
      const SamplingConfig* initial_sampling) override {
    if (tokens.empty()) {
      return Error::InvalidArgument;
    }
    float first_token_temp = temperature_;
    if (initial_sampling != nullptr) {
      if (initial_sampling->top_p != 1.0f || initial_sampling->top_k != 0 ||
          initial_sampling->seed != 0) {
        ET_LOG(
            Error,
            "Gemma4_31BSession: only temperature is supported; top_p/top_k/seed "
            "are not implemented");
        return Error::NotSupported;
      }
      first_token_temp = initial_sampling->temperature;
    }
    const int64_t T = static_cast<int64_t>(tokens.size());
    const auto ctx_it = metadata_.find(kMaxContextLen);
    if (ctx_it != metadata_.end() && pos_ + T >= ctx_it->second) {
      ET_LOG(Error, "prefill_tokens would leave no room to generate");
      return Error::InvalidArgument;
    }

    stop_.store(false, std::memory_order_relaxed);
    int64_t offset = 0;
    while (offset < T) {
      int64_t chunk = T - offset;
      if (max_prefill_chunk_ > 0) {
        chunk = std::min(chunk, max_prefill_chunk_);
      }
#ifdef EXECUTORCH_BUILD_CUDA
      if (chunk > 1 && chunk < min_prefill_chunk_) {
        chunk = 1;
      }
#endif
      auto sampled =
          run_prefill_chunk(tokens.data() + offset, chunk, first_token_temp);
      ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
      pending_ = sampled.get();
      pos_ += chunk;
      offset += chunk;
    }
    prev_decode_token_ = tokens.back();
    return Error::Ok;
  }

  Result<DecodeResult> decode_one(const SamplingConfig& sampling) override {
    if (sampling.top_p != 1.0f || sampling.top_k != 0 || sampling.seed != 0) {
      ET_LOG(
          Error,
          "Gemma4_31BSession: only temperature is supported; top_p/top_k/seed "
          "are not implemented");
      return Error::NotSupported;
    }
    ET_CHECK_OR_RETURN_ERROR(
        pending_.has_value(),
        InvalidState,
        "decode_one requires a pending token; call prefill_tokens() first");
    temperature_ = sampling.temperature;

    const uint64_t token = pending_.value();
    const bool is_eos = eos_ids_.find(token) != eos_ids_.end();
    const uint64_t prev = prev_decode_token_.value_or(token);
    auto dec = tokenizer_->decode(prev, token);
    if (!dec.ok()) {
      ET_LOG(
          Error,
          "Tokenizers error code %d",
          static_cast<uint32_t>(dec.error()));
      return Error::InvalidArgument;
    }
    std::string text_piece = std::move(*dec);

    if (is_eos || stop_.load(std::memory_order_relaxed)) {
      pending_.reset();
      return DecodeResult{
          token, std::move(text_piece), is_eos, /*is_terminal=*/true};
    }

    const auto ctx_it = metadata_.find(kMaxContextLen);
    if (ctx_it != metadata_.end()) {
      ET_CHECK_OR_RETURN_ERROR(
          pos_ < ctx_it->second,
          InvalidArgument,
          "decode_one would exceed context capacity");
    }

    decode_token_data_[0] = static_cast<int64_t>(token);
    decode_pos_data_[0] = pos_;
    std::vector<EValue> inputs;
    inputs.push_back(EValue(decode_tokens_));
    inputs.push_back(EValue(decode_pos_));
#ifdef EXECUTORCH_BUILD_CUDA
    set_temp(temperature_);
    inputs.push_back(EValue(temp_tensor_));
    const char* method = "decode";
#else
    (void)inputs;
    return Error::NotSupported;
#endif
    auto sampled =
        run_locked(method, inputs, temperature_, /*sync_after=*/false);
    ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
    pending_ = sampled.get();
    prev_decode_token_ = token;
    pos_ += 1;
    return DecodeResult{
        token, std::move(text_piece), /*is_eos=*/false, /*is_terminal=*/false};
  }

  Error seek(int64_t pos) override {
    (void)pos;
    return Error::NotSupported;
  }

  int64_t position() const override {
    return pos_;
  }

  Error reset() override {
    pos_ = 0;
    pending_.reset();
    prev_decode_token_.reset();
    stop_.store(false, std::memory_order_relaxed);
    return Error::Ok;
  }

  void stop() override {
    stop_.store(true, std::memory_order_relaxed);
  }

 private:
#ifdef EXECUTORCH_BUILD_CUDA
  void set_temp(float t) {
    temp_val_ = (t <= 0.0f) ? 1e-6f : t;
  }
#endif

  Result<uint64_t>
  run_prefill_chunk(const uint64_t* tokens, int64_t T, float temperature) {
    std::vector<int64_t> token_data(tokens, tokens + T);
    std::vector<int64_t> pos_data(T);
    for (int64_t i = 0; i < T; ++i) {
      pos_data[i] = pos_ + i;
    }
    auto tokens_tensor = from_blob(
        token_data.data(),
        {1, static_cast<SizesType>(T)},
        executorch::aten::ScalarType::Long);
    auto pos_tensor = from_blob(
        pos_data.data(),
        {static_cast<SizesType>(T)},
        executorch::aten::ScalarType::Long);
    std::vector<EValue> inputs;
    inputs.push_back(EValue(tokens_tensor));
    inputs.push_back(EValue(pos_tensor));
#ifdef EXECUTORCH_BUILD_CUDA
    set_temp(temperature);
    inputs.push_back(EValue(temp_tensor_));
    const char* method = (T >= min_prefill_chunk_) ? "prefill" : "decode";
#else
    (void)inputs;
    (void)temperature;
    return Error::NotSupported;
#endif
    return run_locked(method, inputs, temperature, /*sync_after=*/true);
  }

  Result<uint64_t> run_locked(
      const char* method,
      std::vector<EValue>& inputs,
      float temperature,
      bool sync_after) {
    std::lock_guard<std::mutex> guard(*exec_mutex_);
#ifdef EXECUTORCH_BUILD_CUDA
    if (mutable_ctx_ != 0) {
      ::executorch::backends::cuda::mutable_state_set_active(
          mutable_ctx_, session_token_);
    }
#endif
    auto res = module_->execute(method, inputs);
#ifdef EXECUTORCH_BUILD_CUDA
    if (mutable_ctx_ != 0) {
      ::executorch::backends::cuda::mutable_state_set_active(
          mutable_ctx_, ::executorch::backends::cuda::kNoMutableSession);
    }
#endif
    ET_CHECK_OK_OR_RETURN_ERROR(res.error());
    auto sampled = read_sampled_token(res.get()[0].toTensor(), temperature);
    ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
#ifdef EXECUTORCH_BUILD_CUDA
    if (sync_after && cudaDeviceSynchronize() != cudaSuccess) {
      ET_LOG(Error, "run_locked: cudaDeviceSynchronize failed");
      return Error::Internal;
    }
#else
    (void)sync_after;
#endif
    return sampled.get();
  }

  Module* module_;
  std::mutex* exec_mutex_;
  int mutable_ctx_;
  int session_token_;
  std::atomic<int>* live_sessions_;
  ::tokenizers::Tokenizer* tokenizer_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;
  int64_t max_prefill_chunk_;
  int64_t min_prefill_chunk_;

  int64_t pos_ = 0;
  std::optional<uint64_t> pending_;
  std::optional<uint64_t> prev_decode_token_;
  float temperature_ = -1.0f;
  std::atomic<bool> stop_{false};

  int64_t decode_token_data_[1] = {0};
  int64_t decode_pos_data_[1] = {0};
  TensorPtr decode_tokens_;
  TensorPtr decode_pos_;
#ifdef EXECUTORCH_BUILD_CUDA
  float temp_val_ = 1e-6f;
  TensorPtr temp_tensor_;
#endif
};

} // namespace

Result<std::unique_ptr<Gemma4_31BEngine>> Gemma4_31BEngine::create(
    const Gemma4_31BConfig& config) {
  if (config.model_path.empty() || config.tokenizer_path.empty()) {
    ET_LOG(
        Error, "Gemma4_31BEngine: model_path and tokenizer_path are required");
    return Error::InvalidArgument;
  }

  auto tokenizer = std::make_unique<::tokenizers::HFTokenizer>();
  if (tokenizer->load(config.tokenizer_path) != ::tokenizers::Error::Ok) {
    ET_LOG(Error, "Gemma4_31BEngine: failed to load tokenizer");
    return Error::InvalidArgument;
  }

  std::vector<std::string> data_files;
  if (!config.data_path.empty()) {
    data_files.push_back(config.data_path);
  }
  auto meta_module = std::make_unique<Module>(
      config.model_path, data_files, Module::LoadMode::File);
  auto metadata_result = get_llm_metadata(tokenizer.get(), meta_module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Gemma4_31BEngine: failed to read metadata");
    return metadata_result.error();
  }

  auto eos_ids = get_eos_ids(tokenizer.get(), meta_module.get());
  eos_ids.insert(static_cast<uint64_t>(config.eos_id));
  add_token_piece(tokenizer.get(), eos_ids, "<turn|>");

  const auto& metadata = metadata_result.get();
  int64_t max_prefill_chunk = 1;
  auto max_ctx_it = metadata.find(kMaxContextLen);
  if (max_ctx_it != metadata.end() && max_ctx_it->second > 1) {
    max_prefill_chunk = max_ctx_it->second - 1;
  }
  if (auto get_result = meta_module->get("get_max_prefill_chunk");
      get_result.ok()) {
    max_prefill_chunk = get_result->toScalar().to<int64_t>();
  }
  int64_t min_prefill_chunk = 1;
#ifdef EXECUTORCH_BUILD_CUDA
  min_prefill_chunk = 5;
  if (auto get_result = meta_module->get("get_min_prefill_chunk");
      get_result.ok()) {
    min_prefill_chunk = get_result->toScalar().to<int64_t>();
  }
#endif

  bool registered_mutable = false;
  int mutable_ctx = 0;
#ifdef EXECUTORCH_BUILD_CUDA
  if (!config.enable_cuda_graph) {
    mutable_ctx = ::executorch::backends::cuda::mutable_state_create_context();
    if (register_mutable_fqns(meta_module.get(), mutable_ctx) == Error::Ok) {
      registered_mutable = true;
      ::executorch::backends::cuda::mutable_state_begin_load(mutable_ctx);
    } else {
      ::executorch::backends::cuda::mutable_state_destroy_context(mutable_ctx);
      mutable_ctx = 0;
    }
  }
#endif

  auto module_res = build_gemma_module(config);
#ifdef EXECUTORCH_BUILD_CUDA
  if (registered_mutable) {
    ::executorch::backends::cuda::mutable_state_end_load();
  }
#endif
  if (module_res.error() != Error::Ok) {
#ifdef EXECUTORCH_BUILD_CUDA
    if (mutable_ctx != 0) {
      ::executorch::backends::cuda::mutable_state_destroy_context(mutable_ctx);
    }
#endif
    return module_res.error();
  }

  bool rebind_available = false;
#ifdef EXECUTORCH_BUILD_CUDA
  if (mutable_ctx != 0) {
    rebind_available =
        ::executorch::backends::cuda::mutable_state_available(mutable_ctx);
    if (rebind_available &&
        ::executorch::backends::cuda::mutable_state_validate_coverage(
            mutable_ctx) != Error::Ok) {
      ET_LOG(
          Error,
          "Gemma4_31BEngine: mutable-buffer coverage check failed; disabling "
          "multi-session");
      rebind_available = false;
    }
  }
#endif

  return std::unique_ptr<Gemma4_31BEngine>(new Gemma4_31BEngine(
      config,
      std::move(tokenizer),
      metadata,
      std::move(eos_ids),
      std::move(module_res.get()),
      max_prefill_chunk,
      min_prefill_chunk,
      rebind_available,
      mutable_ctx));
}

Gemma4_31BEngine::~Gemma4_31BEngine() {
#ifdef EXECUTORCH_BUILD_CUDA
  if (mutable_ctx_ != 0) {
    ::executorch::backends::cuda::mutable_state_destroy_context(mutable_ctx_);
  }
#endif
}

Result<std::unique_ptr<LLMSession>> Gemma4_31BEngine::create_session() {
  const int cap =
      serving_capacity().max_physical_sessions_without_weight_duplication;
  {
    std::lock_guard<std::mutex> g(exec_mutex_);
    if (live_sessions_.load() >= cap) {
      return Error::InvalidState;
    }
    live_sessions_.fetch_add(1);
  }

  int token = -1;
#ifdef EXECUTORCH_BUILD_CUDA
  if (rebind_available_) {
    auto t = ::executorch::backends::cuda::mutable_state_create_session(
        mutable_ctx_);
    if (t.error() != Error::Ok) {
      live_sessions_.fetch_sub(1);
      return t.error();
    }
    token = t.get();
  }
#endif
  return std::unique_ptr<LLMSession>(new Gemma4_31BSession(
      shared_module_.get(),
      &exec_mutex_,
      mutable_ctx_,
      token,
      &live_sessions_,
      tokenizer_.get(),
      metadata_,
      eos_ids_,
      max_prefill_chunk_,
      min_prefill_chunk_));
}

LLMServingCapacity Gemma4_31BEngine::serving_capacity() const {
  LLMServingCapacity cap;
#ifdef EXECUTORCH_BUILD_CUDA
  if (rebind_available_) {
    cap.max_physical_sessions_without_weight_duplication =
        config_.max_sessions > 1 ? config_.max_sessions : 1;
    cap.estimated_bytes_per_session =
        ::executorch::backends::cuda::mutable_state_bytes_per_session(
            mutable_ctx_);
  }
#endif
  return cap;
}

} // namespace executorch::extension::llm
