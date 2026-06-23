/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/qwen3_5_moe/qwen35_moe_engine.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <cinttypes>
#include <cmath>
#include <cstring>

#include <algorithm>

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#include <executorch/backends/cuda/runtime/cuda_mutable_state.h>
#include <nlohmann/json.hpp>
#else
#include <executorch/extension/llm/sampler/util.h>
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

#ifdef EXECUTORCH_BUILD_MLX
// The MLX export emits a single dynamic-seq `forward` method that handles both
// prefill (T>=2) and decode (T=1). Mirror gemma4_31b's MLX runner, which loads
// and calls `forward` for both phases.
constexpr const char* kPrefillMethod = "forward";
constexpr const char* kDecodeMethod = "forward";
#else
// CUDA/Metal exports emit two separate methods.
constexpr const char* kPrefillMethod = "prefill";
constexpr const char* kDecodeMethod = "decode";
#endif

// Constant method exported by the MLX .pte giving the largest prefill chunk the
// `forward` method was compiled for. Read into the metadata map in create().
constexpr const char* kMaxPrefillChunk = "get_max_prefill_chunk";

Result<uint64_t> read_sampled_token(
    const executorch::aten::Tensor& output,
    float temperature) {
#ifdef EXECUTORCH_BUILD_CUDA
  (void)temperature;
  const void* ptr = output.const_data_ptr();
  cudaPointerAttributes attrs;
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
  return static_cast<uint64_t>(
      logits_to_token(output, temperature < 0.0f ? 0.0f : temperature));
#endif
}

Result<std::unique_ptr<Module>> build_qwen_module(
    const Qwen35MoEConfig& config) {
  std::vector<std::string> data_files;
  if (!config.data_path.empty()) {
    data_files.push_back(config.data_path);
  }
  auto module = std::make_unique<Module>(
      config.model_path,
      data_files,
      Module::LoadMode::File,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/false);

#ifdef EXECUTORCH_BUILD_CUDA
  if (config.enable_cuda_graph) {
    executorch::runtime::BackendOptions<2> cuda_opts;
    ET_CHECK_OK_OR_RETURN_ERROR(
        cuda_opts.set_option("enable_cuda_graph_for_method", "decode"));
    ET_CHECK_OK_OR_RETURN_ERROR(
        executorch::runtime::set_option("CudaBackend", cuda_opts.view()));
    ET_LOG(Info, "Qwen35MoEEngine: CUDA graph enabled for decode method");
  }
  {
    executorch::runtime::BackendOptions<1> backend_options;
    ET_CHECK_OK_OR_RETURN_ERROR(
        backend_options.set_option("weight_sharing_across_methods", true));
    ET_CHECK_OK_OR_RETURN_ERROR(
        executorch::runtime::set_option("CudaBackend", backend_options.view()));
  }
#endif

  ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(kPrefillMethod));
  if (std::string(kDecodeMethod) != std::string(kPrefillMethod)) {
    ET_CHECK_OK_OR_RETURN_ERROR(module->load_method(kDecodeMethod));
  }
  return module;
}

#ifdef EXECUTORCH_BUILD_CUDA
Error register_mutable_fqns(
    Module* module,
    ::executorch::backends::cuda::MutableStateContextOwner& mutable_state) {
  auto res = module->execute("get_mutable_buffer_metadata");
  if (res.error() != Error::Ok) {
    ET_LOG(
        Info,
        "Qwen35MoEEngine: model has no get_mutable_buffer_metadata; "
        "multi-session disabled");
    return res.error();
  }
  const auto& outs = res.get();
  if (outs.empty() || !outs[0].isString()) {
    ET_LOG(Error, "get_mutable_buffer_metadata did not return a string");
    return Error::InvalidProgram;
  }
  std::string json_str(outs[0].toString());
  auto j = nlohmann::json::parse(json_str, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_object()) {
    ET_LOG(Error, "get_mutable_buffer_metadata is not a valid JSON object");
    return Error::InvalidProgram;
  }
  if (!j.contains("version") || !j["version"].is_number_integer() ||
      j["version"].get<int>() != 1) {
    ET_LOG(Error, "get_mutable_buffer_metadata: unsupported/missing version");
    return Error::InvalidProgram;
  }
  if (!j.contains("mutable_buffers") || !j["mutable_buffers"].is_array() ||
      j["mutable_buffers"].empty()) {
    ET_LOG(
        Error,
        "get_mutable_buffer_metadata: mutable_buffers must be a non-empty array");
    return Error::InvalidProgram;
  }
  std::vector<std::string> fqns;
  for (const auto& f : j["mutable_buffers"]) {
    if (!f.is_string() || f.get<std::string>().empty()) {
      ET_LOG(
          Error,
          "get_mutable_buffer_metadata: every mutable_buffers entry must be a "
          "non-empty string");
      return Error::InvalidProgram;
    }
    fqns.push_back(f.get<std::string>());
  }
  mutable_state.register_fqns(fqns);
  return Error::Ok;
}
#endif

class Qwen35MoESession : public LLMSession {
 public:
  Qwen35MoESession(
      Module* module,
      std::mutex* exec_mutex,
      std::atomic<int>* live_sessions,
      ::tokenizers::Tokenizer* tokenizer,
      std::unordered_map<std::string, int64_t> metadata,
      std::unordered_set<uint64_t> eos_ids
#ifdef QWEN_HAS_MUTABLE_STATE
      ,
      MutableStateContextOwner* mutable_state,
      int session_token
#endif
      )
      : module_(module),
        exec_mutex_(exec_mutex),
        live_sessions_(live_sessions),
        tokenizer_(tokenizer),
        metadata_(std::move(metadata)),
        eos_ids_(std::move(eos_ids))
#ifdef QWEN_HAS_MUTABLE_STATE
        ,
        mutable_state_(mutable_state),
        session_token_(session_token)
#endif
  {
    decode_tokens_ = from_blob(
        decode_token_data_, {1, 1}, executorch::aten::ScalarType::Long);
    decode_pos_ =
        from_blob(decode_pos_data_, {1}, executorch::aten::ScalarType::Long);
#ifdef EXECUTORCH_BUILD_CUDA
    temp_tensor_ =
        from_blob(&temp_val_, {1}, executorch::aten::ScalarType::Float);
#endif
  }

  ~Qwen35MoESession() override {
#ifdef QWEN_HAS_MUTABLE_STATE
    if (mutable_state_ != nullptr && session_token_ != kNoMutableSession) {
      mutable_state_->destroy_session(session_token_);
    }
#endif
    if (live_sessions_ != nullptr) {
      live_sessions_->fetch_sub(1);
    }
  }

  Error prefill_tokens(
      const std::vector<uint64_t>& tokens,
      const SamplingConfig* initial_sampling) override {
    if (tokens.empty()) {
      ET_LOG(Error, "prefill_tokens: empty token list");
      return Error::InvalidArgument;
    }
    float first_token_temp = temperature_;
    if (initial_sampling != nullptr) {
      if (initial_sampling->top_p != 1.0f || initial_sampling->top_k != 0 ||
          initial_sampling->seed != 0) {
        ET_LOG(
            Error,
            "prefill_tokens: only temperature is supported; top_p/top_k/seed "
            "are not yet implemented");
        return Error::NotSupported;
      }
      first_token_temp = initial_sampling->temperature;
    }
    if (!valid_temperature(first_token_temp)) {
      ET_LOG(Error, "prefill_tokens: temperature must be -1 or in [0, 1]");
      return Error::InvalidArgument;
    }
    const int64_t T = static_cast<int64_t>(tokens.size());
    const auto ctx_it = metadata_.find(kMaxContextLen);
    if (ctx_it != metadata_.end() && pos_ + T >= ctx_it->second) {
      ET_LOG(
          Error,
          "prefill_tokens would leave no room to generate (pos %" PRId64
          " + %" PRId64 " >= max_context %" PRId64 ")",
          pos_,
          T,
          ctx_it->second);
      return Error::InvalidArgument;
    }

    stop_.store(false, std::memory_order_relaxed);

    // On MLX, run prefill in fixed-size chunks (caps peak memory and the
    // compiled prefill shape). Other backends prefill the whole prompt in one
    // pass. Only the final chunk's sampled token is kept; the recurrence/KV
    // state from earlier chunks persists via pos_ advancement.
#ifdef EXECUTORCH_BUILD_MLX
    // Chunk size: default to the compiled max (kMaxSeqLen - 1), overridden by
    // the exported get_max_prefill_chunk constant when present (mirrors
    // gemma4_31b). Falls back to T (single pass) if no metadata is available at
    // all.
    int64_t chunk_size = T;
    if (auto it = metadata_.find(kMaxSeqLen);
        it != metadata_.end() && it->second > 1) {
      chunk_size = it->second - 1;
    }
    if (auto it = metadata_.find(kMaxPrefillChunk);
        it != metadata_.end() && it->second > 0) {
      chunk_size = it->second;
    }
#else
    const int64_t chunk_size = T;
#endif

    uint64_t sampled_token = 0;
    for (int64_t off = 0; off < T; off += chunk_size) {
      const int64_t len = std::min(chunk_size, T - off);
      std::vector<int64_t> token_data(
          tokens.begin() + off, tokens.begin() + off + len);
      std::vector<int64_t> pos_data(len);
      for (int64_t i = 0; i < len; ++i) {
        pos_data[i] = pos_ + i;
      }
      auto tokens_tensor = from_blob(
          token_data.data(),
          {1, static_cast<SizesType>(len)},
          executorch::aten::ScalarType::Long);
      auto pos_tensor = from_blob(
          pos_data.data(),
          {static_cast<SizesType>(len)},
          executorch::aten::ScalarType::Long);

      const char* method = (len >= 2) ? kPrefillMethod : kDecodeMethod;
      std::vector<EValue> inputs;
      inputs.push_back(tokens_tensor);
      inputs.push_back(pos_tensor);
#ifdef EXECUTORCH_BUILD_CUDA
      set_temp(first_token_temp);
      inputs.push_back(EValue(temp_tensor_));
#endif
      auto sampled =
          run_locked(method, inputs, first_token_temp, /*sync_after=*/true);
      ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
      sampled_token = sampled.get();
      pos_ += len;
    }
    pending_ = sampled_token;
    prev_decode_token_.reset();
    return Error::Ok;
  }

  Result<DecodeResult> decode_one(const SamplingConfig& sampling) override {
    if (sampling.top_p != 1.0f || sampling.top_k != 0 || sampling.seed != 0) {
      ET_LOG(
          Error,
          "Qwen35MoESession: only temperature is supported; top_p/top_k/seed "
          "are not implemented");
      return Error::NotSupported;
    }
    if (!valid_temperature(sampling.temperature)) {
      ET_LOG(Error, "decode_one: temperature must be -1 or in [0, 1]");
      return Error::InvalidArgument;
    }
    ET_CHECK_OR_RETURN_ERROR(
        pending_.has_value(),
        InvalidState,
        "decode_one requires a pending token; call prefill_tokens() first");
    temperature_ = sampling.temperature;

    if (stop_.load(std::memory_order_relaxed)) {
      return DecodeResult{0, "", /*is_eos=*/false, /*is_terminal=*/true};
    }

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

    if (is_eos) {
      pending_.reset();
      return DecodeResult{
          token, std::move(text_piece), is_eos, /*is_terminal=*/true};
    }

    const auto ctx_it = metadata_.find(kMaxContextLen);
    if (ctx_it != metadata_.end()) {
      ET_CHECK_OR_RETURN_ERROR(
          pos_ < ctx_it->second,
          InvalidArgument,
          "decode_one would exceed context capacity: pos_ %" PRId64
          " >= max_context %" PRId64,
          pos_,
          ctx_it->second);
    }

    decode_token_data_[0] = static_cast<int64_t>(token);
    decode_pos_data_[0] = pos_;
    std::vector<EValue> inputs;
    inputs.push_back(EValue(decode_tokens_));
    inputs.push_back(EValue(decode_pos_));
#ifdef EXECUTORCH_BUILD_CUDA
    set_temp(temperature_);
    inputs.push_back(EValue(temp_tensor_));
#endif
    auto sampled =
        run_locked(kDecodeMethod, inputs, temperature_, /*sync_after=*/false);
    ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
    pending_ = sampled.get();
    prev_decode_token_ = token;
    pos_ += 1;
    return DecodeResult{
        token, std::move(text_piece), /*is_eos=*/false, /*is_terminal=*/false};
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
  static bool valid_temperature(float temperature) {
    return temperature == -1.0f || (temperature >= 0.0f && temperature <= 1.0f);
  }

#ifdef EXECUTORCH_BUILD_CUDA
  void set_temp(float t) {
    temp_val_ = (t <= 0.0f) ? 1e-6f : t;
  }
#endif

  Result<uint64_t> run_locked(
      const char* method,
      std::vector<EValue>& inputs,
      float temperature,
      bool sync_after) {
    std::lock_guard<std::mutex> guard(*exec_mutex_);
#ifdef QWEN_HAS_MUTABLE_STATE
    auto res = mutable_state_ != nullptr
        ? mutable_state_->with_active_session(
              session_token_,
              [&]() { return module_->execute(method, inputs); })
        : module_->execute(method, inputs);
#else
    auto res = module_->execute(method, inputs);
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
  std::atomic<int>* live_sessions_;
  ::tokenizers::Tokenizer* tokenizer_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;

  int64_t pos_ = 0;
  std::optional<uint64_t> pending_;
  std::optional<uint64_t> prev_decode_token_;
  float temperature_ = -1.0f;
  std::atomic<bool> stop_{false};

  int64_t decode_token_data_[1] = {0};
  int64_t decode_pos_data_[1] = {0};
  TensorPtr decode_tokens_;
  TensorPtr decode_pos_;
#ifdef QWEN_HAS_MUTABLE_STATE
  MutableStateContextOwner* mutable_state_ = nullptr;
  int session_token_ = kNoMutableSession;
#endif
#ifdef EXECUTORCH_BUILD_CUDA
  float temp_val_ = 1e-6f;
  TensorPtr temp_tensor_;
#endif
};

} // namespace

Result<std::unique_ptr<Qwen35MoEEngine>> Qwen35MoEEngine::create(
    const Qwen35MoEConfig& config) {
  if (config.model_path.empty() || config.tokenizer_path.empty()) {
    ET_LOG(
        Error, "Qwen35MoEEngine: model_path and tokenizer_path are required");
    return Error::InvalidArgument;
  }

  auto tokenizer = std::make_unique<::tokenizers::HFTokenizer>();
  if (tokenizer->load(config.tokenizer_path) != ::tokenizers::Error::Ok) {
    ET_LOG(
        Error,
        "Qwen35MoEEngine: failed to load tokenizer from %s",
        config.tokenizer_path.c_str());
    return Error::InvalidArgument;
  }

  // Read metadata + eos from a lightweight Module (program + tiny metadata
  // methods only; the heavy prefill/decode weights are NOT loaded here).
  std::vector<std::string> data_files;
  if (!config.data_path.empty()) {
    data_files.push_back(config.data_path);
  }
  auto meta_module = std::make_unique<Module>(
      config.model_path, data_files, Module::LoadMode::File);
  auto metadata_result = get_llm_metadata(tokenizer.get(), meta_module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Qwen35MoEEngine: failed to read metadata");
    return metadata_result.error();
  }
#ifdef EXECUTORCH_BUILD_MLX
  // Surface the compiled max prefill chunk (a constant method get_llm_metadata
  // doesn't harvest) into the metadata map so the session can chunk long
  // prompts within the shape `forward` was compiled for.
  if (auto mpc = meta_module->get(kMaxPrefillChunk); mpc.ok()) {
    metadata_result.get()[kMaxPrefillChunk] = mpc->toScalar().to<int64_t>();
  }
#endif
  auto eos_ids = get_eos_ids(tokenizer.get(), meta_module.get());
  // This export's metadata doesn't carry the chat-turn EOS (config.json has no
  // eos_token_id and the .pte exports no get_eos_ids method), so get_eos_ids()
  // misses it and a session would never terminate; it would decode to
  // max_new_tokens every turn. <|im_end|> ends every Qwen assistant turn; add
  // it explicitly so decode_one() stops at end of turn.
  if (auto im_end = tokenizer->piece_to_id("<|im_end|>"); im_end.ok()) {
    eos_ids.insert(*im_end);
  } else {
    ET_LOG(
        Error,
        "Qwen35MoEEngine: could not resolve <|im_end|> token id; the model may "
        "not stop at end of turn");
  }

#ifdef QWEN_HAS_MUTABLE_STATE
  std::unique_ptr<MutableStateContextOwner> mutable_state;
#endif
#ifdef EXECUTORCH_BUILD_CUDA
  if (config.enable_cuda_graph) {
    ET_LOG(
        Info,
        "Qwen35MoEEngine: CUDA graph requested; per-session rebinding disabled "
        "and serving capacity clamped to 1 session.");
  } else {
    auto candidate = std::make_unique<MutableStateContextOwner>();
    if (Error e = register_mutable_fqns(meta_module.get(), *candidate);
        e == Error::Ok) {
      mutable_state = std::move(candidate);
    } else {
      ET_LOG(
          Info,
          "Qwen35MoEEngine: mutable-buffer metadata unavailable or invalid; "
          "serving capacity clamped to 1 session.");
    }
  }
#elif defined(EXECUTORCH_BUILD_MLX)
  // MLX owns mutable buffers directly and selects per-session storage at
  // execute time; no FQN registration or coverage check is required.
  mutable_state = std::make_unique<MutableStateContextOwner>();
#endif

#ifdef QWEN_HAS_MUTABLE_STATE
  auto module_res = mutable_state != nullptr
      ? mutable_state->with_load_scope(
            [&]() { return build_qwen_module(config); })
      : build_qwen_module(config);
#else
  auto module_res = build_qwen_module(config);
#endif
  if (module_res.error() != Error::Ok) {
    return module_res.error();
  }
  std::unique_ptr<Module> shared_module = std::move(module_res.get());

  bool rebind_available = false;
#ifdef QWEN_HAS_MUTABLE_STATE
  rebind_available = mutable_state != nullptr && mutable_state->available();
  if (rebind_available && mutable_state->validate_coverage() != Error::Ok) {
    ET_LOG(
        Error,
        "Qwen35MoEEngine: mutable-buffer coverage check failed; disabling "
        "multi-session (capacity clamped to 1).");
    rebind_available = false;
  }
  if (!rebind_available) {
    ET_LOG(
        Info,
        "Qwen35MoEEngine: per-session rebinding unavailable; serving capacity "
        "clamped to 1 session.");
  }
#endif

  return std::unique_ptr<Qwen35MoEEngine>(new Qwen35MoEEngine(
      config,
      std::move(tokenizer),
      metadata_result.get(),
      std::move(eos_ids),
      std::move(shared_module),
      rebind_available
#ifdef QWEN_HAS_MUTABLE_STATE
      ,
      std::move(mutable_state)
#endif
          ));
}

Qwen35MoEEngine::~Qwen35MoEEngine() = default;

Result<std::unique_ptr<LLMSession>> Qwen35MoEEngine::create_session() {
  // Enforce serving_capacity(): without rebinding, capacity is 1, so a second
  // session would silently share the resident KV/conv/recurrent state. Reserve
  // a slot under the exec lock (released in ~Qwen35MoESession).
  const int cap =
      serving_capacity().max_physical_sessions_without_weight_duplication;
  {
    std::lock_guard<std::mutex> g(exec_mutex_);
    if (live_sessions_.load() >= cap) {
      ET_LOG(
          Error,
          "Qwen35MoEEngine: at session capacity (%d); refusing create_session "
          "(would share state or duplicate weights)",
          cap);
      return Error::InvalidState;
    }
    live_sessions_.fetch_add(1);
  }

  int token = -1; // kNoMutableSession: single-session / no rebind
#ifdef QWEN_HAS_MUTABLE_STATE
  if (rebind_available_) {
    auto t = mutable_state_->create_session();
    if (t.error() != Error::Ok) {
      live_sessions_.fetch_sub(1);
      return t.error();
    }
    token = t.get();
  }
#endif
  return std::unique_ptr<LLMSession>(new Qwen35MoESession(
      shared_module_.get(),
      &exec_mutex_,
      &live_sessions_,
      tokenizer_.get(),
      metadata_,
      eos_ids_
#ifdef QWEN_HAS_MUTABLE_STATE
      ,
      mutable_state_.get(),
      token
#endif
      ));
}

LLMServingCapacity Qwen35MoEEngine::serving_capacity() const {
  LLMServingCapacity cap; // default: 1 session, 0 bytes (unknown)
#ifdef QWEN_HAS_MUTABLE_STATE
  if (rebind_available_) {
    cap.max_physical_sessions_without_weight_duplication =
        config_.max_sessions > 1 ? config_.max_sessions : 1;
    cap.estimated_bytes_per_session = mutable_state_->bytes_per_session();
  }
#endif
  return cap;
}

} // namespace executorch::extension::llm
