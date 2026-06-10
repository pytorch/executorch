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
#include <cstring>

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

// ---------------------------------------------------------------------------
// Backend-specific helpers (the MLX extension points live here). On CUDA the
// model fuses the sampler in and returns the sampled token id as a [B,1] float;
// non-CUDA returns logits and we sample on host. Keep these isolated so the
// session logic below stays backend-agnostic.
// ---------------------------------------------------------------------------

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
      // Don't fabricate token id 0 (a valid token) on a copy failure — that is
      // silent corruption. Surface it so the caller aborts the request.
      ET_LOG(Error, "read_sampled_token: cudaMemcpy D2H failed");
      return Error::Internal;
    }
  } else {
    std::memcpy(&val, ptr, sizeof(float));
  }
  return static_cast<uint64_t>(val);
#else
  return static_cast<uint64_t>(
      logits_to_token(output, temperature < 0.0f ? 0.0f : temperature));
#endif
}

// Build the one shared Qwen Module: shared mutable arenas (so prefill and
// decode share KV/conv/recurrent state) and, on CUDA, the weight-sharing
// backend option that MUST be set before load_method. Loads the prefill+decode
// methods once (the heavy ~weights load). Called once when the engine is
// created.
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
      /*share_memory_arenas=*/true);

#ifdef EXECUTORCH_BUILD_CUDA
  // Backend options are read during backend init(), so they must be set before
  // load_method. (CUDA graph is intentionally not enabled: each session
  // rebinds its mutable buffers before execute, which a captured graph's baked
  // pointers would ignore.)
  {
    // Cross-method per-FQN weight sharing: prefill and decode reuse one weight
    // allocation instead of duplicating it (critical to fit on one GPU).
    executorch::runtime::BackendOptions<1> backend_options;
    ET_CHECK_OK_OR_RETURN_ERROR(
        backend_options.set_option("weight_sharing_across_methods", true));
    ET_CHECK_OK_OR_RETURN_ERROR(
        executorch::runtime::set_option("CudaBackend", backend_options.view()));
  }
#endif

  ET_CHECK_OK_OR_RETURN_ERROR(module->load_method("prefill"));
  ET_CHECK_OK_OR_RETURN_ERROR(module->load_method("decode"));
  return module;
}

#ifdef EXECUTORCH_BUILD_CUDA
// Read the model's per-session mutable-buffer FQNs from its export metadata
// ({"version":1,"mutable_buffers":[...]}) and register them with the CUDA
// backend so it can give each session its own GPU buffers for that state.
Error register_mutable_fqns(Module* module, int mutable_ctx) {
  auto res = module->execute("get_mutable_buffer_metadata");
  if (res.error() != Error::Ok) {
    ET_LOG(
        Error,
        "Qwen35MoEEngine: model has no get_mutable_buffer_metadata; re-export "
        "for multi-session");
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
  ::executorch::backends::cuda::mutable_state_register_fqns(mutable_ctx, fqns);
  return Error::Ok;
}
#endif

// LLMSession over the Qwen3.5 MoE prefill/decode methods. Owns one physical
// Module (one weight allocation + its KV/recurrent/conv state). Internal: the
// server depends only on the LLMSession base.
class Qwen35MoESession : public LLMSession {
 public:
  Qwen35MoESession(
      Module* module,
      std::mutex* exec_mutex,
      int mutable_ctx,
      int session_token,
      std::atomic<int>* live_sessions,
      ::tokenizers::Tokenizer* tokenizer,
      std::unordered_map<std::string, int64_t> metadata,
      std::unordered_set<uint64_t> eos_ids)
      : module_(module),
        exec_mutex_(exec_mutex),
        mutable_ctx_(mutable_ctx),
        session_token_(session_token),
        live_sessions_(live_sessions),
        tokenizer_(tokenizer),
        metadata_(std::move(metadata)),
        eos_ids_(std::move(eos_ids)) {
    // Persistent single-step decode buffers, reused (updated in place) across
    // decode steps to avoid per-step reallocation.
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
#ifdef EXECUTORCH_BUILD_CUDA
    if (session_token_ != ::executorch::backends::cuda::kNoMutableSession) {
      ::executorch::backends::cuda::mutable_state_destroy_session(
          mutable_ctx_, session_token_);
    }
#endif
    // Release the engine's capacity slot reserved in create_session().
    if (live_sessions_ != nullptr) {
      live_sessions_->fetch_sub(1);
    }
  }

  Error prefill_tokens(
      std::vector<uint64_t> tokens,
      const SamplingConfig* initial_sampling) override {
    if (tokens.empty()) {
      ET_LOG(Error, "prefill_tokens: empty token list");
      return Error::InvalidArgument;
    }
    // The model samples the FIRST generated token in-graph during prefill, so
    // it must use the request's sampling, not a stale session default. Only
    // temperature is plumbed; reject non-default top_p/top_k/seed (parity with
    // decode_one).
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
    const int64_t T = static_cast<int64_t>(tokens.size());
    const auto ctx_it = metadata_.find(kMaxContextLen);
    // Require room for at least one generated token: after prefill, pos_ == T
    // and decode_one() forwards the first token at pos_, which must be < the
    // context length. Rejecting pos_ + T == max_context (not just > it) keeps a
    // full prompt from reaching decode_one with no room to step.
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

    // A new prefill starts a fresh generation turn; clear any prior stop.
    stop_.store(false, std::memory_order_relaxed);
    std::vector<int64_t> token_data(tokens.begin(), tokens.end());
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

    // prefill method handles T>=2; the model exports decode for the T==1 case.
    const char* method = (T >= 2) ? "prefill" : "decode";
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
    pending_ = sampled.get();
    prev_decode_token_.reset();
    pos_ += T; // the prompt tokens are now resident in KV/state
    return Error::Ok;
  }

  Result<DecodeResult> decode_one(const SamplingConfig& sampling) override {
    // Only temperature is plumbed; reject the rest rather than silently ignore
    // (callers must not assume top_p/top_k/seed are applied).
    if (sampling.top_p != 1.0f || sampling.top_k != 0 || sampling.seed != 0) {
      ET_LOG(
          Error,
          "Qwen35MoESession: only temperature is supported; top_p/top_k/seed "
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

    // Decode the text piece with BPE context (previous token); surface
    // tokenizer errors instead of hiding them as empty text.
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

    // Terminate WITHOUT forwarding the token: at EOS (like the reference
    // runner, EOS is not made resident and position() does not advance) or at a
    // cooperative stop() observed at this boundary. No pending token remains.
    // is_eos stays literal; is_terminal ends the loop either way.
    if (is_eos || stop_.load(std::memory_order_relaxed)) {
      pending_.reset();
      return DecodeResult{
          token, std::move(text_piece), is_eos, /*is_terminal=*/true};
    }

    // Only a NON-EOS, non-stopped token is forwarded (made resident at pos_),
    // so the capacity check belongs here — after the short-circuit, so a final
    // EOS is still emitted when state is exactly full. Without it, decode would
    // write KV/recurrent state past the context window.
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

    // Forward `token` at pos_ through the decode method to get the next pending
    // token. Update the persistent buffers in place (stable addresses).
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
        run_locked("decode", inputs, temperature_, /*sync_after=*/false);
    ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
    pending_ = sampled.get();
    prev_decode_token_ = token;
    pos_ += 1;
    return DecodeResult{
        token, std::move(text_piece), /*is_eos=*/false, /*is_terminal=*/false};
  }

  Error seek(int64_t pos) override {
    // The hybrid model carries recurrent/conv state that cannot be safely
    // rewound by logical position the way contiguous KV can. Fail closed so the
    // prefix cache falls back to reset + full prefill.
    (void)pos;
    return Error::NotSupported;
  }

  int64_t position() const override {
    return pos_;
  }

  Error reset() override {
    // Logical reset is sufficient: the model zeroes conv_state/recurrent_state
    // whenever prefill runs at input_pos[0]==0 (model.py), and a fresh prefill
    // overwrites the KV cache at [0, T). So rewinding to position 0 and
    // clearing the pending token gives a clean conversation without a Module
    // rebuild.
    pos_ = 0;
    pending_.reset();
    prev_decode_token_.reset();
    stop_.store(false, std::memory_order_relaxed);
    return Error::Ok;
  }

  void stop() override {
    // Cooperative, token-boundary: the driving loop checks between decode_one()
    // calls. A single decode_one() forward is not interruptible.
    stop_.store(true, std::memory_order_relaxed);
  }

 private:
#ifdef EXECUTORCH_BUILD_CUDA
  // Greedy (temperature <= 0) maps to a tiny temperature so the in-graph
  // sampler avoids division by zero while staying effectively argmax.
  void set_temp(float t) {
    temp_val_ = (t <= 0.0f) ? 1e-6f : t;
  }
#endif

  // Run a method with THIS session's mutable state bound, then read the sampled
  // token — all inside one engine-lock critical section so another session
  // cannot rebind between this session's rebind, execute, and read-out.
  Result<uint64_t> run_locked(
      const char* method,
      std::vector<EValue>& inputs,
      float temperature,
      bool sync_after) {
    std::lock_guard<std::mutex> guard(*exec_mutex_);
#ifdef EXECUTORCH_BUILD_CUDA
    ::executorch::backends::cuda::mutable_state_set_active(
        mutable_ctx_, session_token_);
#endif
    auto res = module_->execute(method, inputs);
#ifdef EXECUTORCH_BUILD_CUDA
    ::executorch::backends::cuda::mutable_state_set_active(
        mutable_ctx_, ::executorch::backends::cuda::kNoMutableSession);
#endif
    ET_CHECK_OK_OR_RETURN_ERROR(res.error());
    auto sampled = read_sampled_token(res.get()[0].toTensor(), temperature);
    ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
#ifdef EXECUTORCH_BUILD_CUDA
    // Prefill runs on a different stream than decode; sync so its writes to the
    // session's mutable buffers are visible to the session's first decode (also
    // surfaces any async launch error). Decode reads its own writes in stream
    // order, so it does not need this.
    if (sync_after && cudaDeviceSynchronize() != cudaSuccess) {
      ET_LOG(Error, "run_locked: cudaDeviceSynchronize failed");
      return Error::Internal;
    }
#else
    (void)sync_after;
#endif
    return sampled.get();
  }

  Module* module_; // non-owning; the engine's one shared physical model
  std::mutex*
      exec_mutex_; // non-owning; serializes rebind+execute across sessions
  int mutable_ctx_; // engine's CUDA mutable-state context (per-engine)
  int session_token_; // CUDA per-session mutable-state token (or
                      // kNoMutableSession)
  std::atomic<int>* live_sessions_; // non-owning; engine capacity counter
  ::tokenizers::Tokenizer* tokenizer_; // non-owning; owned by the engine
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;

  int64_t pos_ = 0;
  std::optional<uint64_t> pending_;
  std::optional<uint64_t> prev_decode_token_;
  float temperature_ = -1.0f;
  std::atomic<bool> stop_{false};

  // Persistent single-step decode buffers (reused across decode steps).
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
  auto eos_ids = get_eos_ids(tokenizer.get(), meta_module.get());
  // This export's metadata doesn't carry the chat-turn EOS (config.json has no
  // eos_token_id and the .pte exports no get_eos_ids method), so get_eos_ids()
  // misses it and a session would never terminate — it would decode to
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

  int mutable_ctx = 0; // kInvalidMutableContext
#ifdef EXECUTORCH_BUILD_CUDA
  // Create this engine's own mutable-state context (per-engine, not global) and
  // register the per-session mutable-buffer FQNs from the .pte metadata BEFORE
  // loading the heavy methods, so the CUDA backend associates the load's
  // handles with this context and builds descriptors from the still-initial
  // constants.
  mutable_ctx = ::executorch::backends::cuda::mutable_state_create_context();
  if (Error e = register_mutable_fqns(meta_module.get(), mutable_ctx);
      e != Error::Ok) {
    ::executorch::backends::cuda::mutable_state_destroy_context(mutable_ctx);
    return e;
  }
  ::executorch::backends::cuda::mutable_state_begin_load(mutable_ctx);
#endif

  // Build the ONE shared physical model (the heavy ~weights load). All sessions
  // reuse it; each rebinds its own mutable buffers before execute.
  auto module_res = build_qwen_module(config);
#ifdef EXECUTORCH_BUILD_CUDA
  ::executorch::backends::cuda::mutable_state_end_load();
#endif
  if (module_res.error() != Error::Ok) {
#ifdef EXECUTORCH_BUILD_CUDA
    ::executorch::backends::cuda::mutable_state_destroy_context(mutable_ctx);
#endif
    return module_res.error();
  }
  std::unique_ptr<Module> shared_module = std::move(module_res.get());

  bool rebind_available = false;
#ifdef EXECUTORCH_BUILD_CUDA
  rebind_available =
      ::executorch::backends::cuda::mutable_state_available(mutable_ctx);
  if (rebind_available) {
    // Fail closed: if any declared mutable FQN was not found in the loaded
    // methods' constants, multi-session would run without rebinding it and
    // bleed state — fall back to single-session instead.
    if (::executorch::backends::cuda::mutable_state_validate_coverage(
            mutable_ctx) != Error::Ok) {
      ET_LOG(
          Error,
          "Qwen35MoEEngine: mutable-buffer coverage check failed; disabling "
          "multi-session (capacity clamped to 1).");
      rebind_available = false;
    }
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
      rebind_available,
      mutable_ctx));
}

Qwen35MoEEngine::~Qwen35MoEEngine() {
#ifdef EXECUTORCH_BUILD_CUDA
  if (mutable_ctx_ != 0) {
    ::executorch::backends::cuda::mutable_state_destroy_context(mutable_ctx_);
  }
#endif
}

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
  return std::unique_ptr<LLMSession>(new Qwen35MoESession(
      shared_module_.get(),
      &exec_mutex_,
      mutable_ctx_,
      token,
      &live_sessions_,
      tokenizer_.get(),
      metadata_,
      eos_ids_));
}

LLMServingCapacity Qwen35MoEEngine::serving_capacity() const {
  LLMServingCapacity cap; // default: 1 session, 0 bytes (unknown)
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
