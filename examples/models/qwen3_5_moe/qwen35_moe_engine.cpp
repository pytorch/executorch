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

uint64_t read_sampled_token(
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
      return 0;
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

// Build a Qwen Module with shared mutable arenas (so prefill and decode share
// KV/conv/recurrent state) and, on CUDA, the weight-sharing/cuda-graph backend
// options that MUST be set before load_method. Loads the prefill+decode methods
// (this is the heavy ~weights load). Shared by create_session() and reset().
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
  // load_method.
  if (config.cuda_graph) {
    executorch::runtime::BackendOptions<1> cuda_opts;
    cuda_opts.set_option("enable_cuda_graph_for_method", "decode");
    ET_CHECK_OK_OR_RETURN_ERROR(
        executorch::runtime::set_option("CudaBackend", cuda_opts.view()));
  }
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

// LLMSession over the Qwen3.5 MoE prefill/decode methods. Owns one physical
// Module (one weight allocation + its KV/recurrent/conv state). Internal: the
// server depends only on the LLMSession base.
class Qwen35MoESession : public LLMSession {
 public:
  Qwen35MoESession(
      std::unique_ptr<Module> module,
      ::tokenizers::Tokenizer* tokenizer,
      std::unordered_map<std::string, int64_t> metadata,
      std::unordered_set<uint64_t> eos_ids)
      : module_(std::move(module)),
        tokenizer_(tokenizer),
        metadata_(std::move(metadata)),
        eos_ids_(std::move(eos_ids)) {
    // Persistent single-step decode buffers: stable addresses are required so
    // CUDA-graph capture (which records buffer pointers) can replay each step.
    decode_tokens_ = from_blob(
        decode_token_data_, {1, 1}, executorch::aten::ScalarType::Long);
    decode_pos_ =
        from_blob(decode_pos_data_, {1}, executorch::aten::ScalarType::Long);
#ifdef EXECUTORCH_BUILD_CUDA
    temp_tensor_ =
        from_blob(&temp_val_, {1}, executorch::aten::ScalarType::Float);
#endif
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
    if (ctx_it != metadata_.end() && pos_ + T > ctx_it->second) {
      ET_LOG(
          Error,
          "prefill_tokens would exceed context capacity (pos %" PRId64
          " + %" PRId64 " > %" PRId64 ")",
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
    auto res = module_->execute(method, inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(res.error());
    pending_ = read_sampled_token(res.get()[0].toTensor(), first_token_temp);
    prev_decode_token_.reset();
    pos_ += T; // the prompt tokens are now resident in KV/state
#ifdef EXECUTORCH_BUILD_CUDA
    // Make prefill's writes to the shared mutable arenas visible to decode
    // (which may run on a different stream).
    cudaDeviceSynchronize();
#endif
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
    auto res = module_->execute("decode", inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(res.error());
    pending_ = read_sampled_token(res.get()[0].toTensor(), temperature_);
    prev_decode_token_ = token;
    pos_ += 1;
    return DecodeResult{
        token, std::move(text_piece), /*is_eos=*/false, /*is_terminal=*/false};
  }

  Error seek(int64_t pos) override {
    // The hybrid model carries recurrent/conv state that cannot be safely
    // rewound by logical position the way contiguous KV can. Fail closed so the
    // prefix cache falls back to reset + full prefill (V1).
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

  std::unique_ptr<Module> module_;
  ::tokenizers::Tokenizer* tokenizer_; // non-owning; owned by the engine
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;

  int64_t pos_ = 0;
  std::optional<uint64_t> pending_;
  std::optional<uint64_t> prev_decode_token_;
  float temperature_ = -1.0f;
  std::atomic<bool> stop_{false};

  // Persistent single-step decode buffers (stable addresses for CUDA graph).
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

  return std::unique_ptr<Qwen35MoEEngine>(new Qwen35MoEEngine(
      config, std::move(tokenizer), metadata_result.get(), std::move(eos_ids)));
}

Result<std::unique_ptr<LLMSession>> Qwen35MoEEngine::create_session() {
  auto module = build_qwen_module(config_);
  ET_CHECK_OK_OR_RETURN_ERROR(module.error());
  return std::unique_ptr<LLMSession>(new Qwen35MoESession(
      std::move(module.get()), tokenizer_.get(), metadata_, eos_ids_));
}

} // namespace executorch::extension::llm
