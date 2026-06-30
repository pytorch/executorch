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
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/portable_type/device.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <optional>
#include <vector>

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#else
#include <executorch/extension/llm/sampler/util.h>
#endif

namespace executorch::extension::llm {

using ::executorch::extension::clone_tensor_ptr_to;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Result;
using SizesType = executorch::aten::SizesType;

namespace {

#ifdef EXECUTORCH_BUILD_MLX
constexpr const char* kPrefillMethod = "forward";
constexpr const char* kDecodeMethod = "forward";
#else
constexpr const char* kPrefillMethod = "prefill";
constexpr const char* kDecodeMethod = "decode";
#endif

constexpr const char* kMaxPrefillChunk = "get_max_prefill_chunk";
constexpr const char* kMinPrefillChunk = "get_min_prefill_chunk";

Result<uint64_t> read_sampled_token(
    const executorch::aten::Tensor& output,
    float temperature) {
#ifdef EXECUTORCH_BUILD_CUDA
  (void)temperature;
  const void* ptr = output.const_data_ptr();
  cudaPointerAttributes attrs{};
  const bool on_device = cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice;

  auto copy_scalar = [&](void* dst, size_t nbytes) -> Error {
    if (on_device) {
      if (cudaMemcpy(dst, ptr, nbytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return Error::Internal;
      }
    } else {
      std::memcpy(dst, ptr, nbytes);
    }
    return Error::Ok;
  };

  if (output.scalar_type() == executorch::aten::ScalarType::Long) {
    int64_t val = 0;
    if (copy_scalar(&val, sizeof(val)) != Error::Ok) {
      ET_LOG(Error, "read_sampled_token: cudaMemcpy D2H failed");
      return Error::Internal;
    }
    return static_cast<uint64_t>(val);
  }
  if (output.scalar_type() == executorch::aten::ScalarType::Float) {
    float val = 0.0f;
    if (copy_scalar(&val, sizeof(val)) != Error::Ok) {
      ET_LOG(Error, "read_sampled_token: cudaMemcpy D2H failed");
      return Error::Internal;
    }
    return static_cast<uint64_t>(llrintf(val));
  }
  ET_LOG(
      Error,
      "read_sampled_token: expected Long or Float scalar output, got %d",
      static_cast<int>(output.scalar_type()));
  return Error::InvalidArgument;
#else
  return static_cast<uint64_t>(
      logits_to_token(output, temperature < 0.0f ? 0.0f : temperature));
#endif
}

Result<std::unique_ptr<Module>> build_gemma_module(
    const Gemma4_31BConfig& config,
    bool multi_session) {
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
      /*temp_allocator=*/nullptr);

#ifdef EXECUTORCH_BUILD_CUDA
  if (config.enable_cuda_graph) {
    executorch::runtime::BackendOptions<2> cuda_opts;
    ET_CHECK_OK_OR_RETURN_ERROR(
        cuda_opts.set_option("enable_cuda_graph_for_method", "decode"));
    ET_CHECK_OK_OR_RETURN_ERROR(
        executorch::runtime::set_option("CudaBackend", cuda_opts.view()));
    ET_LOG(Info, "Gemma4_31BEngine: CUDA graph enabled for decode method");
  }
  {
    executorch::runtime::BackendOptions<1> backend_options;
    ET_CHECK_OK_OR_RETURN_ERROR(
        backend_options.set_option("weight_sharing_across_methods", true));
    ET_CHECK_OK_OR_RETURN_ERROR(
        executorch::runtime::set_option("CudaBackend", backend_options.view()));
  }
#endif

  const executorch::runtime::LoadBackendOptionsMap* load_options = nullptr;
#ifdef EXECUTORCH_BUILD_MLX
  // Per-model MLX runtime specs, delivered to MLXBackend::init(). Must outlive
  // the load_method calls below (they read it during init).
  executorch::runtime::BackendOptions<2> mlx_opts;
  executorch::runtime::LoadBackendOptionsMap mlx_options_map;
  // Release MLX's cached buffer pool every N forward calls to bound memory
  // growth during long sessions.
  constexpr int kMLXClearCacheInterval = 1;
  ET_CHECK_OK_OR_RETURN_ERROR(mlx_opts.set_option(
      ::executorch::backends::mlx::kClearCacheIntervalKey,
      kMLXClearCacheInterval));
  ET_LOG(
      Info,
      "Gemma4_31BEngine: MLX clear_cache_interval=%d",
      kMLXClearCacheInterval);
  // skip_mutable_buffer_init must match the multi-session owner: it is only
  // safe when a load scope is active (the caller passes multi_session = "owner
  // exists"), since per-session buffers are then allocated by
  // mlx_mutable_state. The backend also defensively rejects the flag without an
  // active owner.
  if (multi_session) {
    ET_CHECK_OK_OR_RETURN_ERROR(mlx_opts.set_option(
        ::executorch::backends::mlx::kSkipMutableBufferInitKey, true));
    ET_LOG(Info, "Gemma4_31BEngine: MLX skip_mutable_buffer_init=true");
  }
  ET_CHECK_OK_OR_RETURN_ERROR(mlx_options_map.set_options(
      ::executorch::backends::mlx::kMLXBackendId, mlx_opts.view()));
  load_options = &mlx_options_map;
#endif

  ET_CHECK_OK_OR_RETURN_ERROR(
      module->load_method(kPrefillMethod, nullptr, nullptr, load_options));
  if (std::string(kDecodeMethod) != std::string(kPrefillMethod)) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        module->load_method(kDecodeMethod, nullptr, nullptr, load_options));
  }
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
Error register_mutable_fqns(
    Module* module,
    ::executorch::backends::cuda::MutableStateContextOwner& mutable_state) {
  auto res = module->execute("get_mutable_buffer_metadata");
  if (res.error() != Error::Ok) {
    ET_LOG(
        Info,
        "Gemma4_31BEngine: model has no get_mutable_buffer_metadata; "
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

TensorPtr build_decode_pos_table(
    const std::unordered_map<std::string, int64_t>& metadata) {
  auto ctx_it = metadata.find(kMaxContextLen);
  if (ctx_it == metadata.end() || ctx_it->second <= 0) {
    return nullptr;
  }
  std::vector<int64_t> pos_data(ctx_it->second);
  for (int64_t i = 0; i < ctx_it->second; ++i) {
    pos_data[i] = i;
  }
  return clone_tensor_ptr_to(
      from_blob(
          pos_data.data(),
          {static_cast<SizesType>(pos_data.size())},
          executorch::aten::ScalarType::Long),
      executorch::aten::Device(executorch::aten::DeviceType::CUDA, 0));
}
#endif

class Gemma4_31BSession : public LLMSession {
 public:
  Gemma4_31BSession(
      Module* module,
      std::mutex* exec_mutex,
      std::atomic<int>* live_sessions,
      ::tokenizers::Tokenizer* tokenizer,
      std::unordered_map<std::string, int64_t> metadata,
      std::unordered_set<uint64_t> eos_ids,
      int64_t max_prefill_chunk,
      int64_t min_prefill_chunk,
      TensorPtr decode_pos_table_dev,
      GemmaMutableStateContextOwner* mutable_state,
      int session_token)
      : module_(module),
        exec_mutex_(exec_mutex),
        live_sessions_(live_sessions),
        tokenizer_(tokenizer),
        metadata_(std::move(metadata)),
        eos_ids_(std::move(eos_ids)),
        max_prefill_chunk_(max_prefill_chunk),
        min_prefill_chunk_(min_prefill_chunk),
#ifdef EXECUTORCH_BUILD_CUDA
        decode_pos_table_dev_(std::move(decode_pos_table_dev)),
#endif
        mutable_state_(mutable_state),
        session_token_(session_token) {
    decode_tokens_ = from_blob(
        decode_token_data_, {1, 1}, executorch::aten::ScalarType::Long);
    decode_pos_ =
        from_blob(decode_pos_data_, {1}, executorch::aten::ScalarType::Long);
#ifdef EXECUTORCH_BUILD_CUDA
    decode_tokens_dev_ = clone_tensor_ptr_to(decode_tokens_, cuda_device_);
    decode_pos_dev_ = clone_tensor_ptr_to(decode_pos_, cuda_device_);
    auto temp_host =
        from_blob(&temp_val_, {1}, executorch::aten::ScalarType::Float);
    temp_tensor_dev_ = clone_tensor_ptr_to(temp_host, cuda_device_);
#endif
  }

  ~Gemma4_31BSession() override {
    if (mutable_state_ != nullptr && session_token_ != kGemmaNoMutableSession) {
      mutable_state_->destroy_session(session_token_);
    }
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
            "Gemma4_31BSession: only temperature is supported; top_p/top_k/seed "
            "are not implemented");
        return Error::NotSupported;
      }
      first_token_temp = initial_sampling->temperature;
    }
    if (!valid_temperature(first_token_temp)) {
      ET_LOG(Error, "prefill_tokens: temperature must be -1 or in [0, 2]");
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
    if (!valid_temperature(sampling.temperature)) {
      ET_LOG(Error, "decode_one: temperature must be -1 or in [0, 2]");
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
#ifdef EXECUTORCH_BUILD_CUDA
    ET_CHECK_OK_OR_RETURN_ERROR(update_decode_pos_on_cuda());
    ET_CHECK_OK_OR_RETURN_ERROR(set_temperature(temperature_));
    inputs.push_back(EValue(decode_tokens_dev_));
    inputs.push_back(EValue(decode_pos_dev_));
    inputs.push_back(EValue(temp_tensor_dev_));
#else
    inputs.push_back(EValue(decode_tokens_));
    inputs.push_back(EValue(decode_pos_));
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
    return temperature == -1.0f || (temperature >= 0.0f && temperature <= 2.0f);
  }

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
    TensorPtr token_input = tokens_tensor;
    TensorPtr pos_input = pos_tensor;
#ifdef EXECUTORCH_BUILD_CUDA
    std::vector<TensorPtr> device_inputs;
    token_input = to_cuda(token_input, device_inputs);
    pos_input = to_cuda(pos_input, device_inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(set_temperature(temperature));
#endif
    inputs.push_back(EValue(token_input));
    inputs.push_back(EValue(pos_input));
#ifdef EXECUTORCH_BUILD_CUDA
    inputs.push_back(EValue(temp_tensor_dev_));
    const char* method =
        (T >= min_prefill_chunk_) ? kPrefillMethod : kDecodeMethod;
#else
    const char* method = kPrefillMethod;
#endif
    return run_locked(method, inputs, temperature, /*sync_after=*/true);
  }

#ifdef EXECUTORCH_BUILD_CUDA
  TensorPtr to_cuda(TensorPtr tensor, std::vector<TensorPtr>& keep_alive) {
    keep_alive.push_back(clone_tensor_ptr_to(tensor, cuda_device_));
    return keep_alive.back();
  }

  Error set_temperature(float temperature) {
    if (!valid_temperature(temperature)) {
      return Error::InvalidArgument;
    }
    temp_val_ = (temperature <= 0.0f) ? 1e-6f : temperature;
    if (cudaMemcpy(
            temp_tensor_dev_->mutable_data_ptr(),
            &temp_val_,
            sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess) {
      ET_LOG(Error, "set_temperature: cudaMemcpy H2D failed");
      return Error::Internal;
    }
    return Error::Ok;
  }

  Error copy_decode_token_to_cuda(uint64_t token) {
    const int64_t token_value = static_cast<int64_t>(token);
    if (cudaMemcpy(
            decode_tokens_dev_->mutable_data_ptr(),
            &token_value,
            sizeof(int64_t),
            cudaMemcpyHostToDevice) != cudaSuccess) {
      ET_LOG(Error, "copy_decode_token_to_cuda: token H2D failed");
      return Error::Internal;
    }
    return Error::Ok;
  }

  Error stage_next_decode_token_on_cuda(
      const executorch::aten::Tensor& out_tensor,
      uint64_t token) {
    if (out_tensor.scalar_type() == executorch::aten::ScalarType::Long) {
      const void* ptr = out_tensor.const_data_ptr();
      cudaPointerAttributes attrs{};
      const bool on_device =
          cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess &&
          attrs.type == cudaMemoryTypeDevice;
      if (cudaMemcpy(
              decode_tokens_dev_->mutable_data_ptr(),
              ptr,
              sizeof(int64_t),
              on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice) !=
          cudaSuccess) {
        ET_LOG(Error, "stage_next_decode_token_on_cuda: token copy failed");
        return Error::Internal;
      }
      return Error::Ok;
    }
    return copy_decode_token_to_cuda(token);
  }

  Error update_decode_pos_on_cuda() {
    if (decode_pos_table_dev_ != nullptr) {
      auto* pos_table =
          static_cast<int64_t*>(decode_pos_table_dev_->mutable_data_ptr());
      auto* pos_slot =
          static_cast<int64_t*>(decode_pos_dev_->mutable_data_ptr());
      if (cudaMemcpy(
              pos_slot,
              pos_table + pos_,
              sizeof(int64_t),
              cudaMemcpyDeviceToDevice) != cudaSuccess) {
        ET_LOG(Error, "update_decode_pos_on_cuda: position D2D failed");
        return Error::Internal;
      }
      return Error::Ok;
    }
    if (cudaMemcpy(
            decode_pos_dev_->mutable_data_ptr(),
            decode_pos_data_,
            sizeof(int64_t),
            cudaMemcpyHostToDevice) != cudaSuccess) {
      ET_LOG(Error, "update_decode_pos_on_cuda: position H2D failed");
      return Error::Internal;
    }
    return Error::Ok;
  }
#endif

  Result<uint64_t> run_locked(
      const char* method,
      std::vector<EValue>& inputs,
      float temperature,
      bool sync_after) {
    std::lock_guard<std::mutex> guard(*exec_mutex_);
    auto res = mutable_state_ != nullptr
        ? mutable_state_->with_active_session(
              session_token_,
              [&]() { return module_->execute(method, inputs); })
        : module_->execute(method, inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(res.error());
    const auto& out_tensor = res.get()[0].toTensor();
    auto sampled = read_sampled_token(out_tensor, temperature);
    ET_CHECK_OK_OR_RETURN_ERROR(sampled.error());
#ifdef EXECUTORCH_BUILD_CUDA
    ET_CHECK_OK_OR_RETURN_ERROR(
        stage_next_decode_token_on_cuda(out_tensor, sampled.get()));
#endif
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
  TensorPtr decode_pos_table_dev_;
#endif
  GemmaMutableStateContextOwner* mutable_state_ = nullptr;
  int session_token_ = kGemmaNoMutableSession;
#ifdef EXECUTORCH_BUILD_CUDA
  float temp_val_ = 1e-6f;
  executorch::aten::Device cuda_device_ =
      executorch::aten::Device(executorch::aten::DeviceType::CUDA, 0);
  TensorPtr decode_tokens_dev_;
  TensorPtr decode_pos_dev_;
  TensorPtr temp_tensor_dev_;
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
  add_token_piece(tokenizer.get(), eos_ids, "<end_of_turn>");

  auto metadata = metadata_result.get();
  int64_t max_prefill_chunk = 1;
  auto max_ctx_it = metadata.find(kMaxContextLen);
  if (max_ctx_it != metadata.end() && max_ctx_it->second > 1) {
    max_prefill_chunk = max_ctx_it->second - 1;
  }
  if (auto get_result = meta_module->get(kMaxPrefillChunk); get_result.ok()) {
    max_prefill_chunk = get_result->toScalar().to<int64_t>();
    metadata[kMaxPrefillChunk] = max_prefill_chunk;
  }

  int64_t min_prefill_chunk = 1;
#ifdef EXECUTORCH_BUILD_CUDA
  min_prefill_chunk = 5;
  if (auto get_result = meta_module->get(kMinPrefillChunk); get_result.ok()) {
    min_prefill_chunk = get_result->toScalar().to<int64_t>();
  }
  metadata[kMinPrefillChunk] = min_prefill_chunk;
#endif

  std::unique_ptr<GemmaMutableStateContextOwner> mutable_state;
#ifdef EXECUTORCH_BUILD_CUDA
  if (config.enable_cuda_graph) {
    ET_LOG(
        Info,
        "Gemma4_31BEngine: CUDA graph requested; per-session rebinding "
        "disabled and serving capacity clamped to 1 session.");
  } else {
    auto candidate = std::make_unique<GemmaMutableStateContextOwner>();
    if (Error e = register_mutable_fqns(meta_module.get(), *candidate);
        e == Error::Ok) {
      mutable_state = std::move(candidate);
    } else {
      ET_LOG(
          Info,
          "Gemma4_31BEngine: mutable-buffer metadata unavailable or invalid; "
          "serving capacity clamped to 1 session.");
    }
  }
#elif defined(EXECUTORCH_BUILD_MLX)
  // Only enable the per-session mutable-buffer path when actually serving more
  // than one session. For a single session (the CLI runner) the rebind would
  // allocate a second copy of the KV-cache buffers on top of the program's
  // default buffers — doubling KV-cache memory and adding a one-time
  // session-buffer allocation during the first prefill — for no isolation
  // benefit. Leaving mutable_state null keeps the program's default buffers.
  if (config.max_sessions > 1) {
    mutable_state = std::make_unique<GemmaMutableStateContextOwner>();
  }
#endif

  // Pass whether a multi-session owner exists as the single source of truth for
  // skip_mutable_buffer_init, so the skip flag can never diverge from the
  // owner.
  const bool multi_session = mutable_state != nullptr;
  auto module_res = multi_session ? mutable_state->with_load_scope([&]() {
    return build_gemma_module(config, multi_session);
  })
                                  : build_gemma_module(config, multi_session);
  if (module_res.error() != Error::Ok) {
    return module_res.error();
  }
  std::unique_ptr<Module> shared_module = std::move(module_res.get());

  bool rebind_available = false;
  rebind_available = mutable_state != nullptr && mutable_state->available();
  if (rebind_available && mutable_state->validate_coverage() != Error::Ok) {
    ET_LOG(
        Error,
        "Gemma4_31BEngine: mutable-buffer coverage check failed; disabling "
        "multi-session (capacity clamped to 1).");
    rebind_available = false;
  }
  if (!rebind_available) {
    ET_LOG(
        Info,
        "Gemma4_31BEngine: per-session rebinding unavailable; serving capacity "
        "clamped to 1 session.");
  }

  TensorPtr decode_pos_table_dev;
#ifdef EXECUTORCH_BUILD_CUDA
  decode_pos_table_dev = build_decode_pos_table(metadata);
#endif

  return std::unique_ptr<Gemma4_31BEngine>(new Gemma4_31BEngine(
      config,
      std::move(tokenizer),
      std::move(metadata),
      std::move(eos_ids),
      std::move(shared_module),
      max_prefill_chunk,
      min_prefill_chunk,
      std::move(decode_pos_table_dev),
      rebind_available,
      std::move(mutable_state)));
}

Gemma4_31BEngine::~Gemma4_31BEngine() = default;

Result<std::unique_ptr<LLMSession>> Gemma4_31BEngine::create_session() {
  const int cap =
      serving_capacity().max_physical_sessions_without_weight_duplication;
  {
    std::lock_guard<std::mutex> g(exec_mutex_);
    if (live_sessions_.load() >= cap) {
      ET_LOG(
          Error,
          "Gemma4_31BEngine: at session capacity (%d); refusing create_session",
          cap);
      return Error::InvalidState;
    }
    live_sessions_.fetch_add(1);
  }

  int token = -1;
  if (rebind_available_) {
    auto t = mutable_state_->create_session();
    if (t.error() != Error::Ok) {
      live_sessions_.fetch_sub(1);
      return t.error();
    }
    token = t.get();
  }

  return std::unique_ptr<LLMSession>(new Gemma4_31BSession(
      shared_module_.get(),
      &exec_mutex_,
      &live_sessions_,
      tokenizer_.get(),
      metadata_,
      eos_ids_,
      max_prefill_chunk_,
      min_prefill_chunk_,
      decode_pos_table_dev_,
      mutable_state_.get(),
      token));
}

LLMServingCapacity Gemma4_31BEngine::serving_capacity() const {
  LLMServingCapacity cap;
  if (rebind_available_) {
    cap.max_physical_sessions_without_weight_duplication =
        config_.max_sessions > 1 ? config_.max_sessions : 1;
    cap.estimated_bytes_per_session = mutable_state_->bytes_per_session();
  }
  return cap;
}

} // namespace executorch::extension::llm
