/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/cuda_dflash_executor.h>

#include <cuda_runtime.h>
#include <executorch/backends/cuda/runtime/backend_options.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/backend/backend_options_map.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace executorch::extension::llm {
using batching::SessionId;
using batching::Token;
using runtime::Error;
using runtime::Result;
namespace df = cuda_dflash_detail;
namespace cb = ::executorch::backends::cuda::batching;

namespace cuda_dflash_detail {
std::optional<GreedyBurst> verify_greedy(
    const std::vector<Token>& proposals,
    const std::vector<Token>& predictions) {
  if (proposals.empty() || predictions.size() != proposals.size() + 1) {
    return std::nullopt;
  }
  GreedyBurst result;
  for (std::size_t row = 0; row < predictions.size(); ++row) {
    if (row < proposals.size() && proposals[row] == predictions[row]) {
      result.tokens.push_back(proposals[row]);
      ++result.accepted;
    } else {
      result.tokens.push_back(predictions[row]);
      result.bonus = row == proposals.size();
      break;
    }
  }
  return result;
}

bool trim_hidden(
    std::vector<std::uint8_t>& hidden,
    std::size_t row_bytes,
    int draft_frontier,
    int position) {
  if (row_bytes == 0 || draft_frontier < 0 || position < 0 ||
      hidden.size() % row_bytes != 0) {
    return false;
  }
  const auto rows =
      static_cast<std::size_t>(std::max(0, position - draft_frontier));
  if (rows > hidden.size() / row_bytes) {
    return false;
  }
  hidden.resize(rows * row_bytes);
  return true;
}

std::optional<DraftPlan> pack_draft(
    const std::vector<DraftContext>& owners,
    int block,
    Token mask,
    int context_limit,
    int max_context,
    std::size_t row_bytes) {
  if (owners.empty() || owners.size() > 4 || block < 2 || context_limit < 0 ||
      max_context < block || row_bytes == 0 ||
      block > (std::numeric_limits<int>::max() - context_limit) /
          static_cast<int>(owners.size())) {
    return std::nullopt;
  }
  DraftPlan plan;
  const int q = static_cast<int>(owners.size()) * block;
  plan.tokens.assign(q, static_cast<std::int64_t>(mask));
  std::unordered_set<std::int32_t> seen;
  int c = 0;
  for (std::size_t owner = 0; owner < owners.size(); ++owner) {
    const auto& row = owners[owner];
    if (!seen.insert(row.id).second || row.id < 0 || !row.hidden ||
        row.hidden->size() % row_bytes != 0 ||
        row.count < 0 || row.position < 0 || row.count > context_limit - c ||
        row.position > max_context - block - row.count ||
        static_cast<std::size_t>(row.count) > row.hidden->size() / row_bytes) {
      return std::nullopt;
    }
    plan.tokens[owner * block] = static_cast<std::int64_t>(row.anchor);
    plan.counts.push_back(row.count);
    plan.ends.push_back(row.position + row.count);
    plan.hidden.insert(
        plan.hidden.end(),
        row.hidden->begin(),
        row.hidden->begin() + static_cast<std::size_t>(row.count) * row_bytes);
    for (int j = 0; j < row.count; ++j) {
      plan.positions.push_back(row.position + j);
    }
    plan.ids.insert(plan.ids.end(), row.count + block, row.id);
    c += row.count;
  }
  int offset = 0;
  for (std::size_t owner = 0; owner < owners.size(); ++owner) {
    for (int j = 0; j < plan.counts[owner]; ++j) {
      plan.kv_order.push_back(offset++);
    }
    for (int j = 0; j < block; ++j) {
      plan.positions.push_back(plan.ends[owner] + j);
      plan.kv_order.push_back(c + owner * block + j);
    }
  }
  return plan;
}
} // namespace cuda_dflash_detail

namespace {
bool dense(const aten::Tensor& tensor) {
  std::int64_t stride = 1;
  for (int d = tensor.dim() - 1; d >= 0; --d) {
    if (tensor.size(d) > 1 && tensor.strides()[d] != stride) {
      return false;
    }
    stride *= tensor.size(d);
  }
  return true;
}

bool copy_host(
    void* destination,
    const aten::Tensor& tensor,
    std::size_t bytes) {
  if (tensor.device_type() == aten::DeviceType::CPU) {
    std::memcpy(destination, tensor.const_data_ptr(), bytes);
    return true;
  }
  if (tensor.device_type() != aten::DeviceType::CUDA) {
    return false;
  }
  const auto stream = ::executorch::extension::cuda::getCallerStream().value_or(
      cudaStreamPerThread);
  return cudaMemcpyAsync(
             destination,
             tensor.const_data_ptr(),
             bytes,
             cudaMemcpyDeviceToHost,
             stream) == cudaSuccess &&
      cudaStreamSynchronize(stream) == cudaSuccess;
}

Result<int> positive_int(Module& module, const char* name) {
  ET_ASSIGN_OR_RETURN(value, detail::read_required_positive_int(module, name));
  ET_CHECK_OR_RETURN_ERROR(
      value <= std::numeric_limits<int>::max(),
      InvalidProgram,
      "DFlash metadata %s is too large",
      name);
  return static_cast<int>(value);
}

Result<cache::CacheGeometry> draft_geometry(Module& module, int block) {
  ET_ASSIGN_OR_RETURN(layers, positive_int(module, "get_draft_n_caches"));
  ET_ASSIGN_OR_RETURN(
      heads, detail::read_int_tensor_method(module, "get_draft_kv_heads"));
  ET_ASSIGN_OR_RETURN(
      dims, detail::read_int_tensor_method(module, "get_draft_head_dims"));
  ET_ASSIGN_OR_RETURN(
      windows, detail::read_int_tensor_method(module, "get_draft_windows"));
  ET_CHECK_OR_RETURN_ERROR(
      heads.size() == static_cast<std::size_t>(layers) &&
          dims.size() == heads.size() && windows.size() == heads.size(),
      InvalidProgram,
      "DFlash geometry vector sizes disagree");
  cache::CacheGeometry geometry;
  geometry.dflash_block_size = block;
  for (int i = 0; i < layers; ++i) {
    geometry.layers.push_back(
        {{windows[i] ? cache::LayerPolicy::Kind::Ring
                     : cache::LayerPolicy::Kind::Flat,
          windows[i]},
         heads[i],
         dims[i]});
  }
  ET_CHECK_OR_RETURN_ERROR(
      cache::valid(geometry), InvalidProgram, "Invalid DFlash draft geometry");
  return geometry;
}

Result<std::string> cache_key_metadata(Module& module, const char* name) {
  ET_ASSIGN_OR_RETURN(outputs, module.execute(name));
  ET_CHECK_OR_RETURN_ERROR(
      outputs.size() == 1 && outputs[0].isString() &&
          !outputs[0].toString().empty(),
      InvalidProgram,
      "DFlash %s must be a nonempty string",
      name);
  return std::string(outputs[0].toString());
}
} // namespace

MuseGlimmerCudaDFlashExecutor::MuseGlimmerCudaDFlashExecutor(
    std::unique_ptr<Target> target,
    std::shared_ptr<cache::Cache> draft_cache,
    int block,
    int proposals,
    Token mask,
    int max_owners,
    int fresh_context,
    int max_sessions,
    int max_context,
    int hidden_width)
    : draft_cache_(std::move(draft_cache)),
      draft_install_(draft_cache_),
      target_(std::move(target)),
      draft_ctl_(draft_cache_->as<cache::BatchControl>()),
      block_(block),
      proposals_(proposals),
      max_owners_(max_owners),
      fresh_context_(fresh_context),
      max_sessions_(max_sessions),
      max_context_(max_context),
      hidden_width_(hidden_width),
      mask_(mask),
      row_bytes_(static_cast<std::size_t>(hidden_width) * 2) {}

MuseGlimmerCudaDFlashExecutor::~MuseGlimmerCudaDFlashExecutor() {
  for (const auto& item : sessions_) {
    target_->cache_control().seq_rm(item.second.target_id);
    if (!failed_) {
      draft_ctl_->seq_rm(item.second.draft_id);
    }
  }
}

Result<MuseGlimmerBackend> MuseGlimmerCudaDFlashExecutor::create(
    const MuseGlimmerBackendConfig& config) {
  ET_CHECK_OR_RETURN_ERROR(
      config.backend == "cuda" && config.use_dflash &&
          !config.model_path.empty() && config.max_sessions > 0 &&
          config.max_session_tokens >= 0 && config.max_session_tokens != 1 &&
          config.dflash_n_draft >= 0,
      InvalidArgument,
      "Invalid CUDA DFlash configuration");
  cb::CudaExecutorConfig target_config;
  target_config.model_path = config.model_path;
  if (!config.data_path.empty()) {
    target_config.data_files.push_back(config.data_path);
  }
  target_config.max_sessions = config.max_sessions;
  target_config.max_session_tokens = config.max_session_tokens;
  ET_ASSIGN_OR_RETURN(target, Target::create(target_config));
  auto& module = target->module();
  ET_ASSIGN_OR_RETURN(names, module.method_names());
  ET_CHECK_OR_RETURN_ERROR(
      names.count(df::kForward) &&
          !names.count("get_mutable_buffer_metadata") &&
          !names.count("vision_encoder"),
      InvalidProgram,
      "CUDA DFlash requires a tapped text-only off-graph artifact");
  ET_ASSIGN_OR_RETURN(block, positive_int(module, df::kBlock));
  ET_ASSIGN_OR_RETURN(
      declared_block, positive_int(module, "get_draft_dflash_block_size"));
  ET_ASSIGN_OR_RETURN(max_owners, positive_int(module, df::kMaxOwners));
  ET_ASSIGN_OR_RETURN(max_write, positive_int(module, df::kMaxWrite));
  ET_ASSIGN_OR_RETURN(max_cells, positive_int(module, df::kMaxCells));
  ET_ASSIGN_OR_RETURN(tap_layers, positive_int(module, df::kTargetLayers));
  ET_ASSIGN_OR_RETURN(
      layer_ids,
      detail::read_int_tensor_method(module, "get_dflash_target_layers"));
  ET_ASSIGN_OR_RETURN(mask, detail::read_int_method(module, df::kMask));
  ET_ASSIGN_OR_RETURN(
      target_key, cache_key_metadata(module, "get_dflash_target_cache_key"));
  ET_ASSIGN_OR_RETURN(
      draft_key, cache_key_metadata(module, "get_dflash_draft_cache_key"));
  ET_CHECK_OR_RETURN_ERROR(
      block >= 2 && block == declared_block && max_owners <= 4 &&
          block <= (std::numeric_limits<int>::max() - 1) / max_owners &&
          max_write > max_owners * block && max_cells > max_write && mask &&
          *mask >= 0 && *mask < target->vocab_size() &&
          target_key != draft_key &&
          layer_ids.size() == static_cast<std::size_t>(tap_layers) &&
          !layer_ids.empty() && layer_ids.front() >= 0 &&
          std::adjacent_find(
              layer_ids.begin(), layer_ids.end(), std::greater_equal<int>()) ==
              layer_ids.end(),
      InvalidProgram,
      "Invalid packed CUDA DFlash metadata");
  const int proposals =
      config.dflash_n_draft == 0 ? block - 1 : config.dflash_n_draft;
  ET_CHECK_OR_RETURN_ERROR(
      proposals > 0 && proposals < block,
      InvalidArgument,
      "DFlash n_draft must be in [1, trained block size - 1]");
  int hidden_width = 0;
  for (const auto& method : target->method_calls()) {
    ET_ASSIGN_OR_RETURN(meta, module.method_meta(method.name));
    ET_CHECK_OR_RETURN_ERROR(
        meta.num_outputs() == 2,
        InvalidProgram,
        "DFlash target %s needs logits and all-token taps",
        method.name.c_str());
    ET_ASSIGN_OR_RETURN(taps, meta.output_tensor_meta(1));
    ET_ASSIGN_OR_RETURN(tokens, meta.input_tensor_meta(0));
    ET_CHECK_OR_RETURN_ERROR(
        taps.scalar_type() == aten::ScalarType::BFloat16 &&
            taps.sizes().size() == 3 && taps.sizes()[0] == 1 &&
            taps.sizes()[1] == tokens.sizes()[1] && taps.sizes()[2] > 0 &&
            taps.sizes()[2] % tap_layers == 0 &&
            (hidden_width == 0 || hidden_width == taps.sizes()[2]),
        InvalidProgram,
        "DFlash target taps have incompatible shape/dtype");
    hidden_width = taps.sizes()[2];
  }
  ET_ASSIGN_OR_RETURN(draft_meta, module.method_meta(df::kForward));
  ET_CHECK_OR_RETURN_ERROR(
      draft_meta.num_inputs() == 4 && draft_meta.num_outputs() == 1 &&
          draft_meta.num_backends() > 0,
      InvalidProgram,
      "DFlash draft needs four packed inputs and one logits output");
  for (std::size_t i = 0; i < draft_meta.num_backends(); ++i) {
    ET_ASSIGN_OR_RETURN(backend, draft_meta.get_backend_name(i));
    ET_CHECK_OR_RETURN_ERROR(
        std::string(backend) == ::executorch::backends::cuda::kCudaBackendId,
        InvalidProgram,
        "DFlash draft must delegate only to CUDA");
  }
  ET_ASSIGN_OR_RETURN(draft_tokens, draft_meta.input_tensor_meta(0));
  ET_ASSIGN_OR_RETURN(draft_hidden, draft_meta.input_tensor_meta(1));
  ET_ASSIGN_OR_RETURN(draft_positions, draft_meta.input_tensor_meta(2));
  ET_ASSIGN_OR_RETURN(draft_order, draft_meta.input_tensor_meta(3));
  ET_ASSIGN_OR_RETURN(draft_logits, draft_meta.output_tensor_meta(0));
  ET_CHECK_OR_RETURN_ERROR(
      draft_tokens.scalar_type() == aten::ScalarType::Long &&
          draft_tokens.sizes().size() == 2 && draft_tokens.sizes()[0] == 1 &&
          draft_tokens.sizes()[1] > 0 && draft_tokens.sizes()[1] % block == 0 &&
          draft_hidden.scalar_type() == aten::ScalarType::BFloat16 &&
          draft_hidden.sizes().size() == 3 && draft_hidden.sizes()[0] == 1 &&
          draft_hidden.sizes()[2] == hidden_width &&
          draft_positions.scalar_type() == aten::ScalarType::Long &&
          draft_positions.sizes().size() == 1 &&
          draft_order.scalar_type() == aten::ScalarType::Long &&
          draft_order.sizes().size() == 1 &&
          draft_order.sizes()[0] == draft_positions.sizes()[0] &&
          draft_logits.scalar_type() == aten::ScalarType::Float &&
          draft_logits.sizes().size() >= 2 &&
          draft_logits.sizes()[draft_logits.sizes().size() - 1] ==
              target->vocab_size(),
      InvalidProgram,
      "DFlash draft packed tensor ABI mismatch");
  ET_ASSIGN_OR_RETURN(geometry, draft_geometry(module, block));
  const int temporary = max_owners * block;
  const int draft_budget = (max_cells - 1 - temporary) / config.max_sessions;
  const int max_context = std::min(target->max_session_tokens(), draft_budget);
  ET_CHECK_OR_RETURN_ERROR(
      max_context > block &&
          draft_tokens.sizes()[1] == temporary &&
          draft_hidden.sizes()[1] >= max_write - temporary &&
          draft_positions.sizes()[0] >= max_write &&
          (!config.max_session_tokens ||
           max_context == config.max_session_tokens),
      InvalidArgument,
      "DFlash draft pool cannot hold resident contexts plus temporary owner blocks");
  cache::CacheConfig cache_config{};
  cache_config.capacity = max_cells;
  cache_config.kv_dtype = static_cast<int>(aten::ScalarType::BFloat16);
  cache_config.max_write = max_write;
  ET_ASSIGN_OR_RETURN(
      draft_cache,
      cache::CacheFactory::global().build(
          ::executorch::backends::cuda::kCudaBackendId,
          cache::kind::kBatchedCell,
          geometry,
          cache_config));
  const auto* control = draft_cache->as<cache::BatchControl>();
  ET_CHECK_OR_RETURN_ERROR(
      control && draft_cache->as<::executorch::backends::cuda::CudaKVCache>() &&
          (!control->max_seqs() || config.max_sessions <= *control->max_seqs()),
      InvalidProgram,
      "DFlash draft cache does not support requested resident sessions");
  ET_CHECK_OR_RETURN_ERROR(
      target->vocab_size() > 200092 &&
          config.bos_id < static_cast<std::uint64_t>(target->vocab_size()),
      InvalidProgram,
      "Muse Glimmer DFlash vocabulary cannot hold its special tokens");
  auto spec = std::make_shared<const MuseGlimmerPreparationSpec>(
      MuseGlimmerPreparationSpec{
          aten::ScalarType::BFloat16,
          0,
          max_context,
          static_cast<std::int32_t>(target->preferred_batch_tokens()),
          target->vocab_size(),
          false,
          0,
          config.image_limits,
          config.bos_id});
  auto executor =
      std::unique_ptr<batching::Executor>(new MuseGlimmerCudaDFlashExecutor(
          std::move(target),
          std::move(draft_cache),
          block,
          proposals,
          static_cast<Token>(*mask),
          max_owners,
          max_write - temporary,
          config.max_sessions,
          max_context,
          hidden_width));
  return MuseGlimmerBackend{std::move(executor), std::move(spec)};
}

std::size_t MuseGlimmerCudaDFlashExecutor::preferred_batch_tokens() const {
  return target_->preferred_batch_tokens();
}

bool MuseGlimmerCudaDFlashExecutor::initialize() {
  if (failed_) {
    return false;
  }
  if (initialized_) {
    return true;
  }
  if (!target_->initialize() ||
      batching::load_method_with_cache(
          target_->module(), df::kForward,
          ::executorch::backends::cuda::kCudaBackendId, draft_install_) != Error::Ok) {
    ET_LOG(
        Error,
        "CUDA DFlash: could not initialize distinct target/draft caches");
    return false;
  }
  initialized_ = true;
  return true;
}

std::optional<SessionId> MuseGlimmerCudaDFlashExecutor::open_session() {
  if (failed_ || sessions_.size() >= static_cast<std::size_t>(max_sessions_) ||
      next_session_ == std::numeric_limits<SessionId>::max()) {
    return std::nullopt;
  }
  auto target_id = target_->cache_control().seq_new();
  if (!target_id) {
    return std::nullopt;
  }
  auto draft_id = draft_ctl_->seq_new();
  if (!draft_id) {
    target_->cache_control().seq_rm(*target_id);
    return std::nullopt;
  }
#if ET_HAS_EXCEPTIONS
  try {
#endif
    const auto session = next_session_++;
    sessions_.emplace(session, State{*target_id, *draft_id});
    return session;
#if ET_HAS_EXCEPTIONS
  } catch (...) {
    target_->cache_control().seq_rm(*target_id);
    draft_ctl_->seq_rm(*draft_id);
    return std::nullopt;
  }
#endif
}

void MuseGlimmerCudaDFlashExecutor::close_session(SessionId session) {
  auto it = sessions_.find(session);
  if (it == sessions_.end()) {
    return;
  }
  target_->cache_control().seq_rm(it->second.target_id);
  if (!failed_) {
    draft_ctl_->seq_rm(it->second.draft_id);
  }
  sessions_.erase(it);
}

std::optional<SessionId> MuseGlimmerCudaDFlashExecutor::clone(
    SessionId source,
    batching::Position upto) {
  const auto it = sessions_.find(source);
  if (failed_ || it == sessions_.end() || it->second.poisoned || upto < 0 ||
      upto > it->second.frontier ||
      sessions_.size() >= static_cast<std::size_t>(max_sessions_) ||
      next_session_ == std::numeric_limits<SessionId>::max()) {
    return std::nullopt;
  }
  const auto& state = it->second;
  const int draft_position = draft_ctl_->pos(state.draft_id);
  const int draft_upto = std::min<int>(upto, draft_position);
  auto target_id = upto == 0
      ? target_->cache_control().seq_new()
      : target_->cache_control().seq_clone(state.target_id, upto);
  if (!target_id) {
    return std::nullopt;
  }
  auto draft_id = draft_upto == 0
      ? draft_ctl_->seq_new()
      : draft_ctl_->seq_clone(state.draft_id, draft_upto);
  if (!draft_id) {
    target_->cache_control().seq_rm(*target_id);
    return std::nullopt;
  }
#if ET_HAS_EXCEPTIONS
  try {
#endif
    State copy{*target_id, *draft_id};
    copy.frontier = upto;
    copy.hidden = state.hidden;
    if (!df::trim_hidden(copy.hidden, row_bytes_, draft_position, upto)) {
      target_->cache_control().seq_rm(*target_id);
      draft_ctl_->seq_rm(*draft_id);
      return std::nullopt;
    }
    const auto session = next_session_++;
    sessions_.emplace(session, std::move(copy));
    return session;
#if ET_HAS_EXCEPTIONS
  } catch (...) {
    target_->cache_control().seq_rm(*target_id);
    draft_ctl_->seq_rm(*draft_id);
    return std::nullopt;
  }
#endif
}

void MuseGlimmerCudaDFlashExecutor::set_sampling(
    SessionId session,
    const batching::SamplingParams& params,
    std::optional<std::uint64_t> seed) {
  const auto it = sessions_.find(session);
  if (it != sessions_.end()) {
    it->second.sampling = params;
    it->second.seed = seed;
    if (!std::isfinite(params.temperature) || params.temperature != 0.0f) {
      ET_LOG(
          Error,
          "CUDA DFlash supports temperature=0 only; sampled DFlash is not implemented");
    }
  }
}

bool MuseGlimmerCudaDFlashExecutor::rewind(State& state, int position) {
  const int draft_position = draft_ctl_->pos(state.draft_id);
  if (position > state.frontier ||
      !df::trim_hidden(state.hidden, row_bytes_, draft_position, position) ||
      !target_->cache_control().rewind(state.target_id, position) ||
      (position < draft_position &&
       !draft_ctl_->rewind(state.draft_id, position))) {
    state.poisoned = true;
    return false;
  }
  state.frontier = position;
  return true;
}

bool MuseGlimmerCudaDFlashExecutor::run_target(
    const RawStep& step,
    const std::vector<SessionId>& token_owners,
    std::vector<Token>& predictions,
    bool drain_context) {
  if (token_owners.size() != step.tokens.size()) {
    return false;
  }
  predictions.resize(step.logit_indices.size());
  std::size_t predicted = 0;
  const bool ok =
      target_->execute_raw(step, [&](const cb::RawTargetOutput& output) {
        if (!output.taps ||
            output.taps->scalar_type() != aten::ScalarType::BFloat16 ||
            output.taps->size(2) != hidden_width_) {
          return false;
        }
        std::vector<std::uint8_t> copied(
            static_cast<std::size_t>(output.length) * row_bytes_);
        // Copy every real tap before invoking the sampler; no borrowed GPU
        // output survives any subsequent Module::execute call.
        if (!copy_host(copied.data(), *output.taps, copied.size())) {
          return false;
        }
        auto tokens =
            target_->argmax_rows(output.logits, output.selected_indices.size());
        if (!tokens) {
          return false;
        }
        for (std::size_t row = 0; row < tokens->size(); ++row) {
          predictions[output.selected_indices[row]] = (*tokens)[row];
          ++predicted;
        }
        std::unordered_set<SessionId> touched;
        for (int row = 0; row < output.length; ++row) {
          const int index = output.offset + row;
          const auto owner = token_owners[index];
          auto& state = sessions_.at(owner);
          state.hidden.insert(
              state.hidden.end(),
              copied.begin() + static_cast<std::size_t>(row) * row_bytes_,
              copied.begin() + static_cast<std::size_t>(row + 1) * row_bytes_);
          state.frontier = static_cast<int>(step.positions[index]) + 1;
          touched.insert(owner);
        }
        if (drain_context) {
          for (const auto owner : touched) {
            if (!drain(owner)) {
              return false;
            }
          }
        }
        return true;
      });
  return ok &&
      predicted ==
      static_cast<std::size_t>(std::count_if(
          step.logit_indices.begin(), step.logit_indices.end(), [](auto index) {
            return index >= 0;
          }));
}

bool MuseGlimmerCudaDFlashExecutor::drain(SessionId owner) {
  auto& state = sessions_.at(owner);
  while (state.hidden.size() / row_bytes_ >
         static_cast<std::size_t>(fresh_context_)) {
    const int count = std::min(
        fresh_context_,
        max_context_ - block_ - draft_ctl_->pos(state.draft_id));
    if (count <= 0) {
      break; // Near the context limit: ordinary target decode, no draft round.
    }
    if (!draft({owner}, {mask_}, count, nullptr)) {
      return false;
    }
  }
  return true;
}

bool MuseGlimmerCudaDFlashExecutor::draft(
    const std::vector<SessionId>& owners,
    const std::vector<Token>& anchors,
    int context_limit,
    std::vector<std::vector<Token>>* proposals) {
  if (owners.size() != anchors.size() || owners.empty() ||
      owners.size() > static_cast<std::size_t>(max_owners_)) {
    return false;
  }
  std::vector<df::DraftContext> contexts;
  int remaining = context_limit;
  for (std::size_t i = 0; i < owners.size(); ++i) {
    auto& state = sessions_.at(owners[i]);
    const auto rows = state.hidden.size() / row_bytes_;
    const int count = static_cast<int>(std::min<std::size_t>(remaining, rows));
    if (remaining < 0 ||
        (proposals && rows != static_cast<std::size_t>(count))) {
      return false;
    }
    contexts.push_back(
        {state.draft_id,
         draft_ctl_->pos(state.draft_id),
         anchors[i],
         &state.hidden,
         count});
    remaining -= count;
  }
  auto plan = df::pack_draft(
      contexts, block_, mask_, context_limit, max_context_, row_bytes_);
  if (!plan || !draft_ctl_->declare_step(plan->ids)) {
    return false;
  }
  const int q = static_cast<int>(owners.size()) * block_;
  const int c = static_cast<int>(plan->hidden.size() / row_bytes_);
  const int k = c + q;
  // Host-owned inputs are staged by the export's normal H2D boundary. Keep
  // their backing alive through the forward and its stream-fenced argmax.
  auto tokens = make_tensor_ptr({1, q}, std::move(plan->tokens));
  auto hidden = make_tensor_ptr(
      {1, c, hidden_width_},
      std::move(plan->hidden),
      aten::ScalarType::BFloat16);
  auto positions = make_tensor_ptr({k}, std::move(plan->positions));
  auto order = make_tensor_ptr({k}, std::move(plan->kv_order));
  auto result = target_->module().execute(
      df::kForward, {tokens, hidden, positions, order});
  if (!result.ok() || result->size() != 1 || !result->at(0).isTensor()) {
    return false;
  }
  ++stats_.draft_forwards;
  const auto logits = result->at(0).toTensor();
  if (logits.dim() < 2 || logits.scalar_type() != aten::ScalarType::Float ||
      !dense(logits) ||
      logits.numel() != static_cast<std::int64_t>(q) * target_->vocab_size() ||
      logits.size(logits.dim() - 1) != target_->vocab_size()) {
    return false;
  }
  std::optional<std::vector<Token>> picked;
  if (proposals) {
    picked = target_->argmax_rows(logits, q);
    if (!picked) {
      return false;
    }
    proposals->clear();
  }
  for (std::size_t i = 0; i < owners.size(); ++i) {
    auto& state = sessions_.at(owners[i]);
    // The cache worker fences and removes each temporary trailing block at
    // commit_step. A retained block is an ABI failure, never a manual fixup.
    if (draft_ctl_->pos(state.draft_id) != plan->ends[i]) {
      return false;
    }
    state.hidden.erase(
        state.hidden.begin(),
        state.hidden.begin() +
            static_cast<std::size_t>(plan->counts[i]) * row_bytes_);
    if (proposals) {
      std::vector<Token> prefix;
      for (int j = 1; j <= proposals_; ++j) {
        prefix.push_back((*picked)[i * block_ + j]);
      }
      proposals->push_back(std::move(prefix));
    }
  }
  stats_.warmed_context_tokens += c;
  return true;
}

bool MuseGlimmerCudaDFlashExecutor::speculate(
    const batching::BatchInput& batch,
    batching::BatchOutput& out) {
  std::size_t cursor = 0;
  while (cursor < batch.inputs.size()) {
    std::vector<SessionId> owners;
    std::vector<Token> anchors;
    std::vector<std::size_t> indices;
    int context = 0;
    while (cursor < batch.inputs.size()) {
      const auto& input = batch.inputs[cursor];
      auto& state = sessions_.at(input.sid);
      const auto end =
          static_cast<std::int64_t>(input.position) + input.offset + input.size;
      if (!out.outputs[cursor] || end != state.frontier ||
          end > max_context_ - block_) {
        ++cursor;
        continue;
      }
      if (!drain(input.sid)) {
        return false;
      }
      const auto rows = state.hidden.size() / row_bytes_;
      if (rows > static_cast<std::size_t>(fresh_context_)) {
        return false;
      }
      if (owners.size() == static_cast<std::size_t>(max_owners_) ||
          rows > static_cast<std::size_t>(fresh_context_ - context)) {
        break;
      }
      context += static_cast<int>(rows);
      owners.push_back(input.sid);
      anchors.push_back(out.outputs[cursor]->tokens.front());
      indices.push_back(cursor++);
    }
    if (owners.empty()) {
      continue;
    }
    std::vector<std::vector<Token>> proposals;
    if (!draft(owners, anchors, fresh_context_, &proposals)) {
      return false;
    }
    RawStep verification;
    std::vector<SessionId> token_owners;
    std::vector<int> starts;
    const int width = proposals_ + 1;
    for (std::size_t i = 0; i < owners.size(); ++i) {
      const auto& state = sessions_.at(owners[i]);
      starts.push_back(state.frontier);
      for (int j = 0; j < width; ++j) {
        verification.tokens.push_back(
            j == 0 ? anchors[i] : proposals[i][j - 1]);
        verification.positions.push_back(state.frontier + j);
        verification.seq_ids.push_back(state.target_id);
        verification.logit_indices.push_back(verification.tokens.size() - 1);
        token_owners.push_back(owners[i]);
      }
    }
    std::vector<Token> predictions;
    if (!run_target(verification, token_owners, predictions, false)) {
      return false;
    }
    for (std::size_t i = 0; i < owners.size(); ++i) {
      auto burst = df::verify_greedy(
          proposals[i],
          std::vector<Token>(
              predictions.begin() + i * width,
              predictions.begin() + (i + 1) * width));
      if (!burst ||
          !rewind(
              sessions_.at(owners[i]),
              starts[i] + static_cast<int>(burst->tokens.size()))) {
        return false;
      }
      // The regular preamble's prediction is new output, now committed by
      // verification. Append only predictions after that anchor. The last
      // prediction remains uncommitted: frontier = input.end + out.size() - 1.
      auto& output_tokens = out.outputs[indices[i]]->tokens;
      output_tokens.insert(
          output_tokens.end(), burst->tokens.begin(), burst->tokens.end());
      ++stats_.cycles;
      stats_.proposed += proposals_;
      stats_.accepted += burst->accepted;
      stats_.bonuses += burst->bonus;
      stats_.rejected += !burst->bonus;
    }
  }
  return true;
}

bool MuseGlimmerCudaDFlashExecutor::fail_batch(
    const batching::BatchInput& batch,
    batching::BatchOutput& out) {
  (void)batch;
  if (!failed_) {
    failed_ = true;
    for (auto& item : sessions_) {
      item.second.poisoned = true;
      item.second.hidden.clear();
    }
    // An abandoned declaration/forward affects the entire shared draft pool.
    // Never allocate another draft SID or remove stale IDs after this clear.
    draft_ctl_->clear();
    ET_LOG(
        Error,
        "CUDA DFlash: whole draft cache invalidated; model restart required");
  }
  for (auto& output : out.outputs) {
    output.reset();
  }
  ++stats_.failed_batches;
  return false;
}

bool MuseGlimmerCudaDFlashExecutor::execute(
    const batching::BatchInput& batch,
    batching::BatchOutput& out) {
#if ET_HAS_EXCEPTIONS
  try {
#endif
    out.outputs.clear();
    out.outputs.resize(batch.inputs.size());
    if (!initialized_ || failed_) {
      return fail_batch(batch, out);
    }
    RawStep step;
    std::vector<SessionId> token_owners;
    std::unordered_map<SessionId, int> cursor;
    std::vector<std::pair<SessionId, int>> rewinds;
    for (const auto& input : batch.inputs) {
      const auto it = sessions_.find(input.sid);
      const auto* payload =
          std::get_if<batching::TokenInputPtr>(&input.payload);
      if (it == sessions_.end() || it->second.poisoned || !payload ||
          !*payload || input.size == 0 || input.offset > (*payload)->size() ||
          input.size > (*payload)->size() - input.offset ||
          input.position < 0 ||
          input.offset > static_cast<std::size_t>(max_context_) ||
          input.size > static_cast<std::size_t>(max_context_) ||
          (input.produce_output && !it->second.sampling) ||
          (it->second.sampling &&
           (!std::isfinite(it->second.sampling->temperature) ||
            it->second.sampling->temperature != 0.0f))) {
        ET_LOG(
            Error,
            "CUDA DFlash: invalid input or unsupported sampled configuration (temperature=0 required)");
        return fail_batch(batch, out);
      }
      const auto start =
          static_cast<std::int64_t>(input.position) + input.offset;
      const auto end = start + input.size;
      auto [at, first] = cursor.emplace(input.sid, it->second.frontier);
      if (start > at->second || (start < at->second && !first) ||
          end > max_context_ ||
          target_->cache_control().pos(it->second.target_id) !=
              it->second.frontier ||
          step.tokens.size() >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) -
                  input.size) {
        return fail_batch(batch, out);
      }
      if (start < at->second) {
        rewinds.emplace_back(input.sid, static_cast<int>(start));
      }
      at->second = static_cast<int>(end);
      for (std::size_t j = 0; j < input.size; ++j) {
        const auto token = (**payload)[input.offset + j];
        if (token >= static_cast<std::uint64_t>(target_->vocab_size())) {
          return fail_batch(batch, out);
        }
        step.tokens.push_back(token);
        step.positions.push_back(start + j);
        step.seq_ids.push_back(it->second.target_id);
        token_owners.push_back(input.sid);
      }
      step.logit_indices.push_back(
          input.produce_output
              ? static_cast<std::int64_t>(step.tokens.size()) - 1
              : -1);
    }
    // All inputs are validated before either cache is truncated.
    for (const auto& item : rewinds) {
      if (!rewind(sessions_.at(item.first), item.second)) {
        return fail_batch(batch, out);
      }
    }
    std::vector<Token> predictions;
    if (!run_target(step, token_owners, predictions, true)) {
      return fail_batch(batch, out);
    }
    for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
      if (batch.inputs[i].produce_output) {
        out.outputs[i] =
            batching::Output{batch.inputs[i].sid, {predictions[i]}};
      }
    }
    const auto before = stats_;
    if (!speculate(batch, out)) {
      return fail_batch(batch, out);
    }
    ++stats_.executions;
    ET_LOG(
        Info,
        "DFLASH_STATS cycles=%llu proposed=%llu accepted=%llu rejected=%llu bonuses=%llu draft_forwards_total=%llu",
        static_cast<unsigned long long>(stats_.cycles - before.cycles),
        static_cast<unsigned long long>(stats_.proposed - before.proposed),
        static_cast<unsigned long long>(stats_.accepted - before.accepted),
        static_cast<unsigned long long>(stats_.rejected - before.rejected),
        static_cast<unsigned long long>(stats_.bonuses - before.bonuses),
        static_cast<unsigned long long>(stats_.draft_forwards));
    return true;
#if ET_HAS_EXCEPTIONS
  } catch (...) {
    ET_LOG(Error, "CUDA DFlash: batch failed; model restart required");
    return fail_batch(batch, out);
  }
#endif
}
} // namespace executorch::extension::llm
