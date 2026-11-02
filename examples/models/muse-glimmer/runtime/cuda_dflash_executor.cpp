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
#include <executorch/extension/llm/batching/executor_utils.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/tensor/tensor_ptr.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>

namespace executorch::extension::llm {
using batching::SessionId;
using batching::Token;
using runtime::Error;
using runtime::Result;
namespace df = cuda_dflash_detail;
namespace cb = ::executorch::backends::cuda::batching;

namespace cuda_dflash_detail {
std::optional<Burst> make_burst(
    const std::vector<Token>& proposals,
    std::int64_t accepted,
    std::int64_t token,
    std::int32_t vocab_size) {
  if (proposals.empty() || accepted < 0 ||
      accepted > static_cast<std::int64_t>(proposals.size()) || token < 0 ||
      token >= vocab_size) {
    return std::nullopt;
  }
  Burst result;
  result.tokens.assign(proposals.begin(), proposals.begin() + accepted);
  result.tokens.push_back(static_cast<Token>(token));
  result.accepted = static_cast<std::size_t>(accepted);
  result.bonus = result.accepted == proposals.size();
  return result;
}

float positional_uniform(
    std::uint64_t seed,
    batching::Position position,
    int purpose) {
  // SplitMix64 over (seed, purpose, position), as the MLX batching executor
  // derives its per-prediction generators.
  auto mix = [](std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  };
  const std::uint64_t value =
      mix(seed ^ mix(static_cast<std::uint64_t>(purpose)) ^
          mix(static_cast<std::uint64_t>(position) + 1));
  std::seed_seq words{
      static_cast<std::uint32_t>(value),
      static_cast<std::uint32_t>(value >> 32)};
  std::mt19937 generator(words);
  return std::uniform_real_distribution<float>(0.0f, 1.0f)(generator);
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
        row.hidden->size() % row_bytes != 0 || row.count < 0 ||
        row.position < 0 || row.count > context_limit - c ||
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

bool pad_to_static(
    DraftPlan& plan,
    int owners,
    int block,
    std::size_t row_bytes) {
  const int width = owners * block;
  const int queries = width;
  if (row_bytes == 0 || plan.hidden.size() % row_bytes != 0) {
    return false;
  }
  const int context = static_cast<int>(plan.hidden.size() / row_bytes);
  if (context > width ||
      plan.positions.size() != static_cast<std::size_t>(context + queries) ||
      plan.kv_order.size() != plan.positions.size()) {
    return false;
  }
  for (const int count : plan.counts) {
    if (count > block) {
      return false;
    }
  }
  const int pad = width - context;
  plan.hidden.resize(static_cast<std::size_t>(width) * row_bytes, 0);
  plan.positions.insert(plan.positions.begin() + context, pad, 0);
  for (auto& index : plan.kv_order) {
    if (index >= context) {
      index += pad;
    }
  }
  for (int row = context; row < width; ++row) {
    plan.kv_order.push_back(row);
  }
  return true;
}

DeviceStaging::~DeviceStaging() {
  if (device_) {
    (void)cudaFree(device_);
  }
}

std::optional<std::vector<void*>> DeviceStaging::upload(
    const std::vector<std::pair<const void*, std::size_t>>& parts,
    void* stream) {
  std::vector<std::size_t> offsets;
  std::size_t total = 0;
  for (const auto& part : parts) {
    total = (total + 7) & ~std::size_t{7};
    offsets.push_back(total);
    total += part.second;
  }
  if (total > capacity_) {
    if (device_ && cudaFree(device_) != cudaSuccess) {
      return std::nullopt;
    }
    device_ = nullptr;
    capacity_ = 0;
    if (cudaMalloc(&device_, total) != cudaSuccess) {
      return std::nullopt;
    }
    capacity_ = total;
  }
  std::vector<void*> pointers;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    auto* destination = static_cast<std::uint8_t*>(device_) + offsets[i];
    if (cudaMemcpyAsync(
            destination,
            parts[i].first,
            parts[i].second,
            cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(stream)) != cudaSuccess) {
      return std::nullopt;
    }
    pointers.push_back(destination);
  }
  return pointers;
}
} // namespace cuda_dflash_detail

namespace {
bool dense(const aten::Tensor& tensor) {
  std::int64_t stride = 1;
  for (auto d = tensor.dim() - 1; d >= 0; --d) {
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

} // namespace

namespace cuda_dflash_detail {
HiddenSnapshot::~HiddenSnapshot() {
  // Queued copies may still read the staging if the fence failed; leak it.
  if (fence() && device_) {
    (void)cudaFree(device_);
  }
}

bool HiddenSnapshot::fence() {
  if (stream_set_ &&
      cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)) !=
          cudaSuccess) {
    return false;
  }
  stream_set_ = false;
  return true;
}

bool HiddenSnapshot::begin(std::size_t rows, std::size_t row_bytes) {
  if (!fence() || rows == 0 || row_bytes == 0 ||
      rows > std::numeric_limits<std::size_t>::max() / row_bytes) {
    return false;
  }
  rows_ = rows;
  row_bytes_ = row_bytes;
  captured_ = 0;
  device_source_.reset();
  stream_ = nullptr;
  stream_set_ = false;
  return true;
}

bool HiddenSnapshot::capture(
    const aten::Tensor& taps,
    std::size_t offset,
    std::size_t rows) {
  const bool device = taps.device_type() == aten::DeviceType::CUDA;
  if (offset != captured_ || offset > rows_ || rows == 0 ||
      rows > rows_ - offset || !dense(taps) ||
      taps.nbytes() < rows * row_bytes_ ||
      (!device && taps.device_type() != aten::DeviceType::CPU) ||
      (device_source_ && *device_source_ != device)) {
    return false;
  }
  device_source_ = device;
  const auto bytes = rows * row_bytes_;
  if (device) {
    const auto stream =
        ::executorch::extension::cuda::getCallerStream().value_or(
            cudaStreamPerThread);
    if (stream_set_ && stream_ != stream) {
      return false;
    }
    stream_ = stream;
    stream_set_ = true;
    if (capacity_ < rows_ * row_bytes_) {
      if (device_ && cudaFree(device_) != cudaSuccess) {
        return false;
      }
      device_ = nullptr;
      capacity_ = 0;
      if (cudaMalloc(&device_, rows_ * row_bytes_) != cudaSuccess) {
        return false;
      }
      capacity_ = rows_ * row_bytes_;
    }
    if (cudaMemcpyAsync(
            static_cast<std::uint8_t*>(device_) + offset * row_bytes_,
            taps.const_data_ptr(),
            bytes,
            cudaMemcpyDeviceToDevice,
            stream) != cudaSuccess) {
      return false;
    }
  } else {
    host_.resize(rows_ * row_bytes_);
    std::memcpy(
        host_.data() + offset * row_bytes_, taps.const_data_ptr(), bytes);
  }
  captured_ += rows;
  return true;
}

bool HiddenSnapshot::copy_rows(
    std::vector<std::uint8_t>& destination,
    std::size_t offset,
    std::size_t rows) {
  if (offset > captured_ || rows > captured_ - offset || row_bytes_ == 0 ||
      destination.size() % row_bytes_ != 0 ||
      rows > (destination.max_size() - destination.size()) / row_bytes_) {
    return false;
  }
  const auto start = destination.size();
  const auto bytes = rows * row_bytes_;
  destination.resize(start + bytes);
  if (bytes == 0) {
    return true;
  }
  if (on_device()) {
    stream_set_ = true;
    return cudaMemcpyAsync(
               destination.data() + start,
               static_cast<std::uint8_t*>(device_) + offset * row_bytes_,
               bytes,
               cudaMemcpyDeviceToHost,
               static_cast<cudaStream_t>(stream_)) == cudaSuccess;
  }
  std::memcpy(
      destination.data() + start, host_.data() + offset * row_bytes_, bytes);
  return true;
}
} // namespace cuda_dflash_detail

namespace {
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

// Nothing is mutated yet, so the batch is refused without poisoning the model.
bool reject_batch() {
  ET_LOG(Error, "CUDA DFlash rejected an invalid batch");
  return false;
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
  for (int n = 1; n <= df::kOwnerLimit; ++n) {
    target_config.options.additional_cuda_graph_methods.push_back(
        df::kStaticForwardPrefix + std::to_string(n));
  }
  ET_ASSIGN_OR_RETURN(target, Target::create(target_config));
  auto& module = target->module();
  ET_ASSIGN_OR_RETURN(names, module.method_names());
  ET_CHECK_OR_RETURN_ERROR(
      names.count(df::kForward) && names.count(df::kVerify) &&
          names.count(df::kVerifyArgmax) &&
          !names.count("get_mutable_buffer_metadata") &&
          !names.count("vision_encoder"),
      InvalidProgram,
      "CUDA DFlash requires a tapped text-only off-graph artifact with its verifiers");
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
      block >= 2 && block == declared_block && max_owners <= df::kOwnerLimit &&
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
  // The exported verifiers take two to block - 1 proposals.
  ET_CHECK_OR_RETURN_ERROR(
      proposals >= 2 && proposals < block,
      InvalidArgument,
      "DFlash n_draft must be in [2, trained block size - 1]");
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
          draft_tokens.sizes().size() == 2 && draft_tokens.sizes()[0] > 0 &&
          draft_tokens.sizes()[1] == block &&
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
      max_context > block && draft_tokens.sizes()[0] == max_owners &&
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
  auto& module = target_->module();
  if (!target_->initialize() ||
      batching::load_method_with_cache(
          module,
          df::kForward,
          ::executorch::backends::cuda::kCudaBackendId,
          draft_install_) != Error::Ok ||
      module.load_method(df::kVerify) != Error::Ok ||
      module.load_method(df::kVerifyArgmax) != Error::Ok) {
    ET_LOG(
        Error,
        "CUDA DFlash: could not initialize distinct target/draft caches");
    return false;
  }
  const auto names = module.method_names();
  if (!names.ok()) {
    ET_LOG(Error, "CUDA DFlash: could not list the program's methods");
    return false;
  }
  for (int n = 1; n <= max_owners_; ++n) {
    const auto name = df::kStaticForwardPrefix + std::to_string(n);
    if (!names->count(name)) {
      continue;
    }
    const auto meta = module.method_meta(name);
    const int width = n * block_;
    auto shaped = [&](std::size_t index, bool input, std::vector<int> sizes) {
      auto tensor = input ? meta->input_tensor_meta(index)
                          : meta->output_tensor_meta(index);
      return tensor.ok() &&
          std::equal(
                 tensor->sizes().begin(),
                 tensor->sizes().end(),
                 sizes.begin(),
                 sizes.end());
    };
    if (!meta.ok() || meta->num_inputs() != 4 || meta->num_outputs() != 1 ||
        !shaped(0, true, {n, block_}) ||
        !shaped(1, true, {1, width, hidden_width_}) ||
        !shaped(2, true, {2 * width}) || !shaped(3, true, {2 * width}) ||
        !shaped(0, false, {1, width, target_->vocab_size()}) ||
        batching::load_method_with_cache(
            module,
            name,
            ::executorch::backends::cuda::kCudaBackendId,
            draft_install_) != Error::Ok) {
      ET_LOG(Error, "CUDA DFlash: invalid static draft %s", name.c_str());
      return false;
    }
    static_drafts_[n] = true;
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
    copy.pending = upto == state.frontier ? state.pending : std::nullopt;
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
  if (it == sessions_.end()) {
    return;
  }
  auto& state = it->second;
  state.sampling.reset();
  if (!std::isfinite(params.temperature) || params.temperature < 0 ||
      !std::isfinite(params.top_p) || params.top_p < 0 || params.top_p > 1) {
    return;
  }
#if ET_HAS_EXCEPTIONS
  try {
#endif
    if (seed) {
      state.seed = *seed;
    } else {
      std::random_device device;
      state.seed = (static_cast<std::uint64_t>(device()) << 32) ^ device();
    }
    state.sampling = params;
#if ET_HAS_EXCEPTIONS
  } catch (const std::exception& error) {
    ET_LOG(
        Error, "CUDA DFlash: could not initialize sampling: %s", error.what());
  }
#endif
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
  state.pending.reset();
  return true;
}

bool MuseGlimmerCudaDFlashExecutor::run_target(
    const RawStep& step,
    const std::vector<SessionId>& token_owners,
    std::vector<Token>& predictions) {
  if (token_owners.size() != step.tokens.size()) {
    return false;
  }
  predictions.resize(step.logit_indices.size());
  ++stats_.target_preamble_calls;
  std::size_t predicted = 0;
  const bool ok =
      target_->execute_raw(step, [&](const cb::RawTargetOutput& output) {
        ++stats_.target_preamble_forwards;
        if (!output.taps || output.taps->dim() != 3 ||
            output.taps->scalar_type() != aten::ScalarType::BFloat16 ||
            output.taps->size(2) != hidden_width_) {
          return false;
        }
        // The sampler or the next target slice may invalidate these taps.
        std::vector<std::uint8_t> copied(
            static_cast<std::size_t>(output.length) * row_bytes_);
        if (!copy_host(copied.data(), *output.taps, copied.size())) {
          return false;
        }
        if (output.taps->device_type() == aten::DeviceType::CUDA) {
          stats_.hidden_rows_d2h += output.length;
          stats_.hidden_bytes_d2h += copied.size();
        }
        std::vector<float> params;
        for (const auto selected : output.selected_indices) {
          const auto index = step.logit_indices[selected];
          const auto& state = sessions_.at(token_owners[index]);
          const auto& sampling = *state.sampling;
          params.insert(
              params.end(),
              {sampling.temperature,
               sampling.top_p,
               static_cast<float>(sampling.top_k),
               sampling.temperature > 0.0f
                   ? df::positional_uniform(
                         *state.seed,
                         static_cast<batching::Position>(
                             step.positions[index] + 1),
                         0)
                   : 0.0f});
        }
        auto tokens = target_->sample_rows(
            output.logits, output.selected_indices.size(), params);
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
        for (const auto owner : touched) {
          if (!drain(owner)) {
            return false;
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

std::vector<float> MuseGlimmerCudaDFlashExecutor::verify_params(
    const State& state,
    batching::Position first) const {
  const auto& sampling = *state.sampling;
  std::vector<float> params(proposals_ + 4, 0.0f);
  params[0] = sampling.temperature;
  params[1] = static_cast<float>(sampling.top_k);
  params[2] = sampling.top_p;
  if (sampling.temperature > 0.0f) {
    for (int j = 0; j < proposals_; ++j) {
      params[3 + j] = df::positional_uniform(*state.seed, first + j, 2);
    }
    params.back() = df::positional_uniform(*state.seed, first, 3);
  }
  return params;
}

bool MuseGlimmerCudaDFlashExecutor::verify(
    const RawStep& step,
    const std::vector<SessionId>& owners,
    const std::vector<std::int64_t>& candidates,
    std::vector<std::int64_t>& results) {
  const int width = proposals_ + 1;
  if (!verification_hidden_.begin(step.tokens.size(), row_bytes_) ||
      candidates.size() != owners.size() * width) {
    return false;
  }
  results.assign(owners.size() * 2, -1);
  ++stats_.target_verify_calls;
  return target_->execute_raw(step, [&](const cb::RawTargetOutput& output) {
    ++stats_.target_verify_forwards;
    if (!output.taps || output.taps->dim() != 3 ||
        output.taps->scalar_type() != aten::ScalarType::BFloat16 ||
        output.taps->size(2) != hidden_width_ ||
        !verification_hidden_.capture(
            *output.taps, output.offset, output.length)) {
      return false;
    }
    if (verification_hidden_.on_device()) {
      stats_.hidden_rows_d2d += output.length;
      stats_.hidden_bytes_d2d += output.length * row_bytes_;
    }
    // Every verification row samples, so a slice holds whole owner blocks.
    if (output.offset % width != 0 || output.length % width != 0 ||
        output.selected_indices.size() !=
            static_cast<std::size_t>(output.length) ||
        output.logits.device_type() != aten::DeviceType::CUDA ||
        output.logits.scalar_type() != aten::ScalarType::Float) {
      return false;
    }
    const int first = output.offset / width;
    const int count = output.length / width;
    const bool greedy = std::all_of(
        owners.begin() + first,
        owners.begin() + first + count,
        [this](SessionId owner) {
          return sessions_.at(owner).sampling->temperature == 0.0f;
        });
    std::vector<float> params;
    if (!greedy) {
      params.reserve(count * (width + 3));
      for (int i = first; i < first + count; ++i) {
        const auto row = verify_params(
            sessions_.at(owners[i]),
            static_cast<batching::Position>(step.positions[i * width] + 1));
        params.insert(params.end(), row.begin(), row.end());
      }
    }
    const auto stream =
        ::executorch::extension::cuda::getCallerStream().value_or(
            cudaStreamPerThread);
    std::vector<std::pair<const void*, std::size_t>> parts{
        {candidates.data() + first * width,
         static_cast<std::size_t>(count * width) * sizeof(std::int64_t)}};
    if (!greedy) {
      parts.emplace_back(params.data(), params.size() * sizeof(float));
    }
    auto staged = verify_inputs_.upload(parts, stream);
    if (!staged) {
      return false;
    }
    const aten::Device cuda(
        aten::DeviceType::CUDA, output.logits.device_index());
    auto logits = make_tensor_ptr(
        {count, width, target_->vocab_size()},
        output.logits.mutable_data_ptr(),
        aten::ScalarType::Float,
        cuda);
    auto tokens = make_tensor_ptr(
        {count, width}, (*staged)[0], aten::ScalarType::Long, cuda);
    auto result = greedy
        ? target_->module().execute(
              df::kVerifyArgmax, std::vector<runtime::EValue>{logits, tokens})
        : target_->module().execute(
              df::kVerify,
              {logits,
               tokens,
               make_tensor_ptr(
                   {count, width + 3},
                   (*staged)[1],
                   aten::ScalarType::Float,
                   cuda)});
    if (!result.ok() || result->size() != 1 || !result->at(0).isTensor()) {
      return false;
    }
    const auto& verdict = result->at(0).toTensor();
    if (verdict.scalar_type() != aten::ScalarType::Long ||
        verdict.numel() != 2 * count) {
      return false;
    }
    return copy_host(
        results.data() + 2 * first, verdict, 2 * count * sizeof(std::int64_t));
  });
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
    if (remaining < 0) {
      return false;
    }
    const auto rows = state.hidden.size() / row_bytes_;
    const int count =
        static_cast<int>(std::min(static_cast<std::size_t>(remaining), rows));
    if (proposals && rows != static_cast<std::size_t>(count)) {
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
  // Decode contexts fit one block per owner; take the static, graph-captured
  // draft. Longer contexts drain through the dynamic one.
  const bool use_static = proposals && static_drafts_[owners.size()] &&
      std::all_of(plan->counts.begin(), plan->counts.end(), [this](int n) {
                            return n <= block_;
                          });
  if (use_static &&
      !df::pad_to_static(
          *plan, static_cast<int>(owners.size()), block_, row_bytes_)) {
    return false;
  }
  const int padded = static_cast<int>(plan->hidden.size() / row_bytes_);
  const int k = padded + q;
  // Host-owned inputs are staged by the export's normal H2D boundary. Keep
  // their backing alive through the forward and its stream-fenced argmax.
  auto tokens = make_tensor_ptr(
      {static_cast<int>(owners.size()), block_}, std::move(plan->tokens));
  auto hidden = make_tensor_ptr(
      {1, padded, hidden_width_},
      std::move(plan->hidden),
      aten::ScalarType::BFloat16);
  auto positions = make_tensor_ptr({k}, std::move(plan->positions));
  auto order = make_tensor_ptr({k}, std::move(plan->kv_order));
  const std::string method = use_static
      ? df::kStaticForwardPrefix + std::to_string(owners.size())
      : std::string(df::kForward);
  auto result =
      target_->module().execute(method, {tokens, hidden, positions, order});
  if (!result.ok() || result->size() != 1 || !result->at(0).isTensor()) {
    return false;
  }
  ++stats_.draft_forwards;
  stats_.static_draft_forwards += use_static;
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
    batching::BatchOutput& out,
    bool pending_decode) {
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
      const int seed_rows = pending_decode ? 1 : 0;
      if (!out.outputs[cursor] || end != state.frontier + seed_rows ||
          end - seed_rows > max_context_ - block_) {
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
      anchors.push_back(
          pending_decode ? (*std::get<batching::TokenInputPtr>(
                               input.payload))[input.offset]
                         : out.outputs[cursor]->tokens.front());
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
    std::vector<std::int64_t> candidates;
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
        candidates.push_back(verification.tokens.back());
      }
    }
    std::vector<std::int64_t> verdicts;
    if (!verify(verification, owners, candidates, verdicts)) {
      return false;
    }
    std::vector<df::Burst> bursts;
    for (std::size_t i = 0; i < owners.size(); ++i) {
      auto burst = df::make_burst(
          proposals[i],
          verdicts[2 * i],
          verdicts[2 * i + 1],
          target_->vocab_size());
      if (!burst ||
          !verification_hidden_.copy_rows(
              sessions_.at(owners[i]).hidden, i * width, burst->accepted + 1)) {
        // Earlier owners' D2H copies are still queued into their hidden rows.
        (void)verification_hidden_.fence();
        return false;
      }
      if (verification_hidden_.on_device()) {
        stats_.hidden_rows_d2h += burst->accepted + 1;
        stats_.hidden_bytes_d2h += (burst->accepted + 1) * row_bytes_;
      }
      bursts.push_back(std::move(*burst));
    }
    // All host backing remains stable until this coalesced fence, even when
    // verification spanned several target callbacks/static padding shapes.
    if (!verification_hidden_.fence()) {
      return false;
    }
    for (std::size_t i = 0; i < owners.size(); ++i) {
      auto& state = sessions_.at(owners[i]);
      const auto& burst = bursts[i];
      const int committed = starts[i] + static_cast<int>(burst.accepted) + 1;
      // Rejected KV was written speculatively, but rejected hidden was never
      // appended. Do not use the ordinary rewind/hidden trimming path here.
      if (!target_->cache_control().rewind(state.target_id, committed)) {
        return false;
      }
      state.frontier = committed;
      // Pending decode consumes Input's already-emitted token; preamble mode
      // consumes its new anchor. Both append only new predictions, leaving
      // correction/bonus pending: frontier = input.end + out.size() - 1.
      auto& output_tokens = out.outputs[indices[i]]->tokens;
      output_tokens.insert(
          output_tokens.end(), burst.tokens.begin(), burst.tokens.end());
      ++stats_.cycles;
      stats_.proposed += proposals_;
      stats_.accepted += burst.accepted;
      stats_.bonuses += burst.bonus;
      stats_.rejected += !burst.bonus;
    }
  }
  return true;
}

bool MuseGlimmerCudaDFlashExecutor::fail_batch(
    const batching::BatchInput& batch,
    batching::BatchOutput& out) {
  (void)batch;
  // A failed callback/copy may still have queued work touching host backing;
  // if that cannot be fenced, keep the backing alive rather than release it.
  const bool fenced = verification_hidden_.fence();
  if (!failed_) {
    failed_ = true;
    for (auto& item : sessions_) {
      item.second.poisoned = true;
      item.second.pending.reset();
      if (fenced) {
        item.second.hidden.clear();
      }
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
    if (!initialized_) {
      return reject_batch();
    }
    if (failed_) {
      return fail_batch(batch, out);
    }
    RawStep step;
    std::vector<SessionId> token_owners;
    std::unordered_map<SessionId, int> cursor;
    std::vector<std::pair<SessionId, int>> rewinds;
    bool pending_decode = !batch.inputs.empty();
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
            it->second.sampling->temperature < 0.0f))) {
        ET_LOG(Error, "CUDA DFlash: invalid input or sampling configuration");
        return reject_batch();
      }
      const auto start =
          static_cast<std::int64_t>(input.position) + input.offset;
      const auto end = start + input.size;
      auto [at, first] = cursor.emplace(input.sid, it->second.frontier);
      if (target_->cache_control().pos(it->second.target_id) !=
          it->second.frontier) {
        return fail_batch(batch, out);
      }
      if (start > at->second || (start < at->second && !first) ||
          end > max_context_ ||
          step.tokens.size() >
              static_cast<std::size_t>(std::numeric_limits<int>::max()) -
                  input.size) {
        return reject_batch();
      }
      if (start < at->second) {
        rewinds.emplace_back(input.sid, static_cast<int>(start));
      }
      // A one-token prompt/replay is not necessarily emitted decode feedback.
      // Stay on the legacy path unless every owner has an exact pending seed,
      // one input, and room for the complete trained draft block.
      pending_decode = pending_decode && first && input.produce_output &&
          input.size == 1 && start == it->second.frontier &&
          start <= max_context_ - block_ && it->second.pending &&
          *it->second.pending == (**payload)[input.offset];
      at->second = static_cast<int>(end);
      for (std::size_t j = 0; j < input.size; ++j) {
        const auto token = (**payload)[input.offset + j];
        if (token >= static_cast<std::uint64_t>(target_->vocab_size())) {
          return reject_batch();
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
    if (pending_decode) {
      ++stats_.pending_decode_batches;
      stats_.pending_decode_owners += batch.inputs.size();
      for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
        out.outputs[i] = batching::Output{batch.inputs[i].sid, {}};
      }
    } else {
      std::vector<Token> predictions;
      if (!run_target(step, token_owners, predictions)) {
        return fail_batch(batch, out);
      }
      for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
        sessions_.at(batch.inputs[i].sid).pending.reset();
        if (batch.inputs[i].produce_output) {
          out.outputs[i] =
              batching::Output{batch.inputs[i].sid, {predictions[i]}};
        }
      }
    }
    if (!speculate(batch, out, pending_decode)) {
      return fail_batch(batch, out);
    }
    for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
      if (out.outputs[i]) {
        const auto& tokens = out.outputs[i]->tokens;
        const auto& input = batch.inputs[i];
        auto& state = sessions_.at(input.sid);
        const auto end = static_cast<std::int64_t>(input.position) +
            input.offset + input.size;
        if (tokens.empty()) {
          return fail_batch(batch, out);
        }
        const auto expected =
            end + static_cast<std::int64_t>(tokens.size()) - 1;
        if (pending_decode && expected != state.frontier) {
          return fail_batch(batch, out);
        }
        // Only the final input for a repeated owner may seed the next cycle.
        if (expected == state.frontier) {
          state.pending = tokens.back();
        }
      }
    }
    ++stats_.executions;
    return true;
#if ET_HAS_EXCEPTIONS
  } catch (...) {
    ET_LOG(Error, "CUDA DFlash: batch failed; model restart required");
    return fail_batch(batch, out);
  }
#endif
}
} // namespace executorch::extension::llm
