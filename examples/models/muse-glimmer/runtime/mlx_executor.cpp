/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/mlx_executor.h>

#include <executorch/examples/models/muse-glimmer/runtime/embedding_materializer.h>
#include <executorch/examples/models/muse-glimmer/runtime/engine/sampling.h>
#include <executorch/extension/llm/batching/executor_utils.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <limits>
#include <random>

namespace executorch::extension::llm {
namespace {
using runtime::Error;
constexpr const char* kDecoder = "forward_from_embeddings";
constexpr const char* kBackend = "MLXBackend";

bool tensor_abi(
    const runtime::TensorInfo& info,
    aten::ScalarType dtype,
    std::initializer_list<int64_t> shape) {
  if (info.scalar_type() != dtype || info.sizes().size() != shape.size() ||
      info.dim_order().size() != shape.size()) {
    return false;
  }
  size_t bytes = runtime::elementSize(dtype);
  size_t dim = 0;
  for (int64_t size : shape) {
    if (size <= 0 || size != info.sizes()[dim] ||
        info.dim_order()[dim] != dim ||
        static_cast<uint64_t>(size) >
            std::numeric_limits<size_t>::max() / bytes) {
      return false;
    }
    bytes *= static_cast<size_t>(size);
    ++dim;
  }
  return info.nbytes() == bytes;
}

bool dense_tensor(const aten::Tensor& tensor) {
  int64_t stride = 1;
  for (int dim = tensor.dim() - 1; dim >= 0; --dim) {
    if (tensor.size(dim) <= 0 ||
        (tensor.size(dim) != 1 && tensor.strides()[dim] != stride) ||
        stride > std::numeric_limits<int64_t>::max() / tensor.size(dim)) {
      return false;
    }
    stride *= tensor.size(dim);
  }
  return tensor.device().is_cpu() && tensor.const_data_ptr() != nullptr;
}

Error check_mlx_backends(const runtime::MethodMeta& meta) {
  if (meta.num_backends() == 0) {
    return Error::InvalidProgram;
  }
  for (size_t i = 0; i < meta.num_backends(); ++i) {
    ET_ASSIGN_OR_RETURN(name, meta.get_backend_name(i));
    if (std::string(name) != kBackend) {
      return Error::NotSupported;
    }
  }
  return Error::Ok;
}

// Metadata can be read during create(), but no backend may load off-engine.
Error check_constant_method(Module& module, const char* name) {
  ET_ASSIGN_OR_RETURN(meta, module.method_meta(name));
  if (meta.num_inputs() != 0 || meta.num_outputs() != 1 ||
      meta.num_backends() != 0 || meta.num_instructions() != 0) {
    ET_LOG(Error, "Muse Glimmer metadata %s must be constant", name);
    return Error::InvalidProgram;
  }
  return Error::Ok;
}

runtime::Result<int64_t> check_vision_abi(
    Module& module,
    int32_t hidden,
    aten::ScalarType dtype,
    int max_patches) {
  ET_CHECK_OK_OR_RETURN_ERROR(
      check_constant_method(module, "get_max_vision_patches"));
  ET_ASSIGN_OR_RETURN(
      published_patches,
      detail::read_required_positive_int(module, "get_max_vision_patches"));
  if (published_patches < 4 ||
      published_patches > std::numeric_limits<int32_t>::max()) {
    return Error::InvalidProgram;
  }
  // Export bounds patches in complete 2x2 downsampling groups, but publishes
  // the original limit (which need not be divisible by four).
  const int64_t patches = (published_patches / 4) * 4;
  ET_ASSIGN_OR_RETURN(meta, module.method_meta("vision_encoder"));
  if (meta.num_inputs() != 9 || meta.num_outputs() != 1) {
    return Error::InvalidProgram;
  }
  ET_CHECK_OK_OR_RETURN_ERROR(check_mlx_backends(meta));
  // Host preprocessing's nine-input ABI; inspect it without loading vision.
  for (size_t i = 0; i < 9; ++i) {
    ET_ASSIGN_OR_RETURN(input, meta.input_tensor_meta(i));
    bool valid = false;
    if (i == 0) {
      valid = tensor_abi(input, aten::ScalarType::Float, {1, patches, 1176});
    } else if (i == 1) {
      // Positional interpolation is always BF16, even for an FP16 decoder.
      valid = tensor_abi(input, aten::ScalarType::BFloat16, {1, patches, 1536});
    } else if (i == 2 || i == 3) {
      valid = tensor_abi(input, aten::ScalarType::Float, {patches, 48});
    } else if (i == 6 || i == 7) {
      valid =
          tensor_abi(input, aten::ScalarType::Bool, {1, 1, patches, patches});
    } else {
      valid = tensor_abi(input, aten::ScalarType::Long, {patches});
    }
    if (!valid) {
      return Error::InvalidProgram;
    }
  }
  ET_ASSIGN_OR_RETURN(output, meta.output_tensor_meta(0));
  if (!tensor_abi(output, dtype, {1, patches / 4, hidden})) {
    return Error::InvalidProgram;
  }
  return std::min<int64_t>(patches, max_patches) / 4;
}

uint64_t prediction_seed(uint64_t seed, batching::Position position) {
  // SplitMix64: absolute prediction positions, not forward or draw counts.
  uint64_t value =
      seed + 0x9e3779b97f4a7c15ULL * (static_cast<uint64_t>(position) + 1);
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  value ^= value >> 31;
  // Sampler's xorshift generator has an absorbing state at zero.
  return value == 0 ? 0x9e3779b97f4a7c15ULL : value;
}

std::mt19937 draw_rng(uint64_t seed, batching::Position position, int purpose) {
  const auto value =
      prediction_seed(seed ^ prediction_seed(purpose, 0), position);
  std::seed_seq words{
      static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32)};
  return std::mt19937(words);
}
} // namespace

struct MuseGlimmerMLXExecutor::DFlash {
  std::unique_ptr<cache::InstallGuard> guard;
  cache::BatchControl* ctl = nullptr;
  int block = 0, mask = 0, hidden = 0, context = 0, max_owners = 0;
  size_t row_bytes() const {
    return static_cast<size_t>(hidden) * 2;
  }
};

struct MuseGlimmerMLXExecutor::EngineEmbeddings {
  std::shared_ptr<const MuseGlimmerPreparationSpec> spec;
  MuseGlimmerMaterializer materializer;
  std::string pos_embed_path;
  std::mutex vision_mutex;
  std::unique_ptr<MuseGlimmerVisionRuntime> vision;

  EngineEmbeddings(
      std::shared_ptr<const MuseGlimmerPreparationSpec> spec,
      std::string path)
      : spec(std::move(spec)),
        materializer(this->spec),
        pos_embed_path(std::move(path)) {}

  Error initialize(Module& module) {
    ET_CHECK_OK_OR_RETURN_ERROR(module.load_method("embed_text"));
    if (spec->has_vision) {
      ET_CHECK_OK_OR_RETURN_ERROR(module.load_method("vision_encoder"));
      MuseGlimmerVisionRuntimeConfig config;
      config.module = &module;
      config.execution_mutex = &vision_mutex;
      config.pos_embed_path = pos_embed_path;
      config.activation_dtype = spec->activation_dtype;
      config.expected_hidden_dim = spec->hidden_dim;
      config.max_image_tokens = spec->max_soft_tokens;
      config.max_encoded_bytes = spec->image_limits.max_encoded_bytes;
      config.max_image_dimension = spec->image_limits.max_image_dimension;
      config.max_image_pixels = spec->image_limits.max_image_pixels;
      vision = std::make_unique<MuseGlimmerVisionRuntime>(std::move(config));
    }
    return Error::Ok;
  }

  runtime::Result<TensorPtr> materialize(
      Module& module,
      const std::vector<batching::Input>& inputs) {
    return materializer.materialize(
        inputs,
        [&](const MuseGlimmerRGBImage& image)
            -> runtime::Result<PreparedMuseGlimmerImage> {
          return vision->prepare_decoded_image(
              image.rgb.data(), image.width, image.height);
        },
        [&](const std::vector<int64_t>& tokens) -> runtime::Result<TensorPtr> {
          auto input = make_tensor_ptr(
              {1, static_cast<aten::SizesType>(tokens.size())}, tokens);
          ET_ASSIGN_OR_RETURN(
              outputs,
              module.execute(
                  "embed_text", std::vector<runtime::EValue>{input}));
          if (outputs.size() != 1 || !outputs[0].isTensor() ||
              !dense_tensor(outputs[0].toTensor())) {
            return Error::InvalidProgram;
          }
          // The materializer copies these rows before another method can reuse
          // the module's shared output arena.
          return make_tensor_ptr(outputs[0].toTensor());
        });
  }
};

MuseGlimmerMLXExecutor::MuseGlimmerMLXExecutor(
    std::unique_ptr<Module> module,
    std::shared_ptr<cache::Cache> cache,
    std::shared_ptr<const MuseGlimmerPreparationSpec> spec,
    std::string pos_embed_path,
    int max_sessions)
    : install_guard_(cache),
      module_(std::move(module)),
      ctl_(cache->as<cache::BatchControl>()),
      spec_(std::move(spec)),
      embeddings_(
          std::make_unique<EngineEmbeddings>(spec_, std::move(pos_embed_path))),
      max_sessions_(max_sessions) {}

MuseGlimmerMLXExecutor::~MuseGlimmerMLXExecutor() = default;

runtime::Result<MuseGlimmerBackend> MuseGlimmerMLXExecutor::create(
    const MuseGlimmerBackendConfig& config) {
#if ET_HAS_EXCEPTIONS
  try {
#endif
    if (config.backend != "mlx" || config.model_path.empty() ||
        config.max_sessions <= 0 || config.max_session_tokens < 0 ||
        config.max_session_tokens == 1 || config.max_vision_patches < 4 ||
        config.image_limits.max_encoded_bytes == 0 ||
        config.image_limits.max_image_pixels <= 0 ||
        config.image_limits.max_image_dimension <= 0 ||
        config.dflash_n_draft < 0 ||
        (config.dflash_n_draft != 0 && !config.use_dflash)) {
      return Error::InvalidArgument;
    }
    std::vector<std::string> data_files;
    if (!config.data_path.empty()) {
      data_files.push_back(config.data_path);
    }
    auto module = std::make_unique<Module>(
        config.model_path,
        data_files,
        Module::LoadMode::MmapUseMlockIgnoreErrors,
        nullptr,
        nullptr,
        nullptr,
        /*share_memory_arenas=*/true);
    ET_CHECK_OK_OR_RETURN_ERROR(module->load());
    ET_ASSIGN_OR_RETURN(names, module->method_names());
    if (!names.count(kDecoder) || !names.count("embed_text") ||
        (config.use_dflash && !names.count("draft_forward")) ||
        names.count("get_mutable_buffer_metadata")) {
      return Error::InvalidProgram;
    }
    for (const char* name :
         {kActivationDtype,
          kLogitsToKeepMode,
          kMaxContextLen,
          kMaxSeqLen,
          "get_max_prefill_chunk",
          kVocabSize,
          kNumCaches,
          kKVHeads,
          kHeadDims,
          kWindows}) {
      ET_CHECK_OK_OR_RETURN_ERROR(check_constant_method(*module, name));
    }
    ET_ASSIGN_OR_RETURN(dtype, read_activation_dtype(*module));
    ET_ASSIGN_OR_RETURN(mode, read_logits_to_keep_mode(*module));
    ET_ASSIGN_OR_RETURN(context, read_max_context_length(*module));
    ET_ASSIGN_OR_RETURN(width, read_max_seq_len(*module));
    ET_ASSIGN_OR_RETURN(
        prefill,
        detail::read_required_positive_int(*module, "get_max_prefill_chunk"));
    ET_ASSIGN_OR_RETURN(vocab, read_vocab_size(*module));
    ET_ASSIGN_OR_RETURN(geometry, read_cache_geometry(*module));
    if ((dtype != aten::ScalarType::Half &&
         dtype != aten::ScalarType::BFloat16) ||
        mode != LogitsToKeepMode::Selected || context <= 1 ||
        context > std::numeric_limits<int32_t>::max() ||
        vocab > std::numeric_limits<int32_t>::max() || vocab <= 200092 ||
        config.bos_id >= static_cast<uint64_t>(vocab) || width != prefill ||
        width > context) {
      return Error::InvalidProgram;
    }
    if (config.max_session_tokens > context) {
      ET_LOG(
          Error,
          "Muse Glimmer requested context %d exceeds exported context %lld",
          config.max_session_tokens,
          static_cast<long long>(context));
      return Error::InvalidProgram;
    }
    ET_ASSIGN_OR_RETURN(meta, module->method_meta(kDecoder));
    const bool target_taps = names.count("draft_forward") != 0;
    if (meta.num_inputs() != 3 || meta.num_outputs() != (target_taps ? 2 : 1)) {
      return Error::InvalidProgram;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(check_mlx_backends(meta));
    ET_ASSIGN_OR_RETURN(input, meta.input_tensor_meta(0));
    ET_ASSIGN_OR_RETURN(positions, meta.input_tensor_meta(1));
    ET_ASSIGN_OR_RETURN(selector, meta.input_tensor_meta(2));
    ET_ASSIGN_OR_RETURN(logits, meta.output_tensor_meta(0));
    if (input.sizes().size() != 3 || selector.sizes().size() != 1) {
      return Error::InvalidProgram;
    }
    const int32_t hidden = input.sizes()[2];
    const int32_t rows = selector.sizes()[0];
    if (!tensor_abi(input, dtype, {1, width, hidden}) ||
        !tensor_abi(positions, aten::ScalarType::Long, {width}) ||
        rows < width || !tensor_abi(selector, aten::ScalarType::Long, {rows}) ||
        !tensor_abi(logits, aten::ScalarType::Float, {1, rows, vocab})) {
      return Error::InvalidProgram;
    }
    ET_ASSIGN_OR_RETURN(embed_meta, module->method_meta("embed_text"));
    if (embed_meta.num_inputs() != 1 || embed_meta.num_outputs() != 1) {
      return Error::InvalidProgram;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(check_mlx_backends(embed_meta));
    ET_ASSIGN_OR_RETURN(embed_input, embed_meta.input_tensor_meta(0));
    ET_ASSIGN_OR_RETURN(embed_output, embed_meta.output_tensor_meta(0));
    if (embed_input.sizes().size() != 2) {
      return Error::InvalidProgram;
    }
    const int32_t embed_capacity = embed_input.sizes()[1];
    if (embed_capacity < width ||
        !tensor_abi(embed_input, aten::ScalarType::Long, {1, embed_capacity}) ||
        !tensor_abi(embed_output, dtype, {1, embed_capacity, hidden})) {
      return Error::InvalidProgram;
    }
    const bool has_vision = names.count("vision_encoder") != 0;
    int64_t max_soft_tokens = 0;
    if (has_vision) {
      ET_ASSIGN_OR_RETURN(
          soft_tokens,
          check_vision_abi(*module, hidden, dtype, config.max_vision_patches));
      max_soft_tokens = soft_tokens;
    }
    const int capacity = config.max_session_tokens ? config.max_session_tokens
                                                   : static_cast<int>(context);
    cache::CacheConfig cache_config{};
    cache_config.capacity = capacity;
    cache_config.kv_dtype = static_cast<int>(dtype);
    cache_config.max_write = static_cast<int>(width);
    std::unique_ptr<DFlash> dflash;
    if (config.use_dflash) {
      for (const auto* name :
           {"get_block_size",
            "get_mask_token_id",
            "get_n_target_layers",
            "get_draft_n_caches",
            "get_draft_kv_heads",
            "get_draft_head_dims",
            "get_draft_windows"}) {
        ET_CHECK_OK_OR_RETURN_ERROR(check_constant_method(*module, name));
      }
      ET_ASSIGN_OR_RETURN(
          max_block,
          detail::read_required_positive_int(*module, "get_block_size"));
      const int64_t block = config.dflash_n_draft
          ? static_cast<int64_t>(config.dflash_n_draft) + 1
          : max_block;
      if (block < 2 || block > max_block) {
        return Error::InvalidArgument;
      }
      ET_ASSIGN_OR_RETURN(
          mask, detail::read_int_method(*module, "get_mask_token_id"));
      ET_ASSIGN_OR_RETURN(
          taps,
          detail::read_required_positive_int(*module, "get_n_target_layers"));
      ET_ASSIGN_OR_RETURN(
          nc,
          detail::read_required_positive_int(*module, "get_draft_n_caches"));
      ET_ASSIGN_OR_RETURN(
          heads, detail::read_int_tensor_method(*module, "get_draft_kv_heads"));
      ET_ASSIGN_OR_RETURN(
          dims, detail::read_int_tensor_method(*module, "get_draft_head_dims"));
      ET_ASSIGN_OR_RETURN(
          windows,
          detail::read_int_tensor_method(*module, "get_draft_windows"));
      ET_ASSIGN_OR_RETURN(dm, module->method_meta("draft_forward"));
      ET_CHECK_OK_OR_RETURN_ERROR(check_mlx_backends(dm));
      if (dm.num_inputs() != 4 || dm.num_outputs() != 1 || block > context ||
          !mask || *mask < 0 || *mask >= vocab ||
          taps > geometry.layers.size() || nc != heads.size() ||
          nc != dims.size() || nc != windows.size()) {
        return Error::InvalidProgram;
      }
      ET_ASSIGN_OR_RETURN(dt, dm.input_tensor_meta(0));
      ET_ASSIGN_OR_RETURN(dh, dm.input_tensor_meta(1));
      ET_ASSIGN_OR_RETURN(dp, dm.input_tensor_meta(2));
      ET_ASSIGN_OR_RETURN(order, dm.input_tensor_meta(3));
      ET_ASSIGN_OR_RETURN(dl, dm.output_tensor_meta(0));
      ET_ASSIGN_OR_RETURN(th, meta.output_tensor_meta(1));
      if (dt.sizes().size() != 2 || dh.sizes().size() != 3 ||
          th.sizes().size() != 3) {
        return Error::InvalidProgram;
      }
      const int64_t owners = dt.sizes()[0];
      const int64_t q = owners * dt.sizes()[1], c = dh.sizes()[1],
                    h = th.sizes()[2];
      if (owners <= 0 || q < block || c <= 0 || h != taps * hidden ||
          c + q > std::numeric_limits<int>::max() ||
          width + block > std::numeric_limits<int>::max() ||
          !tensor_abi(dt, aten::ScalarType::Long, {owners, max_block}) ||
          !tensor_abi(dh, dtype, {1, c, h}) ||
          !tensor_abi(dp, aten::ScalarType::Long, {c + q}) ||
          !tensor_abi(order, aten::ScalarType::Long, {c + q}) ||
          !tensor_abi(dl, aten::ScalarType::Float, {1, q, vocab}) ||
          !tensor_abi(th, dtype, {1, width, h})) {
        return Error::InvalidProgram;
      }
      cache::CacheGeometry dg;
      dg.dflash_block_size = static_cast<int>(block);
      auto dc = cache_config;
      dc.max_write = static_cast<int>(c);
      for (size_t i = 0; i < heads.size(); ++i) {
        if (windows[i] < 0 ||
            windows[i] > std::numeric_limits<int>::max() - c + 1) {
          return Error::InvalidProgram;
        }
        using Kind = cache::LayerPolicy::Kind;
        const auto kind = windows[i] ? Kind::Ring : Kind::Flat;
        dg.layers.push_back({{kind, windows[i]}, heads[i], dims[i]});
      }
      if (!cache::valid(dg, dc)) {
        return Error::InvalidProgram;
      }
      ET_ASSIGN_OR_RETURN(
          dcache,
          cache::CacheFactory::global().build(
              kBackend, cache::kind::kBatched, dg, dc));
      auto* dctl = dcache->as<cache::BatchControl>();
      if (!dctl || dctl->capacity() < capacity ||
          (dctl->max_seqs() && config.max_sessions > *dctl->max_seqs())) {
        return Error::InvalidProgram;
      }
      dflash = std::make_unique<DFlash>();
      dflash->guard = std::make_unique<cache::InstallGuard>(dcache);
      dflash->ctl = dctl;
      dflash->block = static_cast<int>(block);
      dflash->mask = static_cast<int>(*mask);
      dflash->hidden = static_cast<int>(h);
      dflash->context = static_cast<int>(c);
      dflash->max_owners = static_cast<int>(owners);
      cache_config.max_write = static_cast<int>(width + block);
    }
    for (const auto& layer : geometry.layers) {
      if (layer.policy.window >
          std::numeric_limits<int>::max() - *cache_config.max_write + 1) {
        return Error::InvalidProgram;
      }
    }
    if (!cache::valid(geometry, cache_config)) {
      return Error::InvalidProgram;
    }
    ET_ASSIGN_OR_RETURN(
        cache,
        cache::CacheFactory::global().build(
            kBackend, cache::kind::kBatched, geometry, cache_config));
    auto* ctl = cache ? cache->as<cache::BatchControl>() : nullptr;
    if (!ctl) {
      return Error::InvalidType;
    }
    const auto seq_limit = ctl->max_seqs();
    if (ctl->capacity() < cache_config.capacity ||
        (seq_limit && config.max_sessions > *seq_limit)) {
      return Error::InvalidArgument;
    }
    auto spec = std::make_shared<const MuseGlimmerPreparationSpec>(
        MuseGlimmerPreparationSpec{
            dtype,
            hidden,
            capacity,
            static_cast<int32_t>(width),
            static_cast<int32_t>(vocab),
            has_vision,
            max_soft_tokens,
            config.image_limits,
            config.bos_id});
    auto path = config.pos_embed_path;
    if (path.empty()) {
      const auto slash = config.model_path.find_last_of('/');
      path = (slash == std::string::npos ? "."
                                         : config.model_path.substr(0, slash)) +
          "/pos_embed.bin";
    }
    auto executor =
        std::unique_ptr<MuseGlimmerMLXExecutor>(new MuseGlimmerMLXExecutor(
            std::move(module),
            std::move(cache),
            spec,
            std::move(path),
            config.max_sessions));
    executor->dflash_ = std::move(dflash);
    executor->target_taps_ = target_taps;
    return MuseGlimmerBackend{std::move(executor), spec};
#if ET_HAS_EXCEPTIONS
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer executor creation failed: %s", error.what());
    return Error::Internal;
  }
#endif
}

bool MuseGlimmerMLXExecutor::initialize() {
  if (initialization_attempted_) {
    return initialized_;
  }
  initialization_attempted_ = true;
#if ET_HAS_EXCEPTIONS
  try {
#endif
    auto error = batching::load_method_with_cache(
        *module_, kDecoder, kBackend, install_guard_);
    if (error == Error::Ok && dflash_) {
      error = batching::load_method_with_cache(
          *module_, "draft_forward", kBackend, *dflash_->guard);
    }
    if (error == Error::Ok) {
      error = embeddings_->initialize(*module_);
    }
    if (error != Error::Ok) {
      ET_LOG(
          Error,
          "Muse Glimmer initialization failed: %d",
          static_cast<int>(error));
      return false;
    }
    initialized_ = true;
    return true;
#if ET_HAS_EXCEPTIONS
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer initialization failed: %s", error.what());
    return false;
  }
#endif
}

size_t MuseGlimmerMLXExecutor::preferred_batch_tokens() const {
  return static_cast<size_t>(spec_->max_forward_tokens);
}

std::optional<batching::SessionId> MuseGlimmerMLXExecutor::publish_session(
    int32_t seq_id,
    batching::Position position,
    const SessionState* source) {
  return batching::publish_sequence(
      *ctl_,
      seq_id,
      position,
      next_session_,
      [&](batching::SessionId session, int32_t sequence) {
        SessionState state =
            source ? *source : SessionState{sequence, std::nullopt};
        state.seq_id = sequence;
        state.sampling.reset();
        if (!dflash_) {
          return sessions_.emplace(session, std::move(state)).second;
        }
        const int keep = source
            ? std::min(position, dflash_->ctl->pos(source->draft_id))
            : 0;
        auto draft_id = keep ? dflash_->ctl->seq_clone(source->draft_id, keep)
                             : dflash_->ctl->seq_new();
        if (!draft_id) {
          return false;
        }
        batching::SequenceGuard guard(*dflash_->ctl, *draft_id);
        const size_t bytes =
            static_cast<size_t>(position - keep) * dflash_->row_bytes();
        if (dflash_->ctl->pos(*draft_id) != keep ||
            bytes > state.hidden.size()) {
          return false;
        }
        state.draft_id = *draft_id;
        state.hidden.resize(bytes);
        if (!sessions_.emplace(session, std::move(state)).second) {
          return false;
        }
        guard.release();
        return true;
      });
}

std::optional<batching::SessionId> MuseGlimmerMLXExecutor::open_session() {
  if (!initialized_ || next_session_ == 0 ||
      sessions_.size() >= static_cast<size_t>(max_sessions_)) {
    return std::nullopt;
  }
#if ET_HAS_EXCEPTIONS
  try {
#endif
    auto seq_id = ctl_->seq_new();
    return seq_id ? publish_session(*seq_id, 0) : std::nullopt;
#if ET_HAS_EXCEPTIONS
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer session creation failed: %s", error.what());
    return std::nullopt;
  }
#endif
}

void MuseGlimmerMLXExecutor::close_session(batching::SessionId session) {
  const auto it = sessions_.find(session);
  if (it != sessions_.end()) {
    ctl_->seq_rm(it->second.seq_id);
    if (dflash_) {
      dflash_->ctl->seq_rm(it->second.draft_id);
    }
    sessions_.erase(it);
  }
}

std::optional<batching::SessionId> MuseGlimmerMLXExecutor::clone(
    batching::SessionId source,
    batching::Position upto) {
  const auto it = sessions_.find(source);
  if (!initialized_ || it == sessions_.end() || upto < 0 ||
      upto > spec_->max_context_length || next_session_ == 0 ||
      sessions_.size() >= static_cast<size_t>(max_sessions_)) {
    return std::nullopt;
  }
#if ET_HAS_EXCEPTIONS
  try {
#endif
    if (upto > ctl_->pos(it->second.seq_id)) {
      return std::nullopt;
    }
    auto seq_id = ctl_->seq_clone(it->second.seq_id, upto);
    return seq_id ? publish_session(*seq_id, upto, &it->second) : std::nullopt;
#if ET_HAS_EXCEPTIONS
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer session clone failed: %s", error.what());
    return std::nullopt;
  }
#endif
}

void MuseGlimmerMLXExecutor::set_sampling(
    batching::SessionId session,
    const batching::SamplingParams& params,
    std::optional<uint64_t> seed) {
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
      state.seed = (static_cast<uint64_t>(device()) << 32) ^ device();
    }
    state.sampling = params;
#if ET_HAS_EXCEPTIONS
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer sampling setup failed: %s", error.what());
  }
#endif
}

bool MuseGlimmerMLXExecutor::accepts(
    const batching::PreparedInput& input) const {
  return embeddings_->materializer.accepts(input);
}

runtime::Result<MuseGlimmerMLXExecutor::Rewinds>
MuseGlimmerMLXExecutor::validate_batch(
    const batching::BatchInput& batch) const {
  Rewinds rewinds;
  std::unordered_map<int32_t, int> cursors;
  for (const auto& input : batch.inputs) {
    const auto session = sessions_.find(input.sid);
    if (session == sessions_.end() ||
        (input.produce_output && !session->second.sampling)) {
      return Error::InvalidArgument;
    }
    size_t size = 0;
    const auto* raw = std::get_if<batching::TokenInputPtr>(&input.payload);
    if (raw) {
      if (!*raw) {
        return Error::InvalidArgument;
      }
      size = (*raw)->size();
    } else {
      const auto& prepared =
          std::get<batching::PreparedInputPtr>(input.payload);
      if (!prepared || !accepts(*prepared)) {
        return Error::InvalidArgument;
      }
      size = prepared->size();
    }
    if (input.size == 0 || input.offset > size ||
        input.size > size - input.offset || input.position < 0 ||
        input.position > spec_->max_context_length ||
        input.offset >
            static_cast<size_t>(spec_->max_context_length - input.position)) {
      return Error::InvalidArgument;
    }
    const int start = input.position + static_cast<int>(input.offset);
    if (input.size > static_cast<size_t>(spec_->max_context_length - start)) {
      return Error::OutOfResources;
    }
    if (raw) {
      for (size_t i = input.offset; i < input.offset + input.size; ++i) {
        if ((**raw)[i] >= static_cast<uint64_t>(spec_->vocab_size)) {
          return Error::InvalidArgument;
        }
      }
    }
    const int32_t seq_id = session->second.seq_id;
    const auto [cursor, first] = cursors.try_emplace(seq_id, ctl_->pos(seq_id));
    if (cursor->second < 0 || start > cursor->second ||
        (!first && start != cursor->second)) {
      return Error::InvalidArgument;
    }
    if (start < cursor->second) {
      // Rewind preserves sequence identity even for an empty prefix.
      // Evicted history may still make rewind fail when applied below.
      rewinds.emplace_back(input.sid, start);
    }
    cursor->second = start + static_cast<int>(input.size);
  }
  return rewinds;
}

std::optional<batching::Token> MuseGlimmerMLXExecutor::sample_row(
    aten::Tensor& logits,
    int row,
    batching::SessionId session,
    batching::Position prediction_position,
    std::vector<float>* probabilities,
    bool draft) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end() || !it->second.sampling || row < 0 ||
      row >= logits.size(1)) {
    return std::nullopt;
  }
  const auto& params = *it->second.sampling;
  if (dflash_) {
    const auto* values = logits.const_data_ptr<float>() +
        static_cast<size_t>(row) * spec_->vocab_size;
    if (!std::all_of(values, values + spec_->vocab_size, [](float v) {
          return std::isfinite(v);
        })) {
      return std::nullopt;
    }
    if (draft) {
      return muse_glimmer::argmax_index(values, spec_->vocab_size);
    }
    std::vector<float> scratch;
    auto& p = probabilities ? *probabilities : scratch;
    if (params.temperature == 0) {
      const auto token = muse_glimmer::argmax_index(values, spec_->vocab_size);
      p.assign(spec_->vocab_size, 0.0f);
      p[token] = 1.0f;
      return token;
    }
    p = muse_glimmer::sampling_probabilities(
        values,
        spec_->vocab_size,
        params.temperature,
        params.top_k,
        params.top_p);
    auto rng = draw_rng(it->second.seed, prediction_position, 0);
    return muse_glimmer::categorical_sample(rng, p.data(), p.size());
  }
  Sampler sampler(
      spec_->vocab_size,
      params.temperature,
      params.top_p,
      prediction_seed(it->second.seed, prediction_position));
  sampler.set_topk(params.top_k);
  return batching::sample_from_row(logits, row, sampler);
}

bool MuseGlimmerMLXExecutor::rewind(SessionState& state, int position) {
  if (!ctl_->rewind(state.seq_id, position)) {
    return false;
  }
  if (dflash_) {
    const int frontier = dflash_->ctl->pos(state.draft_id);
    const size_t keep = static_cast<size_t>(std::max(0, position - frontier)) *
        dflash_->row_bytes();
    if (keep > state.hidden.size() ||
        (position < frontier &&
         !dflash_->ctl->rewind(state.draft_id, position))) {
      return false;
    }
    state.hidden.resize(keep);
  }
  return true;
}

bool MuseGlimmerMLXExecutor::drain(batching::SessionId session) {
  auto& state = sessions_.at(session);
  while (state.hidden.size() / dflash_->row_bytes() >
         static_cast<size_t>(dflash_->context)) {
    const int count = std::min(
        dflash_->context,
        spec_->max_context_length - dflash_->block -
            dflash_->ctl->pos(state.draft_id));
    if (count <= 0) {
      break;
    }
    if (!draft(
            {session},
            {static_cast<batching::Token>(dflash_->mask)},
            count,
            nullptr)) {
      return false;
    }
  }
  return true;
}

bool MuseGlimmerMLXExecutor::draft(
    const std::vector<batching::SessionId>& owners,
    const std::vector<batching::Token>& anchors,
    int context_limit,
    batching::BatchInput* verification) {
  const int b = dflash_->block;
  if (owners.empty() || owners.size() != anchors.size() ||
      owners.size() > static_cast<size_t>(dflash_->max_owners) ||
      context_limit < 0 || context_limit > dflash_->context) {
    return false;
  }
  const int q = static_cast<int>(owners.size()) * b;
  std::vector<int64_t> tokens(q, dflash_->mask), positions, order;
  std::vector<int32_t> ids;
  std::vector<int> counts, ends;
  std::vector<uint8_t> hidden;
  for (size_t i = 0; i < owners.size(); ++i) {
    auto& state = sessions_.at(owners[i]);
    const int start = dflash_->ctl->pos(state.draft_id);
    const int count = std::min<size_t>(
        context_limit, state.hidden.size() / dflash_->row_bytes());
    if (start < 0 || start > spec_->max_context_length - b - count ||
        (verification && count * dflash_->row_bytes() != state.hidden.size())) {
      return false;
    }
    tokens[i * b] = anchors[i];
    counts.push_back(count);
    ends.push_back(start + count);
    context_limit -= count;
    hidden.insert(
        hidden.end(),
        state.hidden.begin(),
        state.hidden.begin() + count * dflash_->row_bytes());
    for (int j = 0; j < count; ++j) {
      positions.push_back(start + j);
    }
    ids.insert(ids.end(), count + b, state.draft_id);
  }
  const int c = static_cast<int>(positions.size());
  int offset = 0;
  for (size_t i = 0; i < owners.size(); ++i) {
    for (int j = 0; j < counts[i]; ++j) {
      order.push_back(offset++);
    }
    for (int j = 0; j < b; ++j) {
      positions.push_back(ends[i] + j);
      order.push_back(c + i * b + j);
    }
  }
  auto token_tensor =
      make_tensor_ptr({static_cast<int>(owners.size()), b}, std::move(tokens));
  auto hidden_tensor = make_tensor_ptr(
      {1, c, dflash_->hidden}, std::move(hidden), spec_->activation_dtype);
  auto position_tensor = make_tensor_ptr({c + q}, std::move(positions));
  auto order_tensor = make_tensor_ptr({c + q}, std::move(order));
  if (!dflash_->ctl->declare_step(ids)) {
    return false;
  }
  auto result = module_->execute(
      "draft_forward",
      {token_tensor, hidden_tensor, position_tensor, order_tensor});
  if (!result.ok() || result->size() != 1 || !result->at(0).isTensor()) {
    return false;
  }
  auto logits = result->at(0).toTensor();
  if (logits.dim() != 3 || logits.size(0) != 1 || logits.size(1) != q ||
      logits.size(2) != spec_->vocab_size ||
      logits.scalar_type() != aten::ScalarType::Float ||
      !dense_tensor(logits)) {
    return false;
  }
  for (size_t i = 0; i < owners.size(); ++i) {
    auto& state = sessions_.at(owners[i]);
    if (dflash_->ctl->pos(state.draft_id) != ends[i]) {
      return false;
    }
    state.hidden.erase(
        state.hidden.begin(),
        state.hidden.begin() + counts[i] * dflash_->row_bytes());
    if (verification) {
      auto block =
          std::make_shared<std::vector<batching::Token>>(b, anchors[i]);
      for (int j = 1; j < b; ++j) {
        auto token = sample_row(
            logits, i * b + j, owners[i], ends[i] + j, nullptr, true);
        if (!token) {
          return false;
        }
        (*block)[j] = *token;
      }
      for (int j = 0; j < b; ++j) {
        verification->inputs.push_back(
            {owners[i],
             true,
             static_cast<size_t>(j),
             1,
             batching::TokenInputPtr(block),
             ends[i]});
      }
    }
  }
  return true;
}

bool MuseGlimmerMLXExecutor::speculate(
    const batching::BatchInput& batch,
    batching::BatchOutput& out) {
  const int b = dflash_->block;
  size_t cursor = 0;
  while (cursor < batch.inputs.size()) {
    std::vector<batching::SessionId> owners;
    std::vector<batching::Token> anchors;
    std::vector<size_t> indices;
    size_t context = 0;
    while (cursor < batch.inputs.size()) {
      const auto& input = batch.inputs[cursor];
      auto& state = sessions_.at(input.sid);
      const int end = input.position + input.offset + input.size;
      if (!out.outputs[cursor] || end != ctl_->pos(state.seq_id) ||
          end > spec_->max_context_length - b) {
        ++cursor;
        continue;
      }
      if (!drain(input.sid)) {
        return false;
      }
      const size_t count = state.hidden.size() / dflash_->row_bytes();
      if (owners.size() == static_cast<size_t>(dflash_->max_owners) ||
          context + count > static_cast<size_t>(dflash_->context)) {
        break;
      }
      context += count;
      owners.push_back(input.sid);
      anchors.push_back(out.outputs[cursor]->tokens.front());
      indices.push_back(cursor++);
    }
    if (owners.empty()) {
      continue;
    }
    batching::BatchInput verification;
    if (!draft(owners, anchors, dflash_->context, &verification)) {
      return false;
    }
    Probabilities p(verification.inputs.size());
    batching::BatchOutput predictions;
    if (!execute_impl(verification, predictions, &p)) {
      return false;
    }
    for (size_t i = 0; i < owners.size(); ++i) {
      auto& state = sessions_.at(owners[i]);
      auto& output = out.outputs[indices[i]]->tokens;
      const auto& input = verification.inputs[i * b];
      const auto& block = *std::get<batching::TokenInputPtr>(input.payload);
      for (int j = 0; j < b; ++j) {
        auto& target = p[i * b + j];
        const int position = input.position + j + 1;
        if (j + 1 < b) {
          const auto proposal = block[j + 1];
          auto rng = draw_rng(state.seed, position, 2);
          if (muse_glimmer::accept_with_probability(rng, target[proposal])) {
            output.push_back(proposal);
            continue;
          }
        }
        auto rng = draw_rng(state.seed, position, j + 1 == b ? 4 : 3);
        batching::Token correction;
        if (state.sampling->temperature == 0) {
          correction = muse_glimmer::argmax_index(target.data(), target.size());
        } else if (j + 1 == b) {
          correction = muse_glimmer::categorical_sample(
              rng, target.data(), target.size());
        } else {
          correction = muse_glimmer::sample_excluding_token_in_place(
              rng, target, block[j + 1]);
        }
        output.push_back(correction);
        // Leave the returned correction/bonus uncommitted.
        if (!rewind(state, position)) {
          return false;
        }
        break;
      }
    }
  }
  return true;
}

bool MuseGlimmerMLXExecutor::execute(
    const batching::BatchInput& batch,
    batching::BatchOutput& out) {
  return execute_impl(batch, out, nullptr);
}

bool MuseGlimmerMLXExecutor::execute_impl(
    const batching::BatchInput& batch,
    batching::BatchOutput& out,
    Probabilities* probabilities) {
#if ET_HAS_EXCEPTIONS
  try {
#endif
    out.outputs.clear();
    out.outputs.resize(batch.inputs.size());
    if (!initialized_) {
      return false;
    }
    auto rewinds = validate_batch(batch);
    if (!rewinds.ok()) {
      ET_LOG(Error, "Muse Glimmer rejected an invalid batch");
      return false;
    }
    for (const auto& [session, position] : *rewinds) {
      if (!rewind(sessions_.at(session), position)) {
        return false;
      }
    }
    batching::BatchOutput pending;
    pending.outputs.resize(batch.inputs.size());
    size_t input_index = 0;
    size_t consumed = 0;
    while (input_index < batch.inputs.size()) {
      std::vector<batching::Input> slices;
      std::vector<int64_t> positions;
      std::vector<int32_t> seq_ids;
      std::vector<int64_t> selectors;
      std::vector<size_t> selected_inputs;
      const size_t width = preferred_batch_tokens();
      while (input_index < batch.inputs.size() && positions.size() < width) {
        const auto& input = batch.inputs[input_index];
        const size_t count =
            std::min(input.size - consumed, width - positions.size());
        auto slice = input;
        slice.offset += consumed;
        slice.size = count;
        slice.produce_output =
            input.produce_output && consumed + count == input.size;
        // position remains the backing's base, even on later physical slices.
        const int64_t start = static_cast<int64_t>(slice.position) +
            static_cast<int64_t>(slice.offset);
        for (size_t i = 0; i < count; ++i) {
          positions.push_back(start + static_cast<int64_t>(i));
        }
        seq_ids.insert(seq_ids.end(), count, sessions_.at(input.sid).seq_id);
        if (slice.produce_output) {
          selectors.push_back(static_cast<int64_t>(positions.size()) - 1);
          selected_inputs.push_back(input_index);
        }
        slices.push_back(std::move(slice));
        consumed += count;
        if (consumed == input.size) {
          ++input_index;
          consumed = 0;
        }
      }
      const auto n = static_cast<aten::SizesType>(positions.size());
      if (selectors.empty()) {
        selectors.push_back(n - 1);
      }
      const auto rows = static_cast<aten::SizesType>(selectors.size());
      if (dflash_ && !probabilities) {
        for (const auto& slice : slices) {
          if (!drain(slice.sid)) {
            return false;
          }
        }
      }
      auto embeddings = embeddings_->materialize(*module_, slices);
      if (!embeddings.ok() || !*embeddings ||
          (*embeddings)->scalar_type() != spec_->activation_dtype ||
          (*embeddings)->dim() != 3 || (*embeddings)->size(0) != 1 ||
          (*embeddings)->size(1) != n ||
          (*embeddings)->size(2) != spec_->hidden_dim ||
          !dense_tensor(**embeddings)) {
        return false;
      }
      auto position_tensor = make_tensor_ptr({n}, std::move(positions));
      auto selector_tensor = make_tensor_ptr({rows}, std::move(selectors));
      if (!ctl_->declare_step(seq_ids)) {
        return false;
      }
      auto result = module_->execute(
          kDecoder, {*embeddings, position_tensor, selector_tensor});
      if (!result.ok() || result->size() != (target_taps_ ? 2 : 1) ||
          !result->at(0).isTensor()) {
        return false;
      }
      auto logits = result->at(0).toTensor();
      if (logits.scalar_type() != aten::ScalarType::Float ||
          logits.dim() != 3 || logits.size(0) != 1 || logits.size(1) != rows ||
          logits.size(2) != spec_->vocab_size || !dense_tensor(logits)) {
        return false;
      }
      if (dflash_) {
        if (!result->at(1).isTensor()) {
          return false;
        }
        const auto& hidden = result->at(1).toTensor();
        if (hidden.dim() != 3 || hidden.size(0) != 1 || hidden.size(1) != n ||
            hidden.size(2) != dflash_->hidden ||
            hidden.scalar_type() != spec_->activation_dtype ||
            !dense_tensor(hidden)) {
          return false;
        }
        const auto* data = static_cast<const uint8_t*>(hidden.const_data_ptr());
        for (const auto& slice : slices) {
          auto& state = sessions_.at(slice.sid);
          if (dflash_->ctl->pos(state.draft_id) +
                  state.hidden.size() / dflash_->row_bytes() !=
              static_cast<size_t>(slice.position) + slice.offset) {
            return false;
          }
          const auto bytes = slice.size * dflash_->row_bytes();
          state.hidden.insert(state.hidden.end(), data, data + bytes);
          data += bytes;
        }
      }
      // Consume selected rows once, before the next materialization or forward
      // reuses the module arena. Dummy prefill logits are discarded.
      for (size_t row = 0; row < selected_inputs.size(); ++row) {
        const size_t index = selected_inputs[row];
        const auto& input = batch.inputs[index];
        const auto end = static_cast<batching::Position>(
            static_cast<int64_t>(input.position) + input.offset + input.size);
        const auto token = sample_row(
            logits,
            static_cast<int>(row),
            input.sid,
            end,
            probabilities ? &probabilities->at(index) : nullptr);
        if (!token) {
          return false;
        }
        // This singleton prediction remains uncommitted until runner feedback.
        pending.outputs[index] = batching::Output{input.sid, {*token}};
      }
    }
    if (dflash_ && !probabilities && !speculate(batch, pending)) {
      return false;
    }
    out = std::move(pending);
    return true;
#if ET_HAS_EXCEPTIONS
  } catch (const std::exception& error) {
    ET_LOG(Error, "Muse Glimmer execution failed: %s", error.what());
    return false;
  }
#endif
}

} // namespace executorch::extension::llm
