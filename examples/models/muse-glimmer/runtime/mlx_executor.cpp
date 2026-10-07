/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/mlx_executor.h>

#include <executorch/examples/models/muse-glimmer/runtime/embedding_materializer.h>
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
} // namespace

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

  runtime::Result<TensorPtr> materialize(
      Module& module,
      const std::vector<batching::Input>& inputs) {
    return materializer.materialize(
        inputs,
        [&](const MuseGlimmerRGBImage& image)
            -> runtime::Result<PreparedMuseGlimmerImage> {
          if (!vision) {
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
            vision =
                std::make_unique<MuseGlimmerVisionRuntime>(std::move(config));
          }
          return vision->prepare_decoded_image(
              image.rgb.data(), image.width, image.height);
        },
        [&](const std::vector<int64_t>& tokens) -> runtime::Result<TensorPtr> {
          ET_CHECK_OK_OR_RETURN_ERROR(module.load_method("embed_text"));
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
        config.image_limits.max_image_dimension <= 0) {
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
        names.count("draft_forward") ||
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
        width > context || config.max_session_tokens > context) {
      return Error::InvalidProgram;
    }
    ET_ASSIGN_OR_RETURN(meta, module->method_meta(kDecoder));
    if (meta.num_inputs() != 3 || meta.num_outputs() != 1) {
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
    if (config.max_sessions > std::numeric_limits<int>::max() / capacity) {
      return Error::InvalidArgument;
    }
    cache::CacheConfig cache_config{};
    cache_config.capacity = config.max_sessions * capacity;
    cache_config.kv_dtype = static_cast<int>(dtype);
    cache_config.max_write = static_cast<int>(width);
    for (const auto& layer : geometry.layers) {
      if (layer.policy.window > std::numeric_limits<int>::max() - width + 1) {
        return Error::InvalidProgram;
      }
    }
    if (!cache::valid(geometry, cache_config)) {
      return Error::InvalidProgram;
    }
    ET_ASSIGN_OR_RETURN(
        cache,
        cache::CacheFactory::global().build(
            kBackend, cache::kind::kBatchedSequence, geometry, cache_config));
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
    return MuseGlimmerBackend{
        std::unique_ptr<batching::Executor>(new MuseGlimmerMLXExecutor(
            std::move(module),
            std::move(cache),
            spec,
            std::move(path),
            config.max_sessions)),
        spec};
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
    initialized_ =
        batching::load_method_with_cache(
            *module_, kDecoder, kBackend, install_guard_) == Error::Ok;
    return initialized_;
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
    batching::Position position) {
  return batching::publish_sequence(
      *ctl_,
      seq_id,
      position,
      next_session_,
      [&](batching::SessionId session, int32_t sequence) {
        return sessions_.emplace(session, SessionState{sequence, std::nullopt})
            .second;
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
    return seq_id ? publish_session(*seq_id, upto) : std::nullopt;
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
      // Batched-sequence rewind preserves identity even for an empty prefix.
      // Evicted history may still make rewind fail when applied below.
      rewinds.emplace_back(seq_id, start);
    }
    cursor->second = start + static_cast<int>(input.size);
  }
  return rewinds;
}

std::optional<batching::Token> MuseGlimmerMLXExecutor::sample_row(
    aten::Tensor& logits,
    int row,
    batching::SessionId session,
    batching::Position prediction_position) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end() || !it->second.sampling || row < 0 ||
      row >= logits.size(1)) {
    return std::nullopt;
  }
  const auto& params = *it->second.sampling;
  Sampler sampler(
      spec_->vocab_size,
      params.temperature,
      params.top_p,
      prediction_seed(it->second.seed, prediction_position));
  sampler.set_topk(params.top_k);
  return batching::sample_from_row(logits, row, sampler);
}

bool MuseGlimmerMLXExecutor::execute(
    const batching::BatchInput& batch,
    batching::BatchOutput& out) {
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
    for (const auto& [seq_id, position] : *rewinds) {
      if (!ctl_->rewind(seq_id, position)) {
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
      if (!result.ok() || result->size() != 1 || !result->at(0).isTensor()) {
        return false;
      }
      auto logits = result->at(0).toTensor();
      if (logits.scalar_type() != aten::ScalarType::Float ||
          logits.dim() != 3 || logits.size(0) != 1 || logits.size(1) != rows ||
          logits.size(2) != spec_->vocab_size || !dense_tensor(logits)) {
        return false;
      }
      // Consume selected rows once, before the next materialization or forward
      // reuses the module arena. Dummy prefill logits are discarded.
      for (size_t row = 0; row < selected_inputs.size(); ++row) {
        const size_t index = selected_inputs[row];
        const auto& input = batch.inputs[index];
        const auto end = static_cast<batching::Position>(
            static_cast<int64_t>(input.position) + input.offset + input.size);
        const auto token =
            sample_row(logits, static_cast<int>(row), input.sid, end);
        if (!token) {
          return false;
        }
        // This singleton prediction remains uncommitted until runner feedback.
        pending.outputs[index] = batching::Output{input.sid, {*token}};
      }
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
