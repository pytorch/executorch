/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/batching/cuda_executor.h>

#include <algorithm>
#include <cinttypes>
#include <limits>
#include <new>
#include <random>
#include <utility>

#include <executorch/backends/cuda/batching/step_plan.h>
#include <executorch/backends/cuda/runtime/backend_options.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda::batching {

using ::executorch::extension::make_tensor_ptr;
using ::executorch::extension::Module;
using ::executorch::extension::llm::LogitsToKeepMode;
using ::executorch::extension::llm::Sampler;
using ::executorch::runtime::Error;
using ::executorch::runtime::MethodMeta;
using ::executorch::runtime::Result;
using llm_batching::BatchInput;
using llm_batching::BatchOutput;
using llm_batching::Input;
using llm_batching::Output;
using llm_batching::Position;
using llm_batching::SamplingParams;
using llm_batching::SessionId;
using llm_batching::Token;

namespace metadata = ::executorch::extension::llm;

namespace {

// A prefill forward carries at least this many tokens; one token is decode's.
constexpr int kMinPrefillTokens = 2;

struct SequenceGuard {
  llm_cache::BatchControl& control;
  std::int32_t seq_id;
  bool owned = true;

  ~SequenceGuard() {
    if (owned) {
      control.seq_rm(seq_id);
    }
  }
};

std::uint64_t nondeterministic_seed() {
  std::random_device device;
  return device();
}

bool is_supported_logits_type(::executorch::aten::ScalarType type) {
  using ScalarType = ::executorch::aten::ScalarType;
  return type == ScalarType::Float || type == ScalarType::Half ||
      type == ScalarType::BFloat16;
}

// The widest step `method` takes: the extent its token input was exported
// with, which for a dynamic dimension is its upper bound.
Result<int> step_width(const MethodMeta& meta, const char* method) {
  ET_CHECK_OR_RETURN_ERROR(
      meta.num_inputs() == 3,
      InvalidProgram,
      "CudaExecutor: %s must take tokens, positions and a logits selector",
      method);
  const auto tokens = meta.input_tensor_meta(0);
  const auto positions = meta.input_tensor_meta(1);
  const auto selector = meta.input_tensor_meta(2);
  ET_CHECK_OR_RETURN_ERROR(
      tokens.ok() && positions.ok() && selector.ok(),
      InvalidProgram,
      "CudaExecutor: %s inputs must be tensors",
      method);
  const auto token_sizes = tokens->sizes();
  const auto position_sizes = positions->sizes();
  ET_CHECK_OR_RETURN_ERROR(
      tokens->scalar_type() == ::executorch::aten::ScalarType::Long &&
          token_sizes.size() == 2 && token_sizes[0] == 1 &&
          token_sizes[1] > 0 &&
          positions->scalar_type() == ::executorch::aten::ScalarType::Long &&
          position_sizes.size() == 1 && position_sizes[0] == token_sizes[1] &&
          selector->scalar_type() == ::executorch::aten::ScalarType::Long &&
          selector->sizes().size() == 1,
      InvalidProgram,
      "CudaExecutor: %s must take Long[1, T] tokens, Long[T] positions and a "
      "rank-one Long selector",
      method);
  return static_cast<int>(token_sizes[1]);
}

// The logits width `method` produces.
Result<std::int64_t> logits_width(const MethodMeta& meta, const char* method) {
  ET_CHECK_OR_RETURN_ERROR(
      meta.num_outputs() > 0,
      InvalidProgram,
      "CudaExecutor: %s publishes no outputs",
      method);
  const auto logits = meta.output_tensor_meta(0);
  ET_CHECK_OK_OR_RETURN_ERROR(logits.error());
  const auto sizes = logits->sizes();
  ET_CHECK_OR_RETURN_ERROR(
      sizes.size() >= 2 && sizes[sizes.size() - 1] > 0 &&
          is_supported_logits_type(logits->scalar_type()),
      InvalidProgram,
      "CudaExecutor: %s logits must have shape [..., vocab] and a float dtype",
      method);
  return sizes[sizes.size() - 1];
}

// The one backend `method` delegates to.
Result<std::string> backend_of(const MethodMeta& meta, const char* method) {
  std::string backend_id;
  for (std::size_t i = 0; i < meta.num_backends(); ++i) {
    const auto name = meta.get_backend_name(i);
    ET_CHECK_OK_OR_RETURN_ERROR(name.error());
    ET_CHECK_OR_RETURN_ERROR(
        backend_id.empty() || backend_id == name.get(),
        InvalidProgram,
        "CudaExecutor: %s spans more than one backend",
        method);
    backend_id = name.get();
  }
  ET_CHECK_OR_RETURN_ERROR(
      backend_id == kCudaBackendId,
      InvalidProgram,
      "CudaExecutor: %s is not delegated to %s",
      method,
      kCudaBackendId);
  return backend_id;
}

Error set_backend_options(const CudaExecutorOptions& options) {
  ::executorch::runtime::BackendOptions<2> backend_options;
  ET_CHECK_OK_OR_RETURN_ERROR(backend_options.set_option(
      "weight_sharing_across_methods", options.weight_sharing_across_methods));
  ET_CHECK_OK_OR_RETURN_ERROR(backend_options.set_option(
      "enable_cuda_graph_for_method",
      options.cuda_graph_for_decode ? kDecodeMethod : ""));
  return ::executorch::runtime::set_option(
      kCudaBackendId, backend_options.view());
}

} // namespace

CudaExecutor::CudaExecutor(
    std::unique_ptr<Module> module,
    std::shared_ptr<llm_cache::Cache> cache,
    int max_sessions,
    int max_session_tokens,
    std::string backend_id,
    std::int32_t vocab_size,
    int max_step_tokens)
    : install_guard_(cache),
      module_(std::move(module)),
      ctl_(cache->as<llm_cache::BatchControl>()),
      kv_(cache->as<CudaKVCache>()),
      max_sessions_(max_sessions),
      max_session_tokens_(max_session_tokens),
      backend_id_(std::move(backend_id)),
      vocab_size_(vocab_size),
      max_step_tokens_(max_step_tokens) {}

CudaExecutor::~CudaExecutor() = default;

Result<std::unique_ptr<CudaExecutor>> CudaExecutor::create(
    std::unique_ptr<Module> module,
    int max_sessions,
    int max_session_tokens,
    int kv_dtype,
    int initial_capacity,
    CudaExecutorOptions options) {
  ET_CHECK_OR_RETURN_ERROR(
      module != nullptr, InvalidArgument, "CudaExecutor: no program");
  ET_CHECK_OR_RETURN_ERROR(
      max_sessions > 0 && max_session_tokens > 0 &&
          max_sessions <= std::numeric_limits<int>::max() / max_session_tokens,
      InvalidArgument,
      "CudaExecutor: invalid session limits");
  ET_CHECK_OK_OR_RETURN_ERROR(module->load());

  ET_ASSIGN_OR_RETURN(
      max_context_length,
      metadata::read_max_context_length(*module));
  ET_CHECK_OR_RETURN_ERROR(
      max_session_tokens <= max_context_length,
      InvalidArgument,
      "CudaExecutor: max session tokens %d exceeds model context length %" PRId64,
      max_session_tokens,
      max_context_length);
  ET_ASSIGN_OR_RETURN(
      logits_mode,
      metadata::read_logits_to_keep_mode(*module));
  // decode's selector is static at one row, so only a program whose logits
  // are selected per input fits.
  ET_CHECK_OR_RETURN_ERROR(
      logits_mode == LogitsToKeepMode::Selected,
      NotSupported,
      "CudaExecutor: the program must select its logits rows");
  ET_ASSIGN_OR_RETURN(
      geometry, metadata::read_cache_geometry(*module));
  ET_ASSIGN_OR_RETURN(
      max_cells,
      metadata::detail::read_required_positive_int(*module, kMaxCellsMethod));

  ET_ASSIGN_OR_RETURN(
      decode_meta, module->method_meta(kDecodeMethod));
  ET_ASSIGN_OR_RETURN(
      prefill_meta, module->method_meta(kPrefillMethod));
  ET_ASSIGN_OR_RETURN(
      decode_width, step_width(decode_meta, kDecodeMethod));
  ET_ASSIGN_OR_RETURN(
      max_step_tokens, step_width(prefill_meta, kPrefillMethod));
  ET_CHECK_OR_RETURN_ERROR(
      decode_width == 1 && max_step_tokens >= kMinPrefillTokens &&
          max_step_tokens <= max_cells,
      InvalidProgram,
      "CudaExecutor: decode must take one token and prefill [%d, max_cells]; "
      "got %d and %d",
      kMinPrefillTokens,
      decode_width,
      max_step_tokens);
  ET_ASSIGN_OR_RETURN(
      decode_vocab, logits_width(decode_meta, kDecodeMethod));
  ET_ASSIGN_OR_RETURN(
      prefill_vocab,
      logits_width(prefill_meta, kPrefillMethod));
  ET_CHECK_OR_RETURN_ERROR(
      decode_vocab == prefill_vocab,
      InvalidProgram,
      "CudaExecutor: decode and prefill disagree on the vocabulary");
  ET_ASSIGN_OR_RETURN(
      published_vocab, metadata::read_vocab_size(*module));
  ET_ASSIGN_OR_RETURN(
      vocab_size,
      metadata::check_vocab_size(published_vocab, decode_vocab));
  ET_ASSIGN_OR_RETURN(
      backend_id, backend_of(decode_meta, kDecodeMethod));
  ET_ASSIGN_OR_RETURN(
      prefill_backend,
      backend_of(prefill_meta, kPrefillMethod));
  (void)prefill_backend;

  // Every resident session may fill its budget at once; the pool must hold
  // them all, or a step could find no free cell mid-generation.
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<std::int64_t>(max_sessions) * max_session_tokens <= max_cells,
      InvalidArgument,
      "CudaExecutor: %d sessions of %d tokens exceed the program's %" PRId64
      " cells",
      max_sessions,
      max_session_tokens,
      max_cells);

  // The pool is the program's: its size and widest step fix the shapes the
  // program declared for the step buffers.
  llm_cache::CacheConfig cfg{static_cast<int>(max_cells), kv_dtype};
  cfg.max_write = max_step_tokens;
  if (initial_capacity >= 0) {
    cfg.initial_capacity = std::min(initial_capacity, cfg.capacity);
  }
  ET_CHECK_OR_RETURN_ERROR(
      llm_cache::valid(geometry, cfg),
      InvalidProgram,
      "CudaExecutor: the program's layout is unusable");
  auto built = llm_cache::CacheFactory::global().build(
      backend_id, llm_cache::kind::kBatchedCell, geometry, cfg);
  ET_CHECK_OK_OR_RETURN_ERROR(built.error());
  std::shared_ptr<llm_cache::Cache> cache = built.get();
  auto* const ctl = cache->as<llm_cache::BatchControl>();
  ET_CHECK_OR_RETURN_ERROR(
      ctl != nullptr && cache->as<CudaKVCache>() != nullptr,
      InvalidType,
      "CudaExecutor: the cache is not a batched CUDA cache");
  const std::optional<int> seq_limit = ctl->max_seqs();
  ET_CHECK_OR_RETURN_ERROR(
      !seq_limit || max_sessions <= *seq_limit,
      InvalidArgument,
      "CudaExecutor: %d resident sessions requested, but the layout holds %d",
      max_sessions,
      seq_limit.value_or(0));

  ET_CHECK_OK_OR_RETURN_ERROR(set_backend_options(options));
  return std::unique_ptr<CudaExecutor>(new CudaExecutor(
      std::move(module),
      std::move(cache),
      max_sessions,
      max_session_tokens,
      std::move(backend_id),
      vocab_size,
      max_step_tokens));
}

bool CudaExecutor::initialize() {
  // The delegate resolves the cache from this key while each method loads.
  ::executorch::runtime::BackendOptions<1> options;
  ::executorch::runtime::LoadBackendOptionsMap options_map;
  if (install_guard_.set_option(options) != Error::Ok ||
      options_map.set_options(backend_id_.c_str(), options.view()) !=
          Error::Ok) {
    ET_LOG(Error, "CudaExecutor: could not name the cache to the backend");
    return false;
  }
  for (const char* method : {kDecodeMethod, kPrefillMethod}) {
    if (module_->load_method(
            method,
            /*planned_memory=*/nullptr,
            /*event_tracer=*/nullptr,
            &options_map) != Error::Ok) {
      ET_LOG(Error, "CudaExecutor: could not load %s", method);
      return false;
    }
  }
  return true;
}

// Copied from ModuleExecutor::build_step; see there for the reasoning.
Result<CudaExecutor::Step> CudaExecutor::build_step(const BatchInput& batch) {
  Step step;
  const std::size_t total = batch.size();
  step.tokens.reserve(total);
  step.positions.reserve(total);
  step.seq_ids.reserve(total);
  step.logit_indices.reserve(batch.inputs.size());
  std::vector<std::pair<std::int32_t, int>> rewinds;
  std::unordered_map<std::int32_t, int> cursor;

  for (const Input& input : batch.inputs) {
    const auto seq_it = sessions_.find(input.sid);
    if (seq_it == sessions_.end()) {
      ET_LOG(Error, "build_step: session %" PRId64 " is not open", input.sid);
      return Error::InvalidArgument;
    }
    const std::int32_t seq_id = seq_it->second.seq_id;
    if (input.size == 0 || !input.tokens ||
        input.offset > input.tokens->size() ||
        input.size > input.tokens->size() - input.offset) {
      ET_LOG(
          Error,
          "build_step: session %" PRId64 " gave a slice its tokens do not hold",
          input.sid);
      return Error::InvalidArgument;
    }

    const std::int64_t start = static_cast<std::int64_t>(input.position) +
        static_cast<std::int64_t>(input.offset);
    const auto [cursor_it, first_for_seq] =
        cursor.try_emplace(seq_id, ctl_->pos(seq_id));
    int& at = cursor_it->second;
    if (start > at) {
      ET_LOG(
          Error,
          "build_step: session %" PRId64 " starts at %" PRId64
          " over a sequence holding %d",
          input.sid,
          start,
          at);
      return Error::InvalidArgument;
    }
    if (start < at) {
      if (!first_for_seq) {
        ET_LOG(
            Error,
            "build_step: session %" PRId64 " overlaps its earlier input",
            input.sid);
        return Error::InvalidArgument;
      }
      if (start == 0) {
        ET_LOG(
            Error,
            "build_step: session %" PRId64 " reopens from the start",
            input.sid);
        return Error::InvalidArgument;
      }
      rewinds.emplace_back(seq_id, static_cast<int>(start));
      at = static_cast<int>(start);
    }

    const std::int64_t end = start + static_cast<std::int64_t>(input.size);
    if (end > max_session_tokens_) {
      ET_LOG(
          Error,
          "build_step: session %" PRId64 " reaches %" PRId64 " of %d cells",
          input.sid,
          end,
          max_session_tokens_);
      return Error::OutOfResources;
    }

    const Token* slice = input.tokens->data() + input.offset;
    for (std::size_t k = 0; k < input.size; ++k) {
      step.tokens.push_back(static_cast<std::int64_t>(slice[k]));
      step.positions.push_back(start + static_cast<std::int64_t>(k));
    }
    step.seq_ids.insert(step.seq_ids.end(), input.size, seq_id);
    at = static_cast<int>(end);
    step.logit_indices.push_back(
        input.produce_output ? static_cast<int>(step.tokens.size()) - 1 : -1);
  }

  for (const auto& [seq_id, from] : rewinds) {
    if (!ctl_->rewind(seq_id, from)) {
      ET_LOG(Error, "build_step: sequence %d would not truncate", seq_id);
      return Error::Internal;
    }
  }
  return step;
}

std::optional<SessionId> CudaExecutor::open_session() {
  if (sessions_.size() >= static_cast<std::size_t>(max_sessions_) ||
      next_session_ == 0) {
    return std::nullopt;
  }
#if ET_HAS_EXCEPTIONS
  try {
#endif
    const std::optional<std::int32_t> seq_id = ctl_->seq_new();
    return seq_id ? publish_session(*seq_id, 0) : std::nullopt;
#if ET_HAS_EXCEPTIONS
  } catch (const std::bad_alloc&) {
    return std::nullopt;
  }
#endif
}

std::optional<SessionId> CudaExecutor::publish_session(
    std::int32_t seq_id,
    Position position) {
  SequenceGuard guard{*ctl_, seq_id};
  if (ctl_->pos(seq_id) != position) {
    return std::nullopt;
  }
  const SessionId session = next_session_;
  if (!sessions_.emplace(session, SessionState{seq_id, nullptr}).second) {
    return std::nullopt;
  }
  guard.owned = false;
  next_session_ =
      session == std::numeric_limits<SessionId>::max() ? 0 : session + 1;
  return session;
}

void CudaExecutor::close_session(SessionId session) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end()) {
    return;
  }
  ctl_->seq_rm(it->second.seq_id);
  sessions_.erase(it);
}

std::optional<SessionId> CudaExecutor::clone(SessionId source, Position upto) {
  const auto it = sessions_.find(source);
  if (it == sessions_.end() || upto < 0 || upto > max_session_tokens_ ||
      sessions_.size() >= static_cast<std::size_t>(max_sessions_) ||
      next_session_ == 0) {
    return std::nullopt;
  }
#if ET_HAS_EXCEPTIONS
  try {
#endif
    if (upto > ctl_->pos(it->second.seq_id)) {
      return std::nullopt;
    }
    const auto seq_id = ctl_->seq_clone(it->second.seq_id, upto);
    return seq_id ? publish_session(*seq_id, upto) : std::nullopt;
#if ET_HAS_EXCEPTIONS
  } catch (const std::bad_alloc&) {
    return std::nullopt;
  }
#endif
}

void CudaExecutor::set_sampling(
    SessionId session,
    const SamplingParams& params,
    std::optional<std::uint64_t> seed) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end()) {
    return;
  }
  it->second.sampler = std::make_unique<Sampler>(
      vocab_size_,
      params.temperature,
      params.top_p,
      seed.value_or(nondeterministic_seed()));
  it->second.sampler->set_topk(params.top_k);
}

bool CudaExecutor::execute(const BatchInput& batch, BatchOutput& out) {
  out.outputs.clear();
  out.outputs.resize(batch.inputs.size());

  const Result<Step> step = build_step(batch);
  if (!step.ok()) {
    return false;
  }

  // In order, so a slice attends the cells its predecessors wrote. Each
  // input's logits row falls in exactly one slice.
  const int total = static_cast<int>(step->tokens.size());
  for (const StepSlice& slice :
       plan_slices(total, max_step_tokens_, kMinPrefillTokens)) {
    const int off = slice.offset;
    const int n = slice.length;
    const char* method =
        slice.method == StepMethod::Decode ? kDecodeMethod : kPrefillMethod;
    // Placement checks the forward's token count against the declaration, so
    // each slice declares its own.
    if (!ctl_->declare_step(std::vector<std::int32_t>(
            step->seq_ids.begin() + off, step->seq_ids.begin() + off + n))) {
      ET_LOG(Error, "CudaExecutor: the cache refused a slice of %d", n);
      return false;
    }
    auto tokens = make_tensor_ptr(
        {1, n},
        std::vector<std::int64_t>(
            step->tokens.begin() + off, step->tokens.begin() + off + n));
    auto positions = make_tensor_ptr(
        {n},
        std::vector<std::int64_t>(
            step->positions.begin() + off, step->positions.begin() + off + n));
    std::vector<std::int64_t> selector_values;
    std::vector<std::size_t> selected_inputs;
    for (std::size_t i = 0; i < step->logit_indices.size(); ++i) {
      const int row = step->logit_indices[i];
      if (row >= off && row < off + n) {
        selector_values.push_back(row - off);
        selected_inputs.push_back(i);
      }
    }
    if (selector_values.empty()) {
      // The forward still has to produce a row; nothing reads it.
      selector_values.push_back(n - 1);
    }
    const int rows = static_cast<int>(selector_values.size());
    auto selector = make_tensor_ptr({rows}, std::move(selector_values));

    auto result = module_->execute(method, {tokens, positions, selector});
    if (!result.ok()) {
      ET_LOG(
          Error,
          "CudaExecutor: %s failed with 0x%x",
          method,
          static_cast<unsigned>(result.error()));
      return false;
    }
    if (result->empty() || !result->at(0).isTensor()) {
      ET_LOG(Error, "CudaExecutor: %s returned no logits", method);
      return false;
    }
    auto logits = result->at(0).toTensor();
    if (logits.dim() < 2 || logits.size(logits.dim() - 1) != vocab_size_ ||
        logits.numel() != static_cast<std::int64_t>(rows) * vocab_size_) {
      ET_LOG(
          Error,
          "CudaExecutor: %s returned logits that are not [%d, %d]",
          method,
          rows,
          vocab_size_);
      return false;
    }
    for (std::size_t row = 0; row < selected_inputs.size(); ++row) {
      const std::size_t input_index = selected_inputs[row];
      const SessionId session = batch.inputs[input_index].sid;
      const std::optional<Token> token =
          sample_row(logits, static_cast<int>(row), session);
      if (!token) {
        return false;
      }
      out.outputs[input_index] = Output{session, {*token}};
    }
  }
  return true;
}

std::optional<Token> CudaExecutor::sample_row(
    ::executorch::aten::Tensor& logits,
    int row,
    SessionId session) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end() || it->second.sampler == nullptr) {
    ET_LOG(
        Error, "CudaExecutor: session %" PRId64 " has no sampling policy", session);
    return std::nullopt;
  }
  auto one_row = make_tensor_ptr(
      {vocab_size_},
      static_cast<std::uint8_t*>(logits.mutable_data_ptr()) +
          static_cast<std::size_t>(row) * vocab_size_ *
              ::executorch::runtime::elementSize(logits.scalar_type()),
      logits.scalar_type());
  return static_cast<Token>(
      ::executorch::extension::llm::sample_from_logits(
          *one_row, *it->second.sampler));
}

OffGraphKVMetrics CudaExecutor::kv_metrics() const {
  return kv_->metrics();
}

} // namespace executorch::backends::cuda::batching
