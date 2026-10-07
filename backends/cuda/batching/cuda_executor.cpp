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
#include <utility>
#include <vector>

#include <executorch/backends/cuda/batching/step_plan.h>
#include <executorch/backends/cuda/runtime/backend_options.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda::batching {

using ::executorch::extension::make_tensor_ptr;
using ::executorch::extension::Module;
using ::executorch::extension::llm::LogitsToKeepMode;
using ::executorch::runtime::Error;
using ::executorch::runtime::MethodMeta;
using ::executorch::runtime::Result;
using llm_batching::BatchInput;
using llm_batching::BatchOutput;
using llm_batching::Output;
using llm_batching::Position;
using llm_batching::SamplingParams;
using llm_batching::SessionId;
using llm_batching::Token;

namespace metadata = ::executorch::extension::llm;

namespace {

// A prefill forward carries at least this many tokens; one token is decode's.
constexpr int kMinPrefillTokens = 2;

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
      backend_id_(std::move(backend_id)),
      vocab_size_(vocab_size),
      max_step_tokens_(max_step_tokens),
      sessions_(*ctl_, max_sessions, max_session_tokens, vocab_size) {}

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
  llm_cache::CacheConfig cfg{};
  cfg.capacity = static_cast<int>(max_cells);
  cfg.kv_dtype = kv_dtype;
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

std::optional<SessionId> CudaExecutor::open_session() {
  return sessions_.open();
}

void CudaExecutor::close_session(SessionId session) {
  sessions_.close(session);
}

std::optional<SessionId> CudaExecutor::clone(SessionId source, Position upto) {
  return sessions_.clone(source, upto);
}

void CudaExecutor::set_sampling(
    SessionId session,
    const SamplingParams& params,
    std::optional<std::uint64_t> seed) {
  sessions_.set_sampling(session, params, seed);
}

bool CudaExecutor::execute(const BatchInput& batch, BatchOutput& out) {
  out.outputs.clear();
  out.outputs.resize(batch.inputs.size());

  const Result<llm_batching::util::PackedStep> step = sessions_.pack(batch);
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
    auto selected = llm_batching::util::select_rows(*step, off, n);
    const int rows = static_cast<int>(selected.selector.size());
    auto selector = make_tensor_ptr({rows}, std::move(selected.selector));

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
    for (std::size_t row = 0; row < selected.inputs.size(); ++row) {
      const std::size_t input_index = selected.inputs[row];
      const SessionId session = batch.inputs[input_index].sid;
      const std::optional<Token> token =
          sessions_.sample(session, logits, static_cast<int>(row));
      if (!token) {
        return false;
      }
      out.outputs[input_index] = Output{session, {*token}};
    }
  }
  return true;
}

OffGraphKVMetrics CudaExecutor::kv_metrics() const {
  return kv_->metrics();
}

} // namespace executorch::backends::cuda::batching
