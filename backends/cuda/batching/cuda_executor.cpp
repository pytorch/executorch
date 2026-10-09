/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/batching/cuda_executor.h>

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <limits>
#include <set>
#include <string_view>
#include <utility>
#include <vector>

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

// The token count a static method's name carries: N for forward_{N}.
std::optional<int> static_method_width(std::string_view name) {
  const std::string_view prefix = kStaticMethodPrefix;
  if (name.size() <= prefix.size() || name.substr(0, prefix.size()) != prefix) {
    return std::nullopt;
  }
  int width = 0;
  for (const char c : name.substr(prefix.size())) {
    if (!std::isdigit(static_cast<unsigned char>(c)) ||
        width > (std::numeric_limits<int>::max() - 9) / 10) {
      return std::nullopt;
    }
    width = width * 10 + (c - '0');
  }
  return width > 0 ? std::optional<int>(width) : std::nullopt;
}

Error set_backend_options(
    const CudaExecutorOptions& options,
    const std::vector<ForwardMethod>& methods) {
  // The backend matches a comma-separated list of method names.
  std::string graph_methods;
  if (options.cuda_graph_for_static_methods) {
    for (const ForwardMethod& method : methods) {
      if (method.is_static) {
        graph_methods += (graph_methods.empty() ? "" : ",") + method.name;
      }
    }
  }
  ET_CHECK_OR_RETURN_ERROR(
      graph_methods.size() < ::executorch::runtime::kMaxOptionValueLength,
      InvalidArgument,
      "CudaExecutor: too many static methods to name in one backend option");
  ::executorch::runtime::BackendOptions<2> backend_options;
  ET_CHECK_OK_OR_RETURN_ERROR(backend_options.set_option(
      "weight_sharing_across_methods", options.weight_sharing_across_methods));
  ET_CHECK_OK_OR_RETURN_ERROR(backend_options.set_option(
      "enable_cuda_graph_for_method", graph_methods.c_str()));
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
    std::vector<ForwardMethod> methods,
    int min_selected_rows)
    : install_guard_(cache),
      module_(std::move(module)),
      ctl_(cache->as<llm_cache::BatchControl>()),
      kv_(cache->as<CudaKVCache>()),
      backend_id_(std::move(backend_id)),
      vocab_size_(vocab_size),
      methods_(std::move(methods)),
      calls_(methods_.size(), 0),
      min_selected_rows_(min_selected_rows),
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
      max_context_length, metadata::read_max_context_length(*module));
  ET_CHECK_OR_RETURN_ERROR(
      max_session_tokens <= max_context_length,
      InvalidArgument,
      "CudaExecutor: max session tokens %d exceeds model context length %" PRId64,
      max_session_tokens,
      max_context_length);
  ET_ASSIGN_OR_RETURN(logits_mode, metadata::read_logits_to_keep_mode(*module));
  // decode's selector is static at one row, so only a program whose logits
  // are selected per input fits.
  // Static methods select one row per token, so only a program whose logits
  // are selected per input fits.
  ET_CHECK_OR_RETURN_ERROR(
      logits_mode == LogitsToKeepMode::Selected,
      NotSupported,
      "CudaExecutor: the program must select its logits rows");
  ET_ASSIGN_OR_RETURN(geometry, metadata::read_cache_geometry(*module));
  ET_ASSIGN_OR_RETURN(
      max_cells,
      metadata::detail::read_required_positive_int(*module, kMaxCellsMethod));

  // Every forward_{N} and forward_others the program exports, narrowest first.
  ET_ASSIGN_OR_RETURN(names, module->method_names());
  std::vector<ForwardMethod> methods;
  std::string backend_id;
  std::int64_t method_vocab = 0;
  std::set<int> static_widths;
  for (const std::string& name : names) {
    const std::optional<int> declared = static_method_width(name);
    const bool dynamic = name == kDynamicMethod;
    if (!declared && !dynamic) {
      continue;
    }
    ET_ASSIGN_OR_RETURN(meta, module->method_meta(name.c_str()));
    ET_ASSIGN_OR_RETURN(width, step_width(meta, name.c_str()));
    ET_ASSIGN_OR_RETURN(vocab, logits_width(meta, name.c_str()));
    ET_ASSIGN_OR_RETURN(backend, backend_of(meta, name.c_str()));
    ET_CHECK_OR_RETURN_ERROR(
        !declared || width == *declared,
        InvalidProgram,
        "CudaExecutor: %s takes %d tokens",
        name.c_str(),
        width);
    ET_CHECK_OR_RETURN_ERROR(
        method_vocab == 0 || vocab == method_vocab,
        InvalidProgram,
        "CudaExecutor: the forward methods disagree on the vocabulary");
    method_vocab = vocab;
    backend_id = backend;
    if (declared) {
      static_widths.insert(width);
    }
    methods.push_back({name, width, !dynamic});
  }
  ET_CHECK_OR_RETURN_ERROR(
      !static_widths.empty(),
      InvalidProgram,
      "CudaExecutor: the program exports no %sN method",
      kStaticMethodPrefix);
  std::sort(
      methods.begin(),
      methods.end(),
      [](const ForwardMethod& a, const ForwardMethod& b) {
        return a.max_tokens < b.max_tokens;
      });
  // The dynamic method serves only what no static method holds, so it must be
  // the widest.
  const auto dynamic =
      std::find_if(methods.begin(), methods.end(), [](const ForwardMethod& m) {
        return !m.is_static;
      });
  ET_CHECK_OR_RETURN_ERROR(
      dynamic == methods.end() || dynamic->max_tokens > *static_widths.rbegin(),
      InvalidProgram,
      "CudaExecutor: %s must be wider than every static method",
      kDynamicMethod);
  ET_CHECK_OR_RETURN_ERROR(
      std::count_if(
          methods.begin(),
          methods.end(),
          [](const ForwardMethod& m) { return !m.is_static; }) <= 1,
      InvalidProgram,
      "CudaExecutor: more than one dynamic method");
  const int max_step_tokens = methods.back().max_tokens;
  ET_CHECK_OR_RETURN_ERROR(
      max_step_tokens < max_cells,
      InvalidProgram,
      "CudaExecutor: a %d-token step does not fit %" PRId64 " cells",
      max_step_tokens,
      max_cells);
  // A dynamic method whose kernels switch at a small width is exported from
  // more selected rows than one; its selector is padded up to that.
  int min_selected_rows = 1;
  if (dynamic != methods.end()) {
    ET_ASSIGN_OR_RETURN(
        declared_min_rows,
        metadata::detail::read_int_method(*module, kMinSelectedRowsMethod));
    const std::int64_t requested_min_rows =
        std::max<std::int64_t>(1, declared_min_rows.value_or(1));
    ET_CHECK_OR_RETURN_ERROR(
        requested_min_rows <= max_step_tokens,
        InvalidProgram,
        "CudaExecutor: %" PRId64 " selected rows exceed the widest step",
        requested_min_rows);
    min_selected_rows = static_cast<int>(requested_min_rows);
  }
  ET_ASSIGN_OR_RETURN(published_vocab, metadata::read_vocab_size(*module));
  ET_ASSIGN_OR_RETURN(
      vocab_size, metadata::check_vocab_size(published_vocab, method_vocab));

  // Every resident session may fill its budget at once; the pool must hold
  // them all, or a step could find no free cell mid-generation. One of the
  // program's cells is the padding scratch row, never handed out.
  ET_CHECK_OR_RETURN_ERROR(
      static_cast<std::int64_t>(max_sessions) * max_session_tokens < max_cells,
      InvalidArgument,
      "CudaExecutor: %d sessions of %d tokens exceed the program's %" PRId64
      " cells, one of which is reserved",
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

  ET_CHECK_OK_OR_RETURN_ERROR(set_backend_options(options, methods));
  return std::unique_ptr<CudaExecutor>(new CudaExecutor(
      std::move(module),
      std::move(cache),
      max_sessions,
      max_session_tokens,
      std::move(backend_id),
      vocab_size,
      std::move(methods),
      min_selected_rows));
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
  for (const ForwardMethod& method : methods_) {
    if (module_->load_method(
            method.name,
            /*planned_memory=*/nullptr,
            /*event_tracer=*/nullptr,
            &options_map) != Error::Ok) {
      ET_LOG(Error, "CudaExecutor: could not load %s", method.name.c_str());
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
  for (const StepSlice& slice : plan_slices(total, methods_)) {
    const int off = slice.offset;
    const int n = slice.length;
    const ForwardMethod& method = methods_[slice.method];
    // Only the real tokens are declared; the cache sends a static method's
    // padding rows to its scratch row.
    if (!ctl_->declare_step(std::vector<std::int32_t>(
            step->seq_ids.begin() + off, step->seq_ids.begin() + off + n))) {
      ET_LOG(Error, "CudaExecutor: the cache refused a slice of %d", n);
      return false;
    }
    // Padding tokens are token 0 at position 0: valid inputs whose rows
    // nothing reads.
    std::vector<std::int64_t> token_values(
        step->tokens.begin() + off, step->tokens.begin() + off + n);
    std::vector<std::int64_t> position_values(
        step->positions.begin() + off, step->positions.begin() + off + n);
    token_values.resize(slice.width, 0);
    position_values.resize(slice.width, 0);
    auto tokens = make_tensor_ptr({1, slice.width}, std::move(token_values));
    auto positions = make_tensor_ptr({slice.width}, std::move(position_values));

    // A static method selects exactly its width in rows, and the dynamic one
    // at least its exported minimum. Extra rows repeat the last real one (or
    // row 0 when the slice finishes no input); nothing reads them.
    auto selected = llm_batching::util::select_rows(*step, off, n);
    const std::size_t rows_needed = method.is_static
        ? static_cast<std::size_t>(slice.width)
        : std::max<std::size_t>(
              selected.selector.size(),
              static_cast<std::size_t>(min_selected_rows_));
    const std::int64_t fill =
        selected.selector.empty() ? 0 : selected.selector.back();
    selected.selector.resize(rows_needed, fill);
    const int rows = static_cast<int>(selected.selector.size());
    auto selector = make_tensor_ptr({rows}, std::move(selected.selector));

    auto result = module_->execute(method.name, {tokens, positions, selector});
    if (!result.ok()) {
      ET_LOG(
          Error,
          "CudaExecutor: %s failed with 0x%x",
          method.name.c_str(),
          static_cast<unsigned>(result.error()));
      return false;
    }
    ++calls_[slice.method];
    if (result->empty() || !result->at(0).isTensor()) {
      ET_LOG(Error, "CudaExecutor: %s returned no logits", method.name.c_str());
      return false;
    }
    auto logits = result->at(0).toTensor();
    if (logits.dim() < 2 || logits.size(logits.dim() - 1) != vocab_size_ ||
        logits.numel() != static_cast<std::int64_t>(rows) * vocab_size_) {
      ET_LOG(
          Error,
          "CudaExecutor: %s returned logits that are not [%d, %d]",
          method.name.c_str(),
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

std::vector<ForwardMethodCalls> CudaExecutor::method_calls() const {
  std::vector<ForwardMethodCalls> calls;
  calls.reserve(methods_.size());
  for (std::size_t i = 0; i < methods_.size(); ++i) {
    calls.push_back({methods_[i].name, calls_[i]});
  }
  return calls;
}

} // namespace executorch::backends::cuda::batching
