/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/batching/cuda_executor.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <limits>
#include <set>
#include <string_view>
#include <unordered_set>
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

bool is_long_vector(const MethodMeta& meta, std::size_t output) {
  const auto tensor = meta.output_tensor_meta(output);
  return tensor.ok() &&
      tensor->scalar_type() == ::executorch::aten::ScalarType::Long &&
      tensor->sizes().size() == 1;
}

// Validates an optional device sampler; false when the program lacks it.
Result<bool> check_sampler(
    Module& module,
    const std::unordered_set<std::string>& names,
    const char* method,
    std::size_t num_inputs) {
  if (names.count(method) == 0) {
    return false;
  }
  ET_ASSIGN_OR_RETURN(meta, module.method_meta(method));
  bool valid = meta.num_inputs() == num_inputs && meta.num_outputs() == 1 &&
      is_long_vector(meta, 0);
  for (std::size_t i = 0; valid && i < num_inputs; ++i) {
    const auto input = meta.input_tensor_meta(i);
    valid = input.ok() && input->sizes().size() == 2 &&
        (i == 0 ? is_supported_logits_type(input->scalar_type())
                : input->scalar_type() == ::executorch::aten::ScalarType::Float &&
                 input->sizes()[1] == 4);
  }
  ET_CHECK_OR_RETURN_ERROR(
      valid,
      InvalidProgram,
      "CudaExecutor: %s must map [R, vocab] logits%s to Long [R] tokens",
      method,
      num_inputs == 2 ? " and Float [R, 4] params" : "");
  return true;
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
    int min_selected_rows,
    bool has_sampler,
    bool has_argmax)
    : install_guard_(cache),
      module_(std::move(module)),
      ctl_(cache->as<llm_cache::BatchControl>()),
      kv_(cache->as<CudaKVCache>()),
      backend_id_(std::move(backend_id)),
      vocab_size_(vocab_size),
      methods_(std::move(methods)),
      calls_(methods_.size(), 0),
      min_selected_rows_(min_selected_rows),
      sessions_(*ctl_, max_sessions, max_session_tokens, vocab_size),
      has_sampler_(has_sampler),
      has_argmax_(has_argmax) {}

CudaExecutor::~CudaExecutor() {
  if (device_params_ != nullptr) {
    (void)cudaFree(device_params_);
  }
}

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
  const bool has_dynamic = !methods.back().is_static;
  const int widest_static = *static_widths.rbegin();
  ET_CHECK_OR_RETURN_ERROR(
      !has_dynamic || methods.back().max_tokens > widest_static,
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
  ET_ASSIGN_OR_RETURN(
      declared_min_rows,
      metadata::detail::read_int_method(*module, kMinSelectedRowsMethod));
  const int min_selected_rows = static_cast<int>(
      std::max<std::int64_t>(1, declared_min_rows.value_or(1)));
  ET_CHECK_OR_RETURN_ERROR(
      min_selected_rows <= max_step_tokens,
      InvalidProgram,
      "CudaExecutor: %d selected rows exceed the widest step",
      min_selected_rows);
  ET_ASSIGN_OR_RETURN(published_vocab, metadata::read_vocab_size(*module));
  ET_ASSIGN_OR_RETURN(
      vocab_size, metadata::check_vocab_size(published_vocab, method_vocab));
  ET_ASSIGN_OR_RETURN(has_sampler, check_sampler(*module, names, kSampleMethod, 2));
  ET_ASSIGN_OR_RETURN(has_argmax, check_sampler(*module, names, kArgmaxMethod, 1));
  ET_CHECK_OR_RETURN_ERROR(
      has_sampler || !has_argmax,
      InvalidProgram,
      "CudaExecutor: %s needs %s for non-greedy rows",
      kArgmaxMethod,
      kSampleMethod);

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
      min_selected_rows,
      has_sampler,
      has_argmax));
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
  for (const auto& [present, name] :
       {std::pair{has_sampler_, kSampleMethod},
        std::pair{has_argmax_, kArgmaxMethod}}) {
    if (present && module_->load_method(name) != Error::Ok) {
      ET_LOG(Error, "CudaExecutor: could not load %s", name);
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
    if (selected.inputs.empty()) {
      continue;
    }
    std::vector<SessionId> row_sessions;
    row_sessions.reserve(selected.inputs.size());
    for (const std::size_t input_index : selected.inputs) {
      row_sessions.push_back(batch.inputs[input_index].sid);
    }
    std::vector<Token> tokens_out;
    if (logits.device_type() == ::executorch::aten::DeviceType::CUDA) {
      if (!has_sampler_) {
        ET_LOG(
            Error,
            "CudaExecutor: %s returns device logits but the program has no %s",
            method.name.c_str(),
            kSampleMethod);
        return false;
      }
      auto sampled = sample_on_device(logits, row_sessions);
      if (!sampled) {
        return false;
      }
      tokens_out = std::move(*sampled);
    } else {
      for (std::size_t row = 0; row < row_sessions.size(); ++row) {
        const std::optional<Token> token =
            sessions_.sample(row_sessions[row], logits, static_cast<int>(row));
        if (!token) {
          return false;
        }
        tokens_out.push_back(*token);
      }
    }
    for (std::size_t row = 0; row < selected.inputs.size(); ++row) {
      out.outputs[selected.inputs[row]] =
          Output{row_sessions[row], {tokens_out[row]}};
    }
  }
  return true;
}

std::optional<std::vector<Token>> CudaExecutor::sample_on_device(
    const ::executorch::aten::Tensor& logits,
    const std::vector<SessionId>& sessions) {
  using ::executorch::aten::ScalarType;
  using ::executorch::aten::SizesType;
  const ::executorch::aten::Device cuda(
      ::executorch::aten::DeviceType::CUDA, logits.device_index());
  const auto rows = static_cast<SizesType>(sessions.size());
  // The rows that sample lead the logits; the rest are padding.
  auto row_logits = make_tensor_ptr(
      {rows, static_cast<SizesType>(vocab_size_)},
      logits.mutable_data_ptr(),
      logits.scalar_type(),
      cuda);

  const bool all_greedy = std::all_of(
      sessions.begin(), sessions.end(), [this](SessionId session) {
        return sessions_.greedy(session);
      });
  // Every row's generator advances whichever method runs, as on the host.
  std::vector<float> params;
  params.reserve(sessions.size() * 4);
  for (const SessionId session : sessions) {
    const auto row = sessions_.device_sampling(session);
    if (!row) {
      return std::nullopt;
    }
    params.insert(params.end(), {row->temperature, row->top_p, row->top_k, row->coin});
  }

  auto run_sampler = [&]() -> Result<std::vector<::executorch::runtime::EValue>> {
    if (all_greedy && has_argmax_) {
      return module_->execute(
          kArgmaxMethod, std::vector<::executorch::runtime::EValue>{row_logits});
    }
    if (sessions.size() > device_params_rows_) {
      if (device_params_ != nullptr) {
        (void)cudaFree(device_params_);
        device_params_ = nullptr;
        device_params_rows_ = 0;
      }
      ET_CHECK_OR_RETURN_ERROR(
          cudaMalloc(&device_params_, sessions.size() * 4 * sizeof(float)) ==
              cudaSuccess,
          MemoryAllocationFailed,
          "CudaExecutor: could not stage the sampler's params");
      device_params_rows_ = sessions.size();
    }
    // The backend runs every method on the per-thread stream; ordering the
    // upload there puts it ahead of the sampler.
    ET_CHECK_OR_RETURN_ERROR(
        cudaMemcpyAsync(
            device_params_,
            params.data(),
            params.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            cudaStreamPerThread) == cudaSuccess,
        Internal,
        "CudaExecutor: could not upload the sampler's params");
    auto device_params =
        make_tensor_ptr({rows, 4}, device_params_, ScalarType::Float, cuda);
    return module_->execute(kSampleMethod, {row_logits, device_params});
  };
  const auto result = run_sampler();
  if (!result.ok() || result->empty() || !result->at(0).isTensor()) {
    ET_LOG(Error, "CudaExecutor: the device sampler failed");
    return std::nullopt;
  }
  const auto& tokens = result->at(0).toTensor();
  if (tokens.numel() != rows || tokens.scalar_type() != ScalarType::Long) {
    ET_LOG(Error, "CudaExecutor: the device sampler returned a bad shape");
    return std::nullopt;
  }
  std::vector<std::int64_t> host(sessions.size());
  const auto kind = tokens.device_type() == ::executorch::aten::DeviceType::CUDA
      ? cudaMemcpyDeviceToHost
      : cudaMemcpyHostToHost;
  if (cudaMemcpyAsync(
          host.data(),
          tokens.const_data_ptr(),
          host.size() * sizeof(std::int64_t),
          kind,
          cudaStreamPerThread) != cudaSuccess ||
      cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess) {
    ET_LOG(Error, "CudaExecutor: could not read the sampled tokens");
    return std::nullopt;
  }
  return std::vector<Token>(host.begin(), host.end());
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
