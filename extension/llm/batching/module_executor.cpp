/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/module_executor.h>

#include <algorithm>
#include <limits>
#include <utility>

#include <executorch/extension/llm/batching/executor_utils.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

using ::executorch::extension::make_tensor_ptr;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace {

bool is_supported_logits_type(::executorch::aten::ScalarType type) {
  using ScalarType = ::executorch::aten::ScalarType;
  return type == ScalarType::Float || type == ScalarType::Half ||
      type == ScalarType::BFloat16 || type == ScalarType::UInt16;
}

} // namespace

ModuleExecutor::ModuleExecutor(
    std::unique_ptr<Module> module,
    std::shared_ptr<cache::Cache> cache,
    int max_sessions,
    int max_session_tokens,
    std::string backend_id,
    std::string method,
    std::int32_t vocab_size,
    int max_step_tokens,
    LogitsToKeepMode logits_to_keep_mode)
    : install_guard_(cache),
      module_(std::move(module)),
      ctl_(cache->as<cache::BatchControl>()),
      backend_id_(std::move(backend_id)),
      method_(std::move(method)),
      vocab_size_(vocab_size),
      max_step_tokens_(max_step_tokens),
      logits_to_keep_mode_(logits_to_keep_mode),
      sessions_(*ctl_, max_sessions, max_session_tokens, vocab_size) {}

ModuleExecutor::~ModuleExecutor() = default;

Result<std::unique_ptr<ModuleExecutor>> ModuleExecutor::create(
    std::unique_ptr<Module> module,
    int max_sessions,
    int max_session_tokens,
    int kv_dtype,
    int initial_capacity,
    std::string cache_kind,
    std::string method) {
  if (module == nullptr) {
    ET_LOG(Error, "ModuleExecutor: no program");
    return Error::InvalidArgument;
  }
  if (max_sessions <= 0 || max_session_tokens <= 0) {
    ET_LOG(Error, "ModuleExecutor: invalid session limits");
    return Error::InvalidArgument;
  }
  const Error load_error =
      module->load(); // no-op once the caller has loaded it
  if (load_error != Error::Ok) {
    ET_LOG(Error, "ModuleExecutor: the program did not load");
    return load_error;
  }

  const auto max_context_length = read_max_context_length(*module);
  if (!max_context_length.ok()) {
    ET_LOG(Error, "ModuleExecutor: the program's metadata is malformed");
    return max_context_length.error();
  }
  if (max_session_tokens > *max_context_length) {
    ET_LOG(
        Error,
        "ModuleExecutor: max session tokens %d exceeds model context length %" PRId64,
        max_session_tokens,
        *max_context_length);
    return Error::InvalidArgument;
  }
  const auto logits_mode_result = read_logits_to_keep_mode(*module);
  if (!logits_mode_result.ok()) {
    ET_LOG(Error, "ModuleExecutor: the program's metadata is malformed");
    return logits_mode_result.error();
  }
  const LogitsToKeepMode logits_mode = *logits_mode_result;
  if (logits_mode == LogitsToKeepMode::Last) {
    ET_LOG(
        Error,
        "ModuleExecutor: logits-to-keep mode last is incompatible with batched execution");
    return Error::NotSupported;
  }

  auto geometry = read_cache_geometry(*module);
  if (!geometry.ok()) {
    return geometry.error();
  }
  if (max_sessions > std::numeric_limits<int>::max() / max_session_tokens) {
    ET_LOG(Error, "ModuleExecutor: total cache capacity exceeds int range");
    return Error::InvalidArgument;
  }
  cache::CacheConfig cfg{max_sessions * max_session_tokens, kv_dtype};
  if (initial_capacity >= 0) {
    cfg.initial_capacity = initial_capacity;
  }
  if (!cache::valid(*geometry, cfg)) {
    ET_LOG(Error, "ModuleExecutor: the program's layout is unusable");
    return Error::InvalidProgram;
  }

  const auto meta = module->method_meta(method);
  if (!meta.ok()) {
    ET_LOG(Error, "ModuleExecutor: %s has no metadata", method.c_str());
    return meta.error();
  }

  const std::size_t expected_inputs =
      logits_mode == LogitsToKeepMode::Selected ? 3 : 2;
  if (meta->num_inputs() != expected_inputs) {
    ET_LOG(
        Error,
        "ModuleExecutor: %s expects %zu inputs for its logits mode, got %zu",
        method.c_str(),
        expected_inputs,
        meta->num_inputs());
    return Error::InvalidProgram;
  }

  const auto tokens_info = meta->input_tensor_meta(0);
  const auto positions_info = meta->input_tensor_meta(1);
  if (!tokens_info.ok() || !positions_info.ok()) {
    ET_LOG(Error, "ModuleExecutor: %s inputs must be tensors", method.c_str());
    return Error::InvalidProgram;
  }
  const auto token_sizes = tokens_info->sizes();
  const auto position_sizes = positions_info->sizes();
  if (tokens_info->scalar_type() != ::executorch::aten::ScalarType::Long ||
      token_sizes.size() != 2 || token_sizes[0] != 1 || token_sizes[1] <= 0 ||
      positions_info->scalar_type() != ::executorch::aten::ScalarType::Long ||
      position_sizes.size() != 1 || position_sizes[0] != token_sizes[1]) {
    ET_LOG(
        Error,
        "ModuleExecutor: %s must take Long[1, T] tokens and Long[T] positions",
        method.c_str());
    return Error::InvalidProgram;
  }
  if (logits_mode == LogitsToKeepMode::Selected) {
    const auto selector_info = meta->input_tensor_meta(2);
    if (!selector_info.ok() ||
        selector_info->scalar_type() != ::executorch::aten::ScalarType::Long ||
        selector_info->sizes().size() != 1) {
      ET_LOG(
          Error,
          "ModuleExecutor: %s selected logits selector must be rank-one Long",
          method.c_str());
      return Error::InvalidProgram;
    }
  }

  if (meta->num_outputs() == 0) {
    ET_LOG(Error, "ModuleExecutor: %s publishes no outputs", method.c_str());
    return Error::InvalidProgram;
  }
  const auto logits_info = meta->output_tensor_meta(0);
  if (!logits_info.ok()) {
    ET_LOG(Error, "ModuleExecutor: %s has no logits metadata", method.c_str());
    return logits_info.error();
  }
  const auto logits_sizes = logits_info->sizes();
  if (logits_sizes.size() < 2 || logits_sizes[logits_sizes.size() - 1] <= 0 ||
      !is_supported_logits_type(logits_info->scalar_type())) {
    ET_LOG(
        Error,
        "ModuleExecutor: %s logits must have supported dtype and shape [..., vocab]",
        method.c_str());
    return Error::InvalidProgram;
  }
  const auto published_vocab_size = read_vocab_size(*module);
  if (!published_vocab_size.ok()) {
    ET_LOG(
        Error, "ModuleExecutor: invalid get_vocab_size for %s", method.c_str());
    return published_vocab_size.error();
  }
  const auto vocab_size = check_vocab_size(
      *published_vocab_size, logits_sizes[logits_sizes.size() - 1]);
  if (!vocab_size.ok()) {
    ET_LOG(
        Error,
        "ModuleExecutor: invalid get_vocab_size for %s output width",
        method.c_str());
    return vocab_size.error();
  }

  std::string backend_id;
  for (std::size_t i = 0; i < meta->num_backends(); ++i) {
    const auto name = meta->get_backend_name(i);
    if (!name.ok()) {
      ET_LOG(
          Error, "ModuleExecutor: %s has an unnamed delegate", method.c_str());
      return name.error();
    }
    if (backend_id.empty()) {
      backend_id = name.get();
    } else if (backend_id != name.get()) {
      ET_LOG(
          Error,
          "ModuleExecutor: %s spans more than one backend, so which holds the "
          "cache is ambiguous",
          method.c_str());
      return Error::InvalidProgram;
    }
  }
  if (backend_id.empty()) {
    ET_LOG(Error, "ModuleExecutor: %s delegates to nothing", method.c_str());
    return Error::InvalidProgram;
  }

  auto built = cache::CacheFactory::global().build(
      backend_id, cache_kind, *geometry, cfg);
  if (!built.ok()) {
    ET_LOG(
        Error,
        "ModuleExecutor: failed to build backend %s cache %s",
        backend_id.c_str(),
        cache_kind.c_str());
    return built.error();
  }
  std::shared_ptr<cache::Cache> cache = built.get();
  cache::BatchControl* const ctl = cache->as<cache::BatchControl>();
  if (ctl == nullptr) {
    ET_LOG(Error, "ModuleExecutor: the cache carries no sequence identity");
    return Error::InvalidType;
  }
  // Refused here rather than at the session that would not open, so a caller
  // asking for more than the layout holds hears about it once.
  const std::optional<int> seq_limit = ctl->max_seqs();
  if (seq_limit && max_sessions > *seq_limit) {
    ET_LOG(
        Error,
        "ModuleExecutor: %d resident sessions requested, but the layout holds %d",
        max_sessions,
        *seq_limit);
    return Error::InvalidArgument;
  }

  return std::unique_ptr<ModuleExecutor>(new ModuleExecutor(
      std::move(module),
      std::move(cache),
      max_sessions,
      max_session_tokens,
      std::move(backend_id),
      std::move(method),
      *vocab_size,
      token_sizes[1],
      logits_mode));
}

bool ModuleExecutor::initialize() {
  const auto error = load_method_with_cache(
      *module_, method_, backend_id_.c_str(), install_guard_);
  if (error != Error::Ok) {
    ET_LOG(
        Error,
        "ModuleExecutor: could not load %s with cache binding (0x%x)",
        method_.c_str(),
        static_cast<unsigned int>(error));
    return false;
  }
  return true;
}

std::optional<SessionId> ModuleExecutor::open_session() {
  return sessions_.open();
}

void ModuleExecutor::close_session(SessionId session) {
  sessions_.close(session);
}

std::optional<SessionId> ModuleExecutor::clone(
    SessionId source,
    Position upto) {
  return sessions_.clone(source, upto);
}

void ModuleExecutor::set_sampling(
    SessionId session,
    const SamplingParams& params,
    std::optional<std::uint64_t> seed) {
  sessions_.set_sampling(session, params, seed);
}

bool ModuleExecutor::execute(const BatchInput& batch, BatchOutput& out) {
  out.outputs.clear();
  out.outputs.resize(batch.inputs.size());

  const Result<util::PackedStep> step = sessions_.pack(batch);
  if (!step.ok()) {
    return false;
  }

  // A batch wider than the method was traced at runs as several forwards. They
  // go in order, so a slice attends the cells its predecessors wrote, and each
  // input's logits row falls in exactly one of them.
  const int total = static_cast<int>(step->tokens.size());
  for (int off = 0; off < total; off += max_step_tokens_) {
    const int n = std::min(max_step_tokens_, total - off);
    // Placement checks the forward's token count against the declaration, so
    // each slice declares its own.
    if (!ctl_->declare_step(std::vector<std::int32_t>(
            step->seq_ids.begin() + off, step->seq_ids.begin() + off + n))) {
      ET_LOG(Error, "ModuleExecutor: the cache refused a slice of %d", n);
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
    util::SliceRows selected;
    if (logits_to_keep_mode_ == LogitsToKeepMode::Selected) {
      selected = util::select_rows(*step, off, n);
    }

    const int expected_rows = logits_to_keep_mode_ == LogitsToKeepMode::Selected
        ? static_cast<int>(selected.selector.size())
        : n;
    auto result = [&]() -> Result<std::vector<::executorch::runtime::EValue>> {
      if (logits_to_keep_mode_ == LogitsToKeepMode::Selected) {
        auto selector =
            make_tensor_ptr({expected_rows}, std::move(selected.selector));
        return module_->execute(method_, {tokens, positions, selector});
      }
      return module_->execute(method_, {tokens, positions});
    }();
    if (!result.ok()) {
      ET_LOG(
          Error,
          "ModuleExecutor: %s failed with 0x%x",
          method_.c_str(),
          static_cast<unsigned>(result.error()));
      return false;
    }
    if (result->empty() || !result->at(0).isTensor()) {
      ET_LOG(Error, "ModuleExecutor: %s returned no logits", method_.c_str());
      return false;
    }
    // Non-const: the sampler reduces each row in place. Each is read once.
    auto logits = result->at(0).toTensor();
    if (logits.dim() < 2 || logits.size(logits.dim() - 1) != vocab_size_ ||
        logits.numel() !=
            static_cast<std::int64_t>(expected_rows) * vocab_size_) {
      ET_LOG(
          Error,
          "ModuleExecutor: %s returned invalid logits shape for %d rows and vocab %d",
          method_.c_str(),
          expected_rows,
          vocab_size_);
      return false;
    }

    if (logits_to_keep_mode_ == LogitsToKeepMode::Selected) {
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
    } else {
      for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
        const int row = step->logit_indices[i];
        if (row < off || row >= off + n) {
          continue; // another slice's row, or a dropped chunk prediction
        }
        const SessionId session = batch.inputs[i].sid;
        const std::optional<Token> token =
            sessions_.sample(session, logits, row - off);
        if (!token) {
          return false;
        }
        out.outputs[i] = Output{session, {*token}};
      }
    }
  }
  return true;
}

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
