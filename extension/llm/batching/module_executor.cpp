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
#include <random>
#include <utility>

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/backend_options_map.h>
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

std::uint64_t nondeterministic_seed() {
  std::random_device device;
  return device();
}

} // namespace

Result<ModuleExecutor::Step> ModuleExecutor::build_step(
    const BatchInput& batch) {
  // Flatten the batch and truncate whatever it reopens; execute() declares each
  // slice to the cache as it runs it. A per-sequence cursor carries the batch's
  // own writes, so consecutive chunks of one prompt abut and only the first can
  // reopen committed ground. Every input is checked before any is truncated, so
  // a refusal leaves the cache untouched.
  Step step;
  const std::size_t total = batch.size();
  step.tokens.reserve(total);
  step.positions.reserve(total);
  step.logit_indices.reserve(batch.inputs.size());

  step.seq_ids.reserve(total);
  // Truncations the batch asks for, held until every input has been checked.
  std::vector<std::pair<std::int32_t, int>> rewinds;
  // Where each sequence stands mid-batch: the cache still reports what it held
  // before the step, so the batch's own writes live here.
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
      // Positions nothing attended, and nothing later reaches back to fill.
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
        // Its predecessor in this batch has already been laid down, so a
        // rewind now would truncate committed cells for a step whose
        // positions repeat and cannot be placed.
        ET_LOG(
            Error,
            "build_step: session %" PRId64 " overlaps its earlier input",
            input.sid);
        return Error::InvalidArgument;
      }
      if (start == 0) {
        // Emptying a sequence hands its id back, and the step names it.
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
    }
    for (std::size_t k = 0; k < input.size; ++k) {
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
      max_sessions_(max_sessions),
      max_session_tokens_(max_session_tokens),
      backend_id_(std::move(backend_id)),
      method_(std::move(method)),
      vocab_size_(vocab_size),
      max_step_tokens_(max_step_tokens),
      logits_to_keep_mode_(logits_to_keep_mode) {}

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
    ET_LOG(Error, "ModuleExecutor: session limits must be positive");
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
        "ModuleExecutor: backend %s registers no %s cache",
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
        "ModuleExecutor: %d sessions asked of a cache holding %d",
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
  // The delegate resolves the cache from this key while the method loads.
  ::executorch::runtime::BackendOptions<1> options;
  ::executorch::runtime::LoadBackendOptionsMap options_map;
  if (install_guard_.set_option(options) != Error::Ok ||
      options_map.set_options(backend_id_.c_str(), options.view()) !=
          Error::Ok) {
    ET_LOG(Error, "ModuleExecutor: could not name the cache to the backend");
    return false;
  }
  if (module_->load_method(
          method_,
          /*planned_memory=*/nullptr,
          /*event_tracer=*/nullptr,
          &options_map) != Error::Ok) {
    ET_LOG(Error, "ModuleExecutor: could not load %s", method_.c_str());
    return false;
  }
  return true;
}

std::optional<SessionId> ModuleExecutor::open_session() {
  if (static_cast<int>(sessions_.size()) >= max_sessions_) {
    return std::nullopt;
  }
  const std::optional<std::int32_t> seq_id = ctl_->seq_new();
  if (!seq_id) {
    return std::nullopt;
  }
  const SessionId session = next_session_++;
  sessions_.emplace(session, SessionState{*seq_id, nullptr});
  return session;
}

void ModuleExecutor::close_session(SessionId session) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end()) {
    return;
  }
  // Frees the cells and hands the sequence id back. The session id is not.
  ctl_->seq_rm(it->second.seq_id);
  sessions_.erase(it);
}

void ModuleExecutor::set_sampling(
    SessionId session,
    const SamplingParams& params,
    std::optional<std::uint64_t> seed) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end()) {
    return;
  }
  // One sampler per generation, carrying its own generator state from here on.
  it->second.sampler = std::make_unique<Sampler>(
      vocab_size_,
      params.temperature,
      params.top_p,
      seed.value_or(nondeterministic_seed()));
  it->second.sampler->set_topk(params.top_k);
}

bool ModuleExecutor::execute(const BatchInput& batch, BatchOutput& out) {
  out.outputs.clear();
  out.outputs.resize(batch.inputs.size());

  const Result<Step> step = build_step(batch);
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
    std::vector<std::int64_t> selector_values;
    std::vector<std::size_t> selected_inputs;
    if (logits_to_keep_mode_ == LogitsToKeepMode::Selected) {
      for (std::size_t i = 0; i < step->logit_indices.size(); ++i) {
        const int row = step->logit_indices[i];
        if (row >= off && row < off + n) {
          selector_values.push_back(row - off);
          selected_inputs.push_back(i);
        }
      }
      if (selector_values.empty()) {
        selector_values.push_back(n - 1);
      }
    }

    const int expected_rows = logits_to_keep_mode_ == LogitsToKeepMode::Selected
        ? static_cast<int>(selector_values.size())
        : n;
    auto result = [&]() -> Result<std::vector<::executorch::runtime::EValue>> {
      if (logits_to_keep_mode_ == LogitsToKeepMode::Selected) {
        auto selector =
            make_tensor_ptr({expected_rows}, std::move(selector_values));
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
    } else {
      for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
        const int row = step->logit_indices[i];
        if (row < off || row >= off + n) {
          continue; // another slice's row, or a dropped chunk prediction
        }
        const SessionId session = batch.inputs[i].sid;
        const std::optional<Token> token =
            sample_row(logits, row - off, session);
        if (!token) {
          return false;
        }
        out.outputs[i] = Output{session, {*token}};
      }
    }
  }
  return true;
}

std::optional<Token> ModuleExecutor::sample_row(
    ::executorch::aten::Tensor& logits,
    int row,
    SessionId session) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end() || it->second.sampler == nullptr) {
    ET_LOG(
        Error,
        "ModuleExecutor: session %" PRId64 " has no sampling policy",
        session);
    return std::nullopt;
  }
  if (row >= logits.numel() / vocab_size_) {
    ET_LOG(Error, "ModuleExecutor: logits hold no row %d", row);
    return std::nullopt;
  }
  // A one-row view over the model's own output: sample_from_logits reduces in
  // place and reads the last dimension.
  auto one_row = make_tensor_ptr(
      {vocab_size_},
      static_cast<std::uint8_t*>(logits.mutable_data_ptr()) +
          static_cast<std::size_t>(row) * vocab_size_ *
              ::executorch::runtime::elementSize(logits.scalar_type()),
      logits.scalar_type());
  return static_cast<Token>(sample_from_logits(*one_row, *it->second.sampler));
}

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
