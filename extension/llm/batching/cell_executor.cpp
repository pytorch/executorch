/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/cell_executor.h>

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <random>
#include <utility>

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

using ::executorch::extension::make_tensor_ptr;
using ::executorch::runtime::Error;

namespace {

// Spread one seed over positions so a session's neighbouring tokens draw
// unrelated streams.
std::uint64_t mix(std::uint64_t seed, std::int64_t position) {
  std::uint64_t x =
      seed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(position + 1);
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

std::uint64_t nondeterministic_seed() {
  std::random_device device;
  return (static_cast<std::uint64_t>(device()) << 32) ^ device();
}

} // namespace

std::optional<Step> build_step(
    cache::BatchControl& ctl,
    const BatchInput& batch,
    const std::unordered_map<SessionId, std::int32_t>& seqs,
    int max_session_tokens) {
  Step step;
  const std::size_t total = batch.size();
  step.tokens.reserve(total);
  step.positions.reserve(total);
  step.logit_indices.reserve(batch.inputs.size());

  std::vector<std::int32_t> seq_ids;
  seq_ids.reserve(total);
  // Truncations the batch asks for, held until every input has been checked.
  std::vector<std::pair<std::int32_t, int>> rewinds;
  // Where each sequence stands mid-batch: the cache still reports what it held
  // before the step, so the batch's own writes live here.
  std::unordered_map<std::int32_t, int> cursor;

  for (const Input& input : batch.inputs) {
    const auto seq_it = seqs.find(input.sid);
    if (seq_it == seqs.end()) {
      ET_LOG(Error, "build_step: session %" PRId64 " is not open", input.sid);
      return std::nullopt;
    }
    const std::int32_t seq = seq_it->second;
    if (input.size == 0 || !input.tokens ||
        input.offset + input.size > input.tokens->size()) {
      ET_LOG(
          Error,
          "build_step: session %" PRId64 " gave a slice its tokens do not hold",
          input.sid);
      return std::nullopt;
    }

    const std::int64_t start = static_cast<std::int64_t>(input.position) +
        static_cast<std::int64_t>(input.offset);
    int& at = cursor.try_emplace(seq, ctl.next_pos(seq)).first->second;
    if (start > at) {
      // Positions nothing attended, and nothing later reaches back to fill.
      ET_LOG(
          Error,
          "build_step: session %" PRId64 " starts at %" PRId64
          " over a sequence holding %d",
          input.sid,
          start,
          at);
      return std::nullopt;
    }
    if (start < at) {
      if (start == 0) {
        // Emptying a sequence hands its id back, and the step names it.
        ET_LOG(
            Error,
            "build_step: session %" PRId64 " reopens from the start",
            input.sid);
        return std::nullopt;
      }
      rewinds.emplace_back(seq, static_cast<int>(start));
      at = static_cast<int>(start);
    }

    const std::int64_t end = start + static_cast<std::int64_t>(input.size);
    if (end > max_session_tokens) {
      ET_LOG(
          Error,
          "build_step: session %" PRId64 " reaches %" PRId64 " of %d cells",
          input.sid,
          end,
          max_session_tokens);
      return std::nullopt;
    }

    const Token* slice = input.tokens->data() + input.offset;
    step.tokens.insert(step.tokens.end(), slice, slice + input.size);
    for (std::size_t k = 0; k < input.size; ++k) {
      step.positions.push_back(start + static_cast<std::int64_t>(k));
    }
    seq_ids.insert(seq_ids.end(), input.size, seq);
    at = static_cast<int>(end);
    step.logit_indices.push_back(
        input.produce_output ? static_cast<int>(step.tokens.size()) - 1 : -1);
  }

  for (const auto& [seq, from] : rewinds) {
    if (!ctl.seq_rm(seq, from, std::nullopt)) {
      ET_LOG(Error, "build_step: sequence %d would not truncate", seq);
      return std::nullopt;
    }
  }
  // After the truncations, so the cells they freed count toward admission.
  if (!ctl.declare_step(seq_ids)) {
    ET_LOG(Error, "build_step: the cache turned the step down");
    return std::nullopt;
  }
  return step;
}

CellExecutor::CellExecutor(
    Config config,
    std::unique_ptr<Module> module,
    std::shared_ptr<cache::CacheBase> cache,
    std::unique_ptr<cache::CacheSession> session)
    : config_(std::move(config)),
      session_(std::move(session)),
      cache_(std::move(cache)),
      module_(std::move(module)),
      ctl_(cache_->as_batch_control()) {}

CellExecutor::~CellExecutor() = default;

std::unique_ptr<CellExecutor> CellExecutor::create(
    std::unique_ptr<Module> module,
    Config config) {
  if (module == nullptr) {
    ET_LOG(Error, "CellExecutor: no program");
    return nullptr;
  }
  if (config.max_sessions <= 0 || config.max_session_tokens <= 0) {
    ET_LOG(Error, "CellExecutor: session limits must be positive");
    return nullptr;
  }
  // A batch cannot be refused in part, so the table must hold every session at
  // its bound rather than discover it is short mid-run.
  const std::int64_t needed = static_cast<std::int64_t>(config.max_sessions) *
      config.max_session_tokens;
  if (needed > config.cache.capacity) {
    ET_LOG(
        Error,
        "CellExecutor: %d sessions of %d cells need %" PRId64
        ", capacity is %d",
        config.max_sessions,
        config.max_session_tokens,
        needed,
        config.cache.capacity);
    return nullptr;
  }
  if (!cache::valid(config.cache)) {
    ET_LOG(Error, "CellExecutor: invalid cache config");
    return nullptr;
  }

  if (module->load() != Error::Ok) { // a no-op once the caller has loaded it
    ET_LOG(Error, "CellExecutor: the program did not load");
    return nullptr;
  }

  auto built = cache::CacheBuilderRegistry::global().build(
      config.backend_id, config.cache_kind, config.cache);
  if (!built.ok()) {
    ET_LOG(
        Error,
        "CellExecutor: no %s cache for backend %s",
        config.cache_kind.c_str(),
        config.backend_id.c_str());
    return nullptr;
  }
  std::shared_ptr<cache::CacheBase> cache = built.get();
  if (cache->as_batch_control() == nullptr) {
    ET_LOG(Error, "CellExecutor: the cache carries no sequence identity");
    return nullptr;
  }

  // Checked here rather than at the deferred load: it needs only the config.
  if (config.cache_key_option.size() >=
      ::executorch::runtime::kMaxOptionKeyLength) {
    ET_LOG(
        Error,
        "CellExecutor: option key %s is too long",
        config.cache_key_option.c_str());
    return nullptr;
  }

  auto session =
      std::make_unique<cache::CacheSession>(cache::make_unique_key(), cache);
  return std::unique_ptr<CellExecutor>(new CellExecutor(
      std::move(config),
      std::move(module),
      std::move(cache),
      std::move(session)));
}

bool CellExecutor::ensure_method_loaded() {
  if (method_loaded_) {
    return true;
  }
  // The key is taken as a fixed-width array; create() bounded its length.
  char key[::executorch::runtime::kMaxOptionKeyLength] = {};
  std::memcpy(
      key, config_.cache_key_option.data(), config_.cache_key_option.size());

  ::executorch::runtime::BackendOptions<1> options;
  ::executorch::runtime::LoadBackendOptionsMap options_map;
  if (options.set_option(key, session_->key().c_str()) != Error::Ok ||
      options_map.set_options(config_.backend_id.c_str(), options.view()) !=
          Error::Ok) {
    ET_LOG(Error, "CellExecutor: could not name the cache to the backend");
    return false;
  }
  if (module_->load_method(
          config_.method,
          /*planned_memory=*/nullptr,
          /*event_tracer=*/nullptr,
          &options_map) != Error::Ok) {
    ET_LOG(Error, "CellExecutor: could not load %s", config_.method.c_str());
    return false;
  }
  method_loaded_ = true;
  return true;
}

std::optional<SessionId> CellExecutor::open_session() {
  if (static_cast<int>(seqs_.size()) >= config_.max_sessions) {
    return std::nullopt;
  }
  const std::optional<std::int32_t> seq = ctl_->seq_new();
  if (!seq) {
    return std::nullopt;
  }
  const SessionId session = next_session_++;
  seqs_.emplace(session, *seq);
  return session;
}

void CellExecutor::close_session(SessionId session) {
  const auto it = seqs_.find(session);
  if (it == seqs_.end()) {
    return;
  }
  // Frees the cells and hands the sequence id back. The session id is not.
  ctl_->seq_rm(it->second, 0, std::nullopt);
  seqs_.erase(it);
  sampling_.erase(session);
}

void CellExecutor::set_sampling(
    SessionId session,
    const SamplingParams& params,
    std::optional<std::uint64_t> seed) {
  if (seqs_.count(session) == 0) {
    return;
  }
  sampling_.insert_or_assign(
      session, Sampling{params, seed.value_or(nondeterministic_seed())});
}

bool CellExecutor::execute(const BatchInput& batch, BatchOutput& out) {
  out.outputs.clear();
  out.outputs.resize(batch.inputs.size());

  // Ahead of the step: a refused load must claim no cell.
  if (!ensure_method_loaded()) {
    return false;
  }

  const std::optional<Step> step =
      build_step(*ctl_, batch, seqs_, config_.max_session_tokens);
  if (!step) {
    return false;
  }

  auto tokens =
      make_tensor_ptr({1, static_cast<int>(step->tokens.size())}, step->tokens);
  auto positions = make_tensor_ptr(
      {static_cast<int>(step->positions.size())}, step->positions);
  const auto result = module_->execute(config_.method, {tokens, positions});
  if (!result.ok()) {
    ET_LOG(
        Error,
        "CellExecutor: %s failed with 0x%x",
        config_.method.c_str(),
        static_cast<unsigned>(result.error()));
    return false;
  }
  if (result->empty() || !result->at(0).isTensor()) {
    ET_LOG(
        Error, "CellExecutor: %s returned no logits", config_.method.c_str());
    return false;
  }
  const auto logits = result->at(0).toTensor();

  for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
    const int row = step->logit_indices[i];
    if (row < 0) {
      continue; // a chunk whose prediction is discarded
    }
    const SessionId session = batch.inputs[i].sid;
    // The drawn token lands one past the row that predicted it.
    const std::optional<Token> token =
        sample_row(logits, row, session, step->positions[row] + 1);
    if (!token) {
      return false;
    }
    out.outputs[i] = Output{session, {*token}};
  }
  return true;
}

std::optional<Token> CellExecutor::sample_row(
    const ::executorch::aten::Tensor& logits,
    int row,
    SessionId session,
    std::int64_t position) const {
  const auto policy = sampling_.find(session);
  if (policy == sampling_.end()) {
    ET_LOG(
        Error,
        "CellExecutor: session %" PRId64 " has no sampling policy",
        session);
    return std::nullopt;
  }
  const SamplingParams& params = policy->second.params;
  const auto vocab = logits.size(logits.dim() - 1);
  if (row >= logits.numel() / vocab) {
    ET_LOG(Error, "CellExecutor: logits hold no row %d", row);
    return std::nullopt;
  }

  std::optional<Token> drawn;
  struct {
    [[noreturn]] void fail(Error) {
      ET_CHECK_MSG(false, "CellExecutor: unsupported logits dtype");
    }
  } ctx;
  ET_SWITCH_THREE_TYPES(
      Float,
      Half,
      BFloat16,
      logits.scalar_type(),
      ctx,
      "sample_row",
      CTYPE,
      [&] {
        const CTYPE* begin = logits.const_data_ptr<CTYPE>() + row * vocab;
        if (params.temperature <= 0.0f) {
          drawn = static_cast<Token>(
              std::max_element(begin, begin + vocab) - begin);
          return;
        }
        // The sampler reduces in place, so copy rather than overwrite the
        // model's output.
        std::vector<float> scores(begin, begin + vocab);
        Sampler sampler(
            static_cast<std::int32_t>(vocab),
            params.temperature,
            params.top_p,
            mix(policy->second.seed, position));
        sampler.set_topk(params.top_k);
        drawn = static_cast<Token>(sampler.sample(scores.data()));
      });
  return drawn;
}

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
