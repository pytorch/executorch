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

#include <executorch/extension/llm/cache/cache_et.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/llm/sampler/util.h>
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

// The backend-load option the delegate resolves the cache through; the
// registry's rendezvous convention, not a per-backend name.
constexpr char kCacheKeyOption[] = "cache_key";

std::uint64_t nondeterministic_seed() {
  std::random_device device;
  return (static_cast<std::uint64_t>(device()) << 32) ^ device();
}

} // namespace

std::optional<Step> build_step(
    cache::BatchControl& ctl,
    const BatchInput& batch,
    const std::unordered_map<SessionId, SessionInfo>& sessions,
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
    const auto seq_it = sessions.find(input.sid);
    if (seq_it == sessions.end()) {
      ET_LOG(Error, "build_step: session %" PRId64 " is not open", input.sid);
      return std::nullopt;
    }
    const std::int32_t seq = seq_it->second.seq;
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
    std::unique_ptr<Module> module,
    std::shared_ptr<cache::CacheBase> cache,
    std::unique_ptr<cache::CacheSession> session,
    int max_sessions,
    int max_session_tokens,
    std::string backend_id,
    std::string method,
    std::int32_t vocab_size)
    : session_(std::move(session)),
      cache_(std::move(cache)),
      module_(std::move(module)),
      ctl_(cache_->as_batch_control()),
      max_sessions_(max_sessions),
      max_session_tokens_(max_session_tokens),
      backend_id_(std::move(backend_id)),
      method_(std::move(method)),
      vocab_size_(vocab_size) {}

CellExecutor::~CellExecutor() = default;

std::unique_ptr<CellExecutor> CellExecutor::create(
    std::unique_ptr<Module> module,
    int max_sessions,
    int max_session_tokens,
    int kv_dtype,
    std::string backend_id,
    int initial_capacity,
    std::string method) {
  if (module == nullptr) {
    ET_LOG(Error, "CellExecutor: no program");
    return nullptr;
  }
  if (max_sessions <= 0 || max_session_tokens <= 0) {
    ET_LOG(Error, "CellExecutor: session limits must be positive");
    return nullptr;
  }
  if (module->load() != Error::Ok) { // a no-op once the caller has loaded it
    ET_LOG(Error, "CellExecutor: the program did not load");
    return nullptr;
  }

  auto cfg = cache::et::config_from_program(*module);
  if (!cfg.ok()) {
    return nullptr;
  }
  cfg->capacity = max_sessions * max_session_tokens;
  cfg->kv_dtype = kv_dtype;
  if (initial_capacity >= 0) {
    cfg->initial_capacity = initial_capacity;
  }
  if (!cache::valid(*cfg)) {
    ET_LOG(Error, "CellExecutor: the program's layout is unusable");
    return nullptr;
  }

  auto built =
      cache::CacheBuilderRegistry::global().build(backend_id, "cell", *cfg);
  if (!built.ok()) {
    ET_LOG(
        Error,
        "CellExecutor: no cell cache for backend %s",
        backend_id.c_str());
    return nullptr;
  }
  std::shared_ptr<cache::CacheBase> cache = built.get();
  if (cache->as_batch_control() == nullptr) {
    ET_LOG(Error, "CellExecutor: the cache carries no sequence identity");
    return nullptr;
  }

  const auto meta = module->method_meta(method);
  if (!meta.ok() || meta->num_outputs() == 0) {
    ET_LOG(Error, "CellExecutor: %s publishes no outputs", method.c_str());
    return nullptr;
  }
  const auto logits_info = meta->output_tensor_meta(0);
  if (!logits_info.ok() || logits_info->sizes().empty()) {
    ET_LOG(Error, "CellExecutor: %s has no logits shape", method.c_str());
    return nullptr;
  }
  const auto logits_sizes = logits_info->sizes();

  auto session =
      std::make_unique<cache::CacheSession>(cache::make_unique_key(), cache);
  return std::unique_ptr<CellExecutor>(new CellExecutor(
      std::move(module),
      std::move(cache),
      std::move(session),
      max_sessions,
      max_session_tokens,
      std::move(backend_id),
      std::move(method),
      logits_sizes[logits_sizes.size() - 1]));
}

bool CellExecutor::start() {
  if (method_loaded_) {
    return true;
  }
  // The key is taken as a fixed-width array; create() bounded its length.
  char key[::executorch::runtime::kMaxOptionKeyLength] = {};
  std::memcpy(key, kCacheKeyOption, sizeof(kCacheKeyOption) - 1);

  ::executorch::runtime::BackendOptions<1> options;
  ::executorch::runtime::LoadBackendOptionsMap options_map;
  if (options.set_option(key, session_->key().c_str()) != Error::Ok ||
      options_map.set_options(backend_id_.c_str(), options.view()) !=
          Error::Ok) {
    ET_LOG(Error, "CellExecutor: could not name the cache to the backend");
    return false;
  }
  if (module_->load_method(
          method_,
          /*planned_memory=*/nullptr,
          /*event_tracer=*/nullptr,
          &options_map) != Error::Ok) {
    ET_LOG(Error, "CellExecutor: could not load %s", method_.c_str());
    return false;
  }
  method_loaded_ = true;
  return true;
}

std::optional<SessionId> CellExecutor::open_session() {
  if (static_cast<int>(sessions_.size()) >= max_sessions_) {
    return std::nullopt;
  }
  const std::optional<std::int32_t> seq = ctl_->seq_new();
  if (!seq) {
    return std::nullopt;
  }
  const SessionId session = next_session_++;
  sessions_.emplace(session, SessionInfo{*seq, nullptr});
  return session;
}

void CellExecutor::close_session(SessionId session) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end()) {
    return;
  }
  // Frees the cells and hands the sequence id back. The session id is not.
  ctl_->seq_rm(it->second.seq, 0, std::nullopt);
  sessions_.erase(it);
}

void CellExecutor::set_sampling(
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

bool CellExecutor::execute(const BatchInput& batch, BatchOutput& out) {
  out.outputs.clear();
  out.outputs.resize(batch.inputs.size());

  if (!method_loaded_) {
    ET_LOG(Error, "CellExecutor: execute() before start()");
    return false;
  }

  const std::optional<Step> step =
      build_step(*ctl_, batch, sessions_, max_session_tokens_);
  if (!step) {
    return false;
  }

  auto tokens =
      make_tensor_ptr({1, static_cast<int>(step->tokens.size())}, step->tokens);
  auto positions = make_tensor_ptr(
      {static_cast<int>(step->positions.size())}, step->positions);
  auto result = module_->execute(method_, {tokens, positions});
  if (!result.ok()) {
    ET_LOG(
        Error,
        "CellExecutor: %s failed with 0x%x",
        method_.c_str(),
        static_cast<unsigned>(result.error()));
    return false;
  }
  if (result->empty() || !result->at(0).isTensor()) {
    ET_LOG(Error, "CellExecutor: %s returned no logits", method_.c_str());
    return false;
  }
  // Non-const: the sampler reduces each row in place. Each is read once.
  auto logits = result->at(0).toTensor();

  for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
    const int row = step->logit_indices[i];
    if (row < 0) {
      continue; // a chunk whose prediction is discarded
    }
    const SessionId session = batch.inputs[i].sid;
    const std::optional<Token> token = sample_row(logits, row, session);
    if (!token) {
      return false;
    }
    out.outputs[i] = Output{session, {*token}};
  }
  return true;
}

std::optional<Token> CellExecutor::sample_row(
    ::executorch::aten::Tensor& logits,
    int row,
    SessionId session) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end() || it->second.sampler == nullptr) {
    ET_LOG(
        Error,
        "CellExecutor: session %" PRId64 " has no sampling policy",
        session);
    return std::nullopt;
  }
  if (row >= logits.numel() / vocab_size_) {
    ET_LOG(Error, "CellExecutor: logits hold no row %d", row);
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
