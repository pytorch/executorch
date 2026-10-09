/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/util/session_table.h>

#include <cinttypes>
#include <limits>
#include <new>
#include <random>
#include <utility>

// sampler/util.h switches on dtype with the macros this defines.
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {
namespace util {

using ::executorch::extension::make_tensor_ptr;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace {

// Releases a sequence unless ownership passes to a session.
struct SequenceGuard {
  cache::BatchControl& control;
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

// extension/llm/sampler/sampler.cpp's random_f32: one coin in [0, 1).
float next_coin(std::uint64_t& state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  const auto bits =
      static_cast<std::uint32_t>((state * 0x2545F4914F6CDD1Dull) >> 32);
  return static_cast<float>(bits >> 8) / 16777216.0f;
}

} // namespace

SliceRows select_rows(const PackedStep& step, int offset, int length) {
  SliceRows rows;
  for (std::size_t i = 0; i < step.logit_indices.size(); ++i) {
    const int row = step.logit_indices[i];
    if (row >= offset && row < offset + length) {
      rows.selector.push_back(row - offset);
      rows.inputs.push_back(i);
    }
  }
  if (rows.selector.empty()) {
    rows.selector.push_back(length - 1);
  }
  return rows;
}

SessionTable::SessionTable(
    cache::BatchControl& control,
    int max_sessions,
    int max_session_tokens,
    std::int32_t vocab_size)
    : ctl_(control),
      max_sessions_(max_sessions),
      max_session_tokens_(max_session_tokens),
      vocab_size_(vocab_size) {}

SessionTable::~SessionTable() = default;

std::optional<SessionId> SessionTable::open() {
  if (sessions_.size() >= static_cast<std::size_t>(max_sessions_) ||
      next_session_ == 0) {
    return std::nullopt;
  }
#if ET_HAS_EXCEPTIONS
  try {
#endif
    const std::optional<std::int32_t> seq_id = ctl_.seq_new();
    return seq_id ? publish(*seq_id, 0) : std::nullopt;
#if ET_HAS_EXCEPTIONS
  } catch (const std::bad_alloc&) {
    return std::nullopt;
  }
#endif
}

std::optional<SessionId> SessionTable::publish(
    std::int32_t seq_id,
    Position position) {
  SequenceGuard guard{ctl_, seq_id};
  if (ctl_.pos(seq_id) != position) {
    return std::nullopt;
  }
  const SessionId session = next_session_;
  if (!sessions_.emplace(session, Session{seq_id, nullptr}).second) {
    return std::nullopt;
  }
  guard.owned = false;
  next_session_ =
      session == std::numeric_limits<SessionId>::max() ? 0 : session + 1;
  return session;
}

void SessionTable::close(SessionId session) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end()) {
    return;
  }
  ctl_.seq_rm(it->second.seq_id);
  sessions_.erase(it);
}

std::optional<SessionId> SessionTable::clone(SessionId source, Position upto) {
  const auto it = sessions_.find(source);
  if (it == sessions_.end() || upto < 0 || upto > max_session_tokens_ ||
      sessions_.size() >= static_cast<std::size_t>(max_sessions_) ||
      next_session_ == 0) {
    return std::nullopt;
  }
#if ET_HAS_EXCEPTIONS
  try {
#endif
    if (upto > ctl_.pos(it->second.seq_id)) {
      return std::nullopt;
    }
    const auto seq_id = ctl_.seq_clone(it->second.seq_id, upto);
    return seq_id ? publish(*seq_id, upto) : std::nullopt;
#if ET_HAS_EXCEPTIONS
  } catch (const std::bad_alloc&) {
    return std::nullopt;
  }
#endif
}

void SessionTable::set_sampling(
    SessionId session,
    const SamplingParams& params,
    std::optional<std::uint64_t> seed) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end()) {
    return;
  }
  // One sampler per generation, carrying its own generator state from here on.
  const std::uint64_t rng_seed = seed.value_or(nondeterministic_seed());
  it->second.sampler = std::make_unique<Sampler>(
      vocab_size_, params.temperature, params.top_p, rng_seed);
  it->second.sampler->set_topk(params.top_k);
  it->second.params = params;
  it->second.rng_state = rng_seed;
}

std::optional<DeviceSamplingRow> SessionTable::device_sampling(
    SessionId session) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end() || it->second.sampler == nullptr) {
    ET_LOG(
        Error, "sample: session %" PRId64 " has no sampling policy", session);
    return std::nullopt;
  }
  const SamplingParams& params = it->second.params;
  // Sampler resets a negative temperature to 0.
  const float temperature = params.temperature > 0.0f ? params.temperature : 0.0f;
  const float coin =
      temperature > 0.0f ? next_coin(it->second.rng_state) : 0.0f;
  return DeviceSamplingRow{
      temperature, params.top_p, static_cast<float>(params.top_k), coin};
}

bool SessionTable::greedy(SessionId session) const {
  const auto it = sessions_.find(session);
  return it != sessions_.end() && !(it->second.params.temperature > 0.0f);
}

Result<PackedStep> SessionTable::pack(const BatchInput& batch) {
  PackedStep step;
  const std::size_t total = batch.size();
  step.tokens.reserve(total);
  step.positions.reserve(total);
  step.seq_ids.reserve(total);
  step.logit_indices.reserve(batch.inputs.size());
  // Truncations the batch asks for, held until every input has been checked.
  std::vector<std::pair<std::int32_t, int>> rewinds;
  // Where each sequence stands mid-batch: the cache still reports what it held
  // before the step, so the batch's own writes live here.
  std::unordered_map<std::int32_t, int> cursor;

  for (const Input& input : batch.inputs) {
    const auto seq_it = sessions_.find(input.sid);
    if (seq_it == sessions_.end()) {
      ET_LOG(Error, "pack: session %" PRId64 " is not open", input.sid);
      return Error::InvalidArgument;
    }
    const std::int32_t seq_id = seq_it->second.seq_id;
    const auto* tokens = std::get_if<TokenInputPtr>(&input.payload);
    const auto* token_input = tokens && *tokens ? tokens->get() : nullptr;
    if (input.size == 0 || !token_input || input.offset > token_input->size() ||
        input.size > token_input->size() - input.offset) {
      ET_LOG(
          Error,
          "pack: session %" PRId64 " gave a slice its tokens do not hold",
          input.sid);
      return Error::InvalidArgument;
    }

    const std::int64_t start = static_cast<std::int64_t>(input.position) +
        static_cast<std::int64_t>(input.offset);
    const auto [cursor_it, first_for_seq] =
        cursor.try_emplace(seq_id, ctl_.pos(seq_id));
    int& at = cursor_it->second;
    if (start > at) {
      // Positions nothing attended, and nothing later reaches back to fill.
      ET_LOG(
          Error,
          "pack: session %" PRId64 " starts at %" PRId64
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
            "pack: session %" PRId64 " overlaps its earlier input",
            input.sid);
        return Error::InvalidArgument;
      }
      if (start == 0) {
        // Emptying a sequence hands its id back, and the step names it.
        ET_LOG(
            Error,
            "pack: session %" PRId64 " reopens from the start",
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
          "pack: session %" PRId64 " reaches %" PRId64 " of %d cells",
          input.sid,
          end,
          max_session_tokens_);
      return Error::OutOfResources;
    }

    const Token* slice = token_input->data() + input.offset;
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
    if (!ctl_.rewind(seq_id, from)) {
      ET_LOG(Error, "pack: sequence %d would not truncate", seq_id);
      return Error::Internal;
    }
  }
  return step;
}

std::optional<Token> SessionTable::sample(
    SessionId session,
    ::executorch::aten::Tensor& logits,
    int row) {
  const auto it = sessions_.find(session);
  if (it == sessions_.end() || it->second.sampler == nullptr) {
    ET_LOG(
        Error, "sample: session %" PRId64 " has no sampling policy", session);
    return std::nullopt;
  }
  if (row < 0 || row >= logits.numel() / vocab_size_) {
    ET_LOG(Error, "sample: logits hold no row %d", row);
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

} // namespace util
} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
