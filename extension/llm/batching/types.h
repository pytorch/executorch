/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The vocabulary shared by the runner, the scheduler, and the executor.
//
// An Input is one slice of work for one session, either a decode token or one
// chunk of a prompt, never a whole generation. A Task is an Input plus the
// scheduling identity used to order and cancel it.

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

using Token ET_EXPERIMENTAL = std::uint64_t;
using SessionId ET_EXPERIMENTAL = std::int64_t;
using Position ET_EXPERIMENTAL = std::int32_t;
// Wide enough that a monotonically issued id cannot wrap in any realistic
// lifetime, so ids never have to be recycled.
using TaskId ET_EXPERIMENTAL = std::int64_t;

using TokenInputPtr ET_EXPERIMENTAL = std::shared_ptr<const std::vector<Token>>;
using ContentKey ET_EXPERIMENTAL = std::array<std::uint8_t, 32>;

struct ET_EXPERIMENTAL TokenSpan {
  TokenInputPtr tokens;
  std::size_t offset;
  std::size_t size;
};
struct ET_EXPERIMENTAL OpaqueSpan {
  ContentKey key;
  std::size_t offset;
  std::size_t size;
};
using PrefixSpan ET_EXPERIMENTAL = std::variant<TokenSpan, OpaqueSpan>;

// Independent immutable metadata, never an owner of execution/image backing.
// Keys describe content under one immutable model/configuration.
struct ET_EXPERIMENTAL PrefixIdentity {
  std::vector<PrefixSpan> spans;

  // Zero denotes empty or invalid metadata. Validate before comparison/use.
  std::size_t size() const {
    std::size_t total = 0;
    for (const auto& span : spans) {
      const bool valid = std::visit(
          [&](const auto& s) {
            if (!s.size ||
                s.offset > std::numeric_limits<std::size_t>::max() - s.size ||
                total > std::numeric_limits<std::size_t>::max() - s.size) {
              return false;
            }
            total += s.size;
            return true;
          },
          span);
      if (!valid) {
        return 0;
      }
      if (const auto* s = std::get_if<TokenSpan>(&span)) {
        if (!s->tokens || s->offset + s->size > s->tokens->size()) {
          return 0;
        }
      }
    }
    return total;
  }
};
using PrefixIdentityPtr ET_EXPERIMENTAL = std::shared_ptr<const PrefixIdentity>;

ET_EXPERIMENTAL inline PrefixIdentity token_identity(TokenInputPtr tokens) {
  return PrefixIdentity{{TokenSpan{tokens, 0, tokens ? tokens->size() : 0}}};
}

// Compare positions, not span boundaries. Opaque offsets are content-relative.
ET_EXPERIMENTAL inline std::size_t common_prefix(
    const PrefixIdentity& a,
    const PrefixIdentity& b) {
  if (!a.size() || !b.size()) {
    return 0;
  }
  std::size_t i = 0, j = 0, x = 0, y = 0, matched = 0;
  while (i < a.spans.size() && j < b.spans.size()) {
    const auto& left = a.spans[i];
    const auto& right = b.spans[j];
    if (left.index() != right.index()) {
      break;
    }
    const auto n = std::min(
        std::visit([](const auto& s) { return s.size; }, left) - x,
        std::visit([](const auto& s) { return s.size; }, right) - y);
    if (const auto* l = std::get_if<TokenSpan>(&left)) {
      const auto& r = std::get<TokenSpan>(right);
      for (std::size_t k = 0; k < n; ++k) {
        if ((*l->tokens)[l->offset + x + k] != (*r.tokens)[r.offset + y + k]) {
          return matched + k;
        }
      }
    } else {
      const auto& opaque = std::get<OpaqueSpan>(left);
      const auto& r = std::get<OpaqueSpan>(right);
      if (opaque.key != r.key || opaque.offset + x != r.offset + y) {
        break;
      }
    }
    matched += n;
    x += n;
    y += n;
    if (x == std::visit([](const auto& s) { return s.size; }, left)) {
      ++i;
      x = 0;
    }
    if (y == std::visit([](const auto& s) { return s.size; }, right)) {
      ++j;
      y = 0;
    }
  }
  return matched;
}

// Opaque owned backing, compatible with its consuming executor as a caller
// precondition. Logical contents, size and layout are fixed before scheduling
// and immutable for the backing's lifetime. Backend-private lazy caches may
// only be mutated on the consuming Runner's engine thread; sharing mutable
// caches across Runners requires backend synchronization.
class ET_EXPERIMENTAL PreparedInput {
 public:
  // The last owner may release on any thread, after the executor is gone.
  // Destruction must be thread-safe and independent of the executor's lifetime.
  virtual ~PreparedInput() = default;
  // Stable, process-local identity for the concrete backing type. Use the
  // address of one canonical non-const object with static storage per type
  // (e.g. static char kKind), so identical constants cannot be folded together.
  // Producers and consumers must share this address across shared-library
  // boundaries. This does not identify weights or devices.
  virtual const void* kind() const = 0;
  // Number of decoder positions in this view, not bytes.
  virtual std::size_t size() const = 0;
  virtual Token last_prompt_token() const = 0;
  // Same concrete type, shared backing, view-relative executor offsets.
  // The view ends at the original end and keeps last_prompt_token unchanged.
  // Invalid or empty suffixes return null.
  virtual std::shared_ptr<const PreparedInput> suffix(
      std::size_t start) const = 0;
  // Optional; when supplied, must cover exactly size() positions.
  virtual PrefixIdentityPtr prefix_identity() const {
    return nullptr;
  }
};
using PreparedInputPtr ET_EXPERIMENTAL = std::shared_ptr<const PreparedInput>;
// Prepared prompts are opaque; generated token feedback needs no preparation.
using InputPayload ET_EXPERIMENTAL =
    std::variant<TokenInputPtr, PreparedInputPtr>;

// Sampling policy for a generation. Installed on the session before its tasks
// are submitted, so it does not ride on every Input.
struct ET_EXPERIMENTAL SamplingParams {
  float temperature = 0.0f;
  float top_p = 1.0f;
  std::int32_t top_k = 0;
};

struct ET_EXPERIMENTAL Input {
  SessionId sid;
  bool produce_output;

  // The selected slice is payload[offset : offset + size]. It starts at the
  // absolute logical position `position + offset`; `position` is the base of
  // the complete backing, not of the slice.
  size_t offset;
  size_t size;

  InputPayload payload;
  Position position;
};

struct ET_EXPERIMENTAL Output {
  SessionId sid;

  // What this input produced: usually one token, can be more than one for
  // speculative decoding
  std::vector<Token> tokens;
};

struct ET_EXPERIMENTAL Task {
  TaskId tid;
  bool cancelled;
  Input input;
  bool is_decode;
};

struct ET_EXPERIMENTAL BatchInput {
  std::vector<Input> inputs;
  size_t size() const {
    size_t sz = 0;
    for (const auto& i : inputs) {
      sz += i.size;
    }
    return sz;
  }
};

// The executor's view of a batch: the Inputs, without the tid, cancelled flag,
// and is_decode that only the runner and scheduler use.
//
// Moves each Input out of its Task, preserving task order, so outputs[i]
// answers batch.inputs[i].
ET_EXPERIMENTAL inline BatchInput to_batch_input(std::vector<Task>& tasks) {
  BatchInput batch;
  batch.inputs.reserve(tasks.size());
  for (Task& t : tasks) {
    batch.inputs.push_back(std::move(t.input));
  }
  return batch;
}

struct ET_EXPERIMENTAL BatchOutput {
  std::vector<std::optional<Output>> outputs;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
