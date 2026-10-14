/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include <executorch/extension/llm/batching/prefix_cache_retention.h>
#include <executorch/extension/llm/batching/runner.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

struct ET_EXPERIMENTAL PrefixMatch {
  Session session;
  std::size_t matched_tokens;
};

struct ET_EXPERIMENTAL PrefixCheckpoint {
  Position position;
  CheckpointMetadata metadata = {};
};

// Absolute periodic boundaries, plus an optional final prompt checkpoint.
inline std::vector<Position> periodic_prefill_breakpoints(
    Position start,
    Position end,
    Position interval,
    bool include_end = false) {
  std::vector<Position> positions;
  if (start < 0 || end <= start || interval <= 0) {
    return positions;
  }
  for (std::int64_t next =
           (static_cast<std::int64_t>(start) / interval + 1) * interval;
       next <= end;
       next += interval) {
    positions.push_back(static_cast<Position>(next));
  }
  if (include_end && (positions.empty() || positions.back() != end)) {
    positions.push_back(end);
  }
  return positions;
}

// Caller-thread policy over retained, idle snapshot sessions. Scope one cache
// to one Runner and immutable model/configuration; identities must describe
// all token and opaque content under those fixed execution semantics.
// Serialize cache calls externally. lookup() and PromptCapture::collect() may
// wait for cloning and must never run in a Runner callback. Keep this object
// outside the Runner: its Sessions keep the runner's internals alive.
//
// Backend numerical parity is separate from snapshot correctness. On MLX,
// cold/warm logits may differ as prefill shapes change. Keep MLX reuse opt-in
// and greedy-only until non-greedy sampling is validated; even greedy token
// parity is not guaranteed.
class ET_EXPERIMENTAL PrefixCache {
 public:
  using EntryView = PrefixCacheEntryView;
  using Entries = PrefixCacheEntries;
  using RetentionPolicy = PrefixCacheRetentionPolicy;

  // One generation's best-effort prompt capture. The wrapped callback owns
  // its state, so moving or discarding this handle does not invalidate it.
  class PromptCapture {
   public:
    PromptCapture() = default;
    PromptCapture(PromptCapture&&) noexcept = default;
    PromptCapture& operator=(PromptCapture&&) noexcept = default;
    PromptCapture(const PromptCapture&) = delete;
    PromptCapture& operator=(const PromptCapture&) = delete;

    // Wrap once for one generation. Enqueue the clone before delivering the
    // first tokens, without waiting on the engine thread. User callback
    // exceptions retain the normal generation-failure behavior.
    GenerationCallback wrap(GenerationCallback on_update = {}) {
      if (!state_) {
        return on_update;
      }
      return [state = state_, on_update = std::move(on_update)](
                 const GenerationUpdate& update) {
        if (!state->attempted && !update.tokens.empty()) {
          state->attempted = true;
#if ET_HAS_EXCEPTIONS
          try {
#endif
            state->snapshot = state->request_clone(
                static_cast<Position>(state->tokens.size()));
#if ET_HAS_EXCEPTIONS
          } catch (const std::bad_alloc&) {
          }
#endif
        }
        if (on_update) {
          on_update(update);
        }
      };
    }

    // Queue without waiting; may run in a Runner callback. Serialize with
    // collect(); at most one clone may be pending. Reusable after collect.
    bool capture_at(Position position, CheckpointMetadata metadata = {}) {
      return state_ && queue_capture(*state_, position, std::move(metadata));
    }

    // The caller owns admission and collection. Each successful acquisition
    // receives one notification: queued=true publishes work for collect();
    // queued=false releases admission after optional capture failed.
    // Optional checkpoint metadata is matched by position; points must
    // increase.
    std::function<void(Position)> observer(
        std::function<bool()> try_acquire,
        std::function<void(bool queued)> on_capture,
        std::vector<PrefixCheckpoint> checkpoints = {}) {
      if (!state_) {
        return {};
      }
      return [state = state_,
              try_acquire = std::move(try_acquire),
              on_capture = std::move(on_capture),
              checkpoints = std::move(checkpoints)](Position position) {
        if (!try_acquire()) {
          return;
        }
        bool queued = false;
#if ET_HAS_EXCEPTIONS
        try {
#endif
          const auto point = std::lower_bound(
              checkpoints.begin(),
              checkpoints.end(),
              position,
              [](const auto& checkpoint, Position value) {
                return checkpoint.position < value;
              });
          queued = queue_capture(
              *state,
              position,
              point != checkpoints.end() && point->position == position
                  ? point->metadata
                  : CheckpointMetadata{});
#if ET_HAS_EXCEPTIONS
        } catch (...) {
          // Optional capture failure leaves the generation usable.
        }
#endif
        on_capture(queued);
      };
    }

    // Caller thread only, after wrap's generation has settled or capture_at
    // has returned. May wait for the clone. The PrefixCache must still live.
    bool collect() {
      auto state = state_;
      if (!state || !state->snapshot.valid()) {
        return false;
      }
#if ET_HAS_EXCEPTIONS
      try {
#endif
        auto metadata = std::move(state->metadata);
        auto snapshot = state->snapshot.get();
        if (!snapshot) {
          return false;
        }
        PrefixIdentity prefix;
        auto remaining = state->captured_tokens;
        for (auto span : state->tokens.spans) {
          if (!remaining) {
            break;
          }
          std::visit(
              [&](auto& s) {
                s.size = std::min(s.size, remaining);
                remaining -= s.size;
              },
              span);
          prefix.spans.push_back(std::move(span));
        }
        return cache_->insert_identity(
            prefix, std::move(*snapshot), std::move(metadata));
#if ET_HAS_EXCEPTIONS
      } catch (const std::bad_alloc&) {
        return false;
      }
#endif
    }

   private:
    friend class PrefixCache;
    struct State {
      std::function<std::future<std::optional<Session>>(Position)>
          request_clone;
      PrefixIdentity tokens;
      std::future<std::optional<Session>> snapshot;
      std::size_t captured_tokens = 0;
      bool attempted = false;
      CheckpointMetadata metadata;
    };

    static bool queue_capture(
        State& state,
        Position position,
        CheckpointMetadata metadata) {
      if (state.snapshot.valid() || position <= 0 ||
          static_cast<std::size_t>(position) > state.tokens.size()) {
        return false;
      }
      state.captured_tokens = static_cast<std::size_t>(position);
      state.snapshot = state.request_clone(position);
      state.metadata = std::move(metadata);
      return true;
    }

    PrefixCache* cache_ = nullptr;
    std::shared_ptr<State> state_;
  };

  explicit PrefixCache(
      std::size_t max_entries,
      RetentionPolicy retention_policy = {})
      : max_entries_(max_entries),
        retention_policy_(
            retention_policy ? std::move(retention_policy)
                             : RetentionPolicy{lru_prefix_cache_retention}) {}

  PrefixCache(const PrefixCache&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;
  PrefixCache(PrefixCache&&) = delete;
  PrefixCache& operator=(PrefixCache&&) = delete;

  // tokens is the complete prompt history, including any reused prefix.
  // Capture binds to the session identity, so the source Session may move.
  // It does not own the source: destroying its owner still requests closure.
  // Disabled caching, invalid input, or allocation refusal gives a no-op
  // capture; it still forwards the wrapped callback.
  PromptCapture capture_prompt(
      const Session& source,
      const std::vector<Token>& tokens,
      CheckpointMetadata metadata = {}) {
    return capture_identity(source, copy_tokens(tokens), std::move(metadata));
  }

  PromptCapture capture_identity(
      const Session& source,
      const PrefixIdentity& tokens,
      CheckpointMetadata metadata = {}) {
    PromptCapture capture;
    if (max_entries_ == 0 || !tokens.size() || !source.valid() ||
        tokens.size() >
            static_cast<std::size_t>(std::numeric_limits<Position>::max())) {
      return capture;
    }
#if ET_HAS_EXCEPTIONS
    try {
#endif
      capture.state_ =
          std::make_shared<PromptCapture::State>(PromptCapture::State{
              source.make_clone_request(),
              tokens,
              {},
              tokens.size(),
              false,
              std::move(metadata)});
      capture.cache_ = this;
#if ET_HAS_EXCEPTIONS
    } catch (const std::bad_alloc&) {
    }
#endif
    return capture;
  }

  // Takes ownership of a successful clone at exactly tokens.size(). The caller
  // supplies its exact committed tokens and must not have generated on the
  // clone. Queue clone_async at a supported boundary before later forwards
  // overwrite it (e.g. from an on_prefill observer), then collect the future
  // and insert here on the caller thread. PromptCapture handles this queueing
  // and collection. Refusal releases the supplied snapshot and leaves existing
  // entries intact.
  // Duplicate identities refresh recency and replace metadata, keeping the
  // retained snapshot. Metadata never affects matching or clone safety.
  bool insert(
      const std::vector<Token>& tokens,
      Session snapshot,
      CheckpointMetadata metadata = {}) {
    return insert_identity(
        copy_tokens(tokens), std::move(snapshot), std::move(metadata));
  }

  bool insert_identity(
      const PrefixIdentity& tokens,
      Session snapshot,
      CheckpointMetadata metadata = {}) {
    if (max_entries_ == 0 || !tokens.size() || !snapshot.valid() ||
        tokens.size() >
            static_cast<std::size_t>(std::numeric_limits<Position>::max()) ||
        snapshot.position() != static_cast<Position>(tokens.size())) {
      return false;
    }
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->tokens.size() == tokens.size() &&
          common_prefix(it->tokens, tokens) == tokens.size()) {
        it->metadata = std::move(metadata);
        entries_.splice(entries_.begin(), entries_, it);
        return true;
      }
    }
#if ET_HAS_EXCEPTIONS
    try {
#endif
      auto victim = entries_.end();
      if (entries_.size() == max_entries_) {
        Entries entries;
        entries.reserve(entries_.size());
        for (const auto& entry : entries_) {
          entries.push_back(EntryView{entry.tokens, entry.metadata});
        }
        std::optional<std::size_t> index;
#if ET_HAS_EXCEPTIONS
        try {
#endif
          index = retention_policy_(EntryView{tokens, metadata}, entries);
#if ET_HAS_EXCEPTIONS
        } catch (...) {
          return false;
        }
#endif
        if (!index || *index >= entries_.size()) {
          return false;
        }
        victim = std::prev(entries_.end(), entries_.size() - *index);
      }
      entries_.push_front(
          Entry{tokens, std::move(snapshot), std::move(metadata)});
      if (victim != entries_.end()) {
        entries_.erase(victim);
      }
      return true;
#if ET_HAS_EXCEPTIONS
    } catch (const std::bad_alloc&) {
      return false;
    }
#endif
  }

  // Longest token match whose boundary clones, breaking ties by recency.
  // Each snapshot is tried once at its longest match; refusal skips it
  // without retrying shorter boundaries. Leave the final requested token for
  // a forward under the new generation's sampling policy.
  // The returned session is independently writable and survives eviction.
  std::optional<PrefixMatch> lookup(const std::vector<Token>& request) {
    return lookup_identity(copy_tokens(request));
  }

  std::optional<PrefixMatch> lookup_identity(const PrefixIdentity& request) {
    if (request.size() < 2 || entries_.empty()) {
      return std::nullopt;
    }
#if ET_HAS_EXCEPTIONS
    try {
#endif
      struct Candidate {
        std::list<Entry>::iterator entry;
        std::size_t matched;
        std::size_t rank;
      };
      std::vector<Candidate> candidates;
      candidates.reserve(entries_.size());
      std::size_t rank = 0;
      for (auto it = entries_.begin(); it != entries_.end(); ++it, ++rank) {
        const auto matched =
            std::min(common_prefix(it->tokens, request), request.size() - 1);
        if (matched > 0) {
          candidates.push_back(Candidate{it, matched, rank});
        }
      }
      std::sort(
          candidates.begin(),
          candidates.end(),
          [](const auto& a, const auto& b) {
            return a.matched != b.matched ? a.matched > b.matched
                                          : a.rank < b.rank;
          });
      for (const auto& candidate : candidates) {
        auto session =
            candidate.entry->snapshot
                .clone_async(static_cast<Position>(candidate.matched))
                .get();
        if (session) {
          entries_.splice(entries_.begin(), entries_, candidate.entry);
          return PrefixMatch{std::move(*session), candidate.matched};
        }
      }
#if ET_HAS_EXCEPTIONS
    } catch (const std::bad_alloc&) {
      return std::nullopt;
    }
#endif
    return std::nullopt;
  }

  // Session destruction queues closes. Runner::shutdown() is the completion
  // boundary; a subsequent open/clone is ordered after these closes.
  void clear() {
    entries_.clear();
  }

  std::size_t size() const {
    return entries_.size();
  }

 private:
  PrefixIdentity copy_tokens(const std::vector<Token>& tokens) const {
    if (!max_entries_ || tokens.empty()) {
      return {};
    }
#if ET_HAS_EXCEPTIONS
    try {
#endif
      return token_identity(std::make_shared<const std::vector<Token>>(tokens));
#if ET_HAS_EXCEPTIONS
    } catch (const std::bad_alloc&) {
      return {};
    }
#endif
  }

  struct Entry {
    PrefixIdentity tokens;
    Session snapshot;
    CheckpointMetadata metadata;
  };

  const std::size_t max_entries_;
  RetentionPolicy retention_policy_;
  std::list<Entry> entries_; // most recently used first
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
