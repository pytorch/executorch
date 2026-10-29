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

// Absolute periodic capture boundaries; retention is configured independently.
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

// Prefix snapshots for one Runner and immutable model configuration. Reused
// snapshots retain clone access through their new working owner. Return a
// transferred session with reclaim() before retiring it to preserve its prefix.
// Serialize calls; lookup/collect may block and cannot run in Runner callbacks.
// Reuse may change numerical results as prefill shapes change.
class ET_EXPERIMENTAL PrefixCache {
 public:
  using EntryView = PrefixCacheEntryView;
  using Entries = PrefixCacheEntries;
  using RetentionPolicy = PrefixCacheRetentionPolicy;

  // Best-effort capture; callbacks retain state independently of this handle.
  class PromptCapture {
   public:
    PromptCapture() = default;
    PromptCapture(PromptCapture&&) noexcept = default;
    PromptCapture& operator=(PromptCapture&&) noexcept = default;
    PromptCapture(const PromptCapture&) = delete;
    PromptCapture& operator=(const PromptCapture&) = delete;

    // Wrap once per generation; queue a clone before the first output.
    // User callback exceptions fail generation.
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

    // Metadata points must increase. After successful try_acquire, on_capture
    // reports whether the caller should collect (true) or release admission
    // (false). All callbacks must be nonblocking.
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

    // May block; serialize with capture_at or wait for wrap's generation.
    // The PrefixCache must outlive collection.
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

  // Supply the complete prompt. Keep the source owned until generation ends.
  // Invalid or disabled capture still forwards the wrapped callback.
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

  // Takes an idle clone at exactly tokens.size(); refusal releases it.
  // Duplicate keys refresh recency and metadata and restore borrowed snapshots.
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
    prune();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->tokens.size() == tokens.size() &&
          common_prefix(it->tokens, tokens) == tokens.size()) {
        if (!it->snapshot.can_clone() || !it->transferable) {
          it->snapshot = std::move(snapshot);
          it->borrowed_clone = {};
          it->borrowed_valid = {};
          it->borrowed_identity = nullptr;
          it->transferable = true;
        }
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
          Entry{tokens, std::move(snapshot), std::move(metadata), {}});
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

  // Return a transferred session before destroying it. Evicted entries are not
  // recreated. Only a snapshot returned without generation may transfer again;
  // a continued session can have pending output and must serve via cloning.
  void reclaim(Session session, bool pristine = false) {
    prune();
    for (auto& entry : entries_) {
      if (entry.borrowed_identity == session.state_.get() &&
          !entry.snapshot.can_clone() && session.can_clone()) {
        entry.snapshot = std::move(session);
        entry.borrowed_clone = {};
        entry.borrowed_valid = {};
        entry.borrowed_identity = nullptr;
        entry.transferable = pristine;
        return;
      }
    }
  }

  // Find the longest reusable prefix, breaking ties by recency. Leave the
  // final token for a forward. Clone by default; an opted-in complete match
  // transfers ownership. The returned session survives cache eviction.
  std::optional<PrefixMatch> lookup(
      const std::vector<Token>& request,
      bool reuse_session = false) {
    return lookup_identity(copy_tokens(request), reuse_session);
  }

  // Transfer complete idle snapshots when requested. The cache retains clone
  // access while the working session is alive; partial matches still clone.
  std::optional<PrefixMatch> lookup_identity(
      const PrefixIdentity& request,
      bool reuse_session = false) {
    prune();
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
        auto& entry = *candidate.entry;
        const auto position = static_cast<Position>(candidate.matched);
        if (reuse_session && entry.transferable &&
            entry.snapshot.retains_prefixes() &&
            candidate.matched == entry.tokens.size() &&
            entry.snapshot.position() == position) {
          entry.borrowed_clone = entry.snapshot.make_clone_request();
          entry.borrowed_valid = entry.snapshot.make_valid_check();
          entry.borrowed_identity = entry.snapshot.state_.get();
          auto session = std::move(entry.snapshot);
          entries_.splice(entries_.begin(), entries_, candidate.entry);
          return PrefixMatch{std::move(session), candidate.matched};
        }
        std::optional<Session> session;
        if (entry.snapshot.can_clone()) {
          session = entry.snapshot.clone_async(position).get();
        } else if (entry.borrowed_clone) {
          session = entry.borrowed_clone(position).get();
        }
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
    return std::count_if(
        entries_.begin(), entries_.end(), [](const auto& entry) {
          return entry.snapshot.can_clone() ||
              (entry.borrowed_valid && entry.borrowed_valid());
        });
  }

 private:
  void prune() {
    entries_.remove_if([](const auto& entry) {
      return !entry.snapshot.can_clone() &&
          (!entry.borrowed_valid || !entry.borrowed_valid());
    });
  }

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
    // Bound to the moved session's original identity. Closure invalidates it,
    // including when the executor later recycles the physical row id.
    std::function<std::future<std::optional<Session>>(Position)> borrowed_clone;
    std::function<bool()> borrowed_valid;
    const void* borrowed_identity = nullptr;
    bool transferable = true;
  };

  const std::size_t max_entries_;
  RetentionPolicy retention_policy_;
  std::list<Entry> entries_; // most recently used first
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
