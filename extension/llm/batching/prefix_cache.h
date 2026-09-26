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
#include <functional>
#include <future>
#include <limits>
#include <list>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include <executorch/extension/llm/batching/runner.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

struct ET_EXPERIMENTAL PrefixMatch {
  Session session;
  std::size_t matched_tokens;
};

// Caller-thread policy over retained, idle snapshot sessions. Scope one cache
// to one Runner and immutable model/configuration; token-only keys cannot
// describe external embeddings or changes to adapters/positional semantics.
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
            state->snapshot = state->request_clone();
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

    // Caller thread only, after the generation's wait() has returned. May
    // wait for the queued clone. Returns false if capture/insertion failed
    // or this handle was already collected. The PrefixCache must still live.
    bool collect() {
      auto state = std::move(state_);
      if (!state || !state->snapshot.valid()) {
        return false;
      }
#if ET_HAS_EXCEPTIONS
      try {
#endif
        auto snapshot = state->snapshot.get();
        return snapshot && cache_->insert(state->tokens, std::move(*snapshot));
#if ET_HAS_EXCEPTIONS
      } catch (const std::bad_alloc&) {
        return false;
      }
#endif
    }

   private:
    friend class PrefixCache;
    struct State {
      std::function<std::future<std::optional<Session>>()> request_clone;
      std::vector<Token> tokens;
      std::future<std::optional<Session>> snapshot;
      bool attempted = false;
    };

    PrefixCache* cache_ = nullptr;
    std::shared_ptr<State> state_;
  };

  explicit PrefixCache(std::size_t max_entries) : max_entries_(max_entries) {}

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
      const std::vector<Token>& tokens) {
    PromptCapture capture;
    if (max_entries_ == 0 || tokens.empty() || !source.valid() ||
        tokens.size() >
            static_cast<std::size_t>(std::numeric_limits<Position>::max())) {
      return capture;
    }
#if ET_HAS_EXCEPTIONS
    try {
#endif
      capture.state_ =
          std::make_shared<PromptCapture::State>(PromptCapture::State{
              source.make_clone_request(static_cast<Position>(tokens.size())),
              tokens,
              {},
              false});
      capture.cache_ = this;
#if ET_HAS_EXCEPTIONS
    } catch (const std::bad_alloc&) {
    }
#endif
    return capture;
  }

  // Takes ownership of a successful clone at exactly tokens.size(). The caller
  // supplies its exact committed tokens and must not have generated on the
  // clone. Capture it with clone_async(prompt_end) in the first output
  // callback, before returning to decode; collect the future and insert here
  // later. Refusal releases the supplied snapshot and leaves existing entries
  // intact.
  bool insert(const std::vector<Token>& tokens, Session snapshot) {
    if (max_entries_ == 0 || tokens.empty() || !snapshot.valid() ||
        tokens.size() >
            static_cast<std::size_t>(std::numeric_limits<Position>::max()) ||
        snapshot.position() != static_cast<Position>(tokens.size())) {
      return false;
    }
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->tokens == tokens) {
        entries_.splice(entries_.begin(), entries_, it);
        return true;
      }
    }
#if ET_HAS_EXCEPTIONS
    try {
#endif
      entries_.push_front(Entry{tokens, std::move(snapshot)});
      if (entries_.size() > max_entries_) {
        entries_.pop_back();
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
        const auto limit = std::min(it->tokens.size(), request.size() - 1);
        std::size_t matched = 0;
        while (matched < limit && it->tokens[matched] == request[matched]) {
          ++matched;
        }
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
  struct Entry {
    std::vector<Token> tokens;
    Session snapshot;
  };

  const std::size_t max_entries_;
  std::list<Entry> entries_; // most recently used first
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
