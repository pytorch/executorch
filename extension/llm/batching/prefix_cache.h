/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
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
  std::uint64_t cache_entry_id = 0;
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

// Caller-owned snapshots for one Runner and immutable model configuration.
// Serialize calls; lookup/collect may block and cannot run in Runner callbacks.
// Reuse may change numerical results as prefill shapes change.
class ET_EXPERIMENTAL PrefixCache {
 public:
  using EntryView = PrefixCacheEntryView;
  using Entries = PrefixCacheEntries;
  using RetentionPolicy = PrefixCacheRetentionPolicy;
  enum class MatchDecision { Use, Skip, Stop };
  struct FindResult {
    std::optional<PrefixMatch> match;
    bool stopped = false;
  };
  // One find_identity() request for find_identities(). The request and
  // selector must outlive the call.
  struct IdentityQuery {
    const PrefixIdentity* request = nullptr;
    std::size_t longer_than = 0;
    std::size_t at_most = std::numeric_limits<std::size_t>::max();
    std::function<MatchDecision(std::size_t)> select;
  };
  struct IdentityResult : FindResult {
    // Set when this query's selector or clone request threw anything other
    // than std::bad_alloc; the other queries are unaffected.
    std::exception_ptr error;
  };

  // Best-effort capture; callbacks retain state independently of this handle.
  class PromptCapture {
   public:
    PromptCapture() = default;
    PromptCapture(PromptCapture&&) noexcept = default;
    PromptCapture& operator=(PromptCapture&&) noexcept = default;
    PromptCapture(const PromptCapture&) = delete;
    PromptCapture& operator=(const PromptCapture&) = delete;

    // Wrap once per generation; queue a clone before the first output.
    // on_ready runs at most once, from the first update at which collect()
    // would not block: once that clone has settled, or immediately if none was
    // queued. If generation ends first, collect after it ends. on_ready must
    // not block or throw. User callback exceptions fail generation. An inactive
    // capture returns on_update unchanged.
    GenerationCallback wrap(
        GenerationCallback on_update = {},
        std::function<void()> on_ready = {}) {
      if (!state_) {
        return on_update;
      }
      return [state = state_,
              on_update = std::move(on_update),
              on_ready = std::move(on_ready)](const GenerationUpdate& update) {
        if (!state->attempted && !update.tokens.empty()) {
          state->attempted = true;
#if ET_HAS_EXCEPTIONS
          try {
#endif
            state->snapshot = state->request_clone(
                static_cast<Position>(state->tokens.size()));
#if ET_HAS_EXCEPTIONS
          } catch (...) {
            // Optional capture failure leaves generation usable.
          }
#endif
        }
        // After notification the collector owns the future.
        if (on_ready && state->attempted && !state->notified &&
            (!state->snapshot.valid() ||
             state->snapshot.wait_for(std::chrono::seconds(0)) ==
                 std::future_status::ready)) {
          state->notified = true;
          on_ready();
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

    // May block; serialize with capture_at. With wrap, collect on another
    // thread after on_ready, or wait for generation to finish.
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
      bool notified = false;
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
  // Duplicate keys refresh recency and metadata, keeping the existing snapshot.
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
    if (touch_identity(tokens)) {
      entries_.front().metadata = std::move(metadata);
      return true;
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
      entries_.push_front(Entry{
          tokens, std::move(snapshot), std::move(metadata), next_entry_id_++});
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

  // Clone the longest reusable prefix, breaking ties by recency. Leave the
  // final token for a forward. The returned session survives cache eviction.
  std::optional<PrefixMatch> lookup(const std::vector<Token>& request) {
    return lookup_identity(copy_tokens(request));
  }

  // Refresh recency as inserting this exact identity again would, without a
  // clone.
  bool touch_identity(const PrefixIdentity& tokens) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->tokens.size() == tokens.size() &&
          common_prefix(it->tokens, tokens) == tokens.size()) {
        entries_.splice(entries_.begin(), entries_, it);
        return true;
      }
    }
    return false;
  }

  std::optional<PrefixMatch> lookup_identity(const PrefixIdentity& request) {
    auto result = find_identity(request);
    if (result.match) {
      touch(*result.match);
    }
    return std::move(result.match);
  }

  // Select candidates before cloning, without refreshing recency. Call
  // touch() after adopting the match. The selector must not mutate this cache.
  // Matched lengths are capped at request.size() - 1 and must lie in
  // (longer_than, at_most]: the lower bound is exclusive, the upper inclusive.
  // Candidates arrive in descending length order, with repeated lengths
  // allowed. Use attempts a clone and continues if refused; Skip tries the next
  // candidate without cloning. Stop returns stopped=true, distinct from an
  // exhausted search. Allocation failure
  // ends the search as a miss; other exceptions propagate.
  FindResult find_identity(
      const PrefixIdentity& request,
      std::size_t longer_than = 0,
      std::size_t at_most = std::numeric_limits<std::size_t>::max(),
      const std::function<MatchDecision(std::size_t)>& select = {}) {
    if (request.size() < 2 || entries_.empty()) {
      return {};
    }
#if ET_HAS_EXCEPTIONS
    try {
#endif
      std::vector<IdentityQuery> query(1);
      query.front() = {&request, longer_than, at_most, {}};
      if (select) {
        query.front().select = [&select](std::size_t matched) {
          return select(matched);
        };
      }
      auto found = find_identities(query);
#if ET_HAS_EXCEPTIONS
      if (found.front().error) {
        std::rethrow_exception(found.front().error);
      }
#endif
      return std::move(static_cast<FindResult&>(found.front()));
#if ET_HAS_EXCEPTIONS
    } catch (const std::bad_alloc&) {
      return {};
    }
#endif
  }

  // find_identity() for several requests at once, without refreshing recency.
  // Each result is what find_identity() would return for that query against
  // the cache as it was at the call, except that errors stay per query (see
  // IdentityResult). The search runs in waves: every query still searching
  // selects its next candidate and queues its clone before any clone of the
  // wave is awaited, so a Runner can serve one wave in a single engine
  // iteration. A query whose clone is refused continues in the next wave.
  // Selectors of different queries may interleave; each sees its own
  // candidates in find_identity() order. Every queued clone has settled when
  // this returns, and unused sessions close on the calling thread.
  std::vector<IdentityResult> find_identities(
      const std::vector<IdentityQuery>& queries) {
    std::vector<IdentityResult> results(queries.size());
    if (entries_.empty()) {
      return results;
    }
#if ET_HAS_EXCEPTIONS
    try {
#endif
      std::vector<Search> searches(queries.size());
      for (std::size_t i = 0; i < queries.size(); ++i) {
        const auto& query = queries[i];
        if (!query.request || query.request->size() < 2) {
          continue;
        }
        const auto& request = *query.request;
        auto& candidates = searches[i].candidates;
        candidates.reserve(entries_.size());
        std::size_t rank = 0;
        for (auto it = entries_.begin(); it != entries_.end(); ++it, ++rank) {
          const auto matched = match_length(*it, request);
          if (matched > query.longer_than && matched <= query.at_most) {
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
        searches[i].searching = !candidates.empty();
      }
      // Both helpers settle every clone they queue within its wave and catch
      // every exception, so no future outlives this loop.
      for (bool searching = true; searching;) {
        searching = false;
        for (std::size_t i = 0; i < queries.size(); ++i) {
          if (searches[i].searching) {
            queue_next_clone(queries[i], searches[i], results[i]);
          }
        }
        for (std::size_t i = 0; i < queries.size(); ++i) {
          if (searches[i].clone.valid()) {
            settle_clone(searches[i], results[i]);
          }
          searching = searching || searches[i].searching;
        }
      }
#if ET_HAS_EXCEPTIONS
    } catch (const std::bad_alloc&) {
      // Allocation failure before any clone is queued ends every search as a
      // miss.
    }
#endif
    return results;
  }

  // The longest match find_identity() would try first: the longest cached
  // prefix in (longer_than, at_most], capped at request.size() - 1, or 0. One
  // scan, without allocating, cloning or refreshing recency.
  std::size_t longest_match(
      const PrefixIdentity& request,
      std::size_t longer_than = 0,
      std::size_t at_most = std::numeric_limits<std::size_t>::max()) const {
    std::size_t longest = 0;
    if (request.size() < 2) {
      return longest;
    }
    for (const auto& entry : entries_) {
      const auto matched = match_length(entry, request);
      if (matched > longer_than && matched <= at_most && matched > longest) {
        longest = matched;
      }
    }
    return longest;
  }

  // Refresh only a match from this cache; eviction invalidates its entry ID.
  void touch(const PrefixMatch& match) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->id == match.cache_entry_id) {
        entries_.splice(entries_.begin(), entries_, it);
        return;
      }
    }
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
    std::uint64_t id;
  };

  struct Candidate {
    std::list<Entry>::iterator entry;
    std::size_t matched;
    std::size_t rank;
  };
  struct Search {
    std::vector<Candidate> candidates;
    std::size_t next = 0;
    bool searching = false;
    std::future<std::optional<Session>> clone;
  };

  static std::size_t match_length(
      const Entry& entry,
      const PrefixIdentity& request) {
    return std::min(common_prefix(entry.tokens, request), request.size() - 1);
  }

  // Selects this query's next candidate and queues its clone, or ends the
  // search on Stop, exhaustion or an exception.
  static void queue_next_clone(
      const IdentityQuery& query,
      Search& search,
      IdentityResult& result) {
#if ET_HAS_EXCEPTIONS
    try {
#endif
      while (search.next < search.candidates.size()) {
        const auto& candidate = search.candidates[search.next++];
        if (query.select) {
          const auto decision = query.select(candidate.matched);
          if (decision == MatchDecision::Stop) {
            result.stopped = true;
            search.searching = false;
            return;
          }
          if (decision == MatchDecision::Skip) {
            continue;
          }
        }
        search.clone = candidate.entry->snapshot.clone_async(
            static_cast<Position>(candidate.matched));
        return;
      }
#if ET_HAS_EXCEPTIONS
    } catch (const std::bad_alloc&) {
    } catch (...) {
      result.error = std::current_exception();
    }
#endif
    search.searching = false;
  }

  // Adopts a settled clone; a refused clone leaves the search running.
  static void settle_clone(Search& search, IdentityResult& result) {
#if ET_HAS_EXCEPTIONS
    try {
#endif
      auto session = search.clone.get();
      if (!session) {
        search.searching = search.next < search.candidates.size();
        return;
      }
      const auto& candidate = search.candidates[search.next - 1];
      result.match = PrefixMatch{
          std::move(*session), candidate.matched, candidate.entry->id};
#if ET_HAS_EXCEPTIONS
    } catch (const std::bad_alloc&) {
    } catch (...) {
      result.error = std::current_exception();
    }
#endif
    search.searching = false;
  }

  const std::size_t max_entries_;
  RetentionPolicy retention_policy_;
  std::list<Entry> entries_; // most recently used first
  std::uint64_t next_entry_id_ = 1;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
