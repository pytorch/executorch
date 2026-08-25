/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Decodes first, up to max_decode_sequences, then spends the rest of
// max_batch_tokens on prefill, so prefill never delays a queued decode.
// Prefill rotates over sessions taking one chunk each, so a long prompt
// cannot monopolise a batch.
//
// Reads only input.size, input.sid, is_decode and tid from a Task. The rest is
// carried through untouched.

#include <cassert>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <executorch/extension/llm/batching/scheduler.h>
#include <executorch/extension/llm/batching/types.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

// Shared so that pending_ and the queues name the same task. Cancelling marks
// it once, and each queue skips it when scheduling reaches it.
using TaskPtr = std::shared_ptr<Task>;
using TaskQueue = std::deque<TaskPtr>;

class DecodeFirstScheduler : public Scheduler {
 public:
  // Returns nullptr if the limits are unusable. All three must be non-zero,
  // and the budget must cover a saturated decode batch plus one full chunk.
  // Below that floor a chunk of max_prefill_chunk_size could be admitted and
  // then never fit in any batch, stranding the task rather than delaying it.
  // The same floor is what lets take_decodes_ spend without checking the
  // budget.
  //
  // A factory rather than a throwing constructor, because
  // EXECUTORCH_OPTIMIZE_SIZE builds with -fno-exceptions, and this header is
  // deliberately free of ExecuTorch runtime types, so ET_CHECK is unavailable
  // too.
  static std::unique_ptr<DecodeFirstScheduler> create(
      std::size_t max_batch_tokens = 544,
      std::size_t max_decode_sequences = 32,
      std::size_t max_prefill_chunk_size = 256) {
    if (max_batch_tokens == 0 || max_decode_sequences == 0 ||
        max_prefill_chunk_size == 0) {
      return nullptr;
    }
    if (max_decode_sequences >= max_batch_tokens) {
      return nullptr; // no room left for prefill
    }
    // Ordered by the check above, so the subtraction cannot wrap.
    if (max_prefill_chunk_size > max_batch_tokens - max_decode_sequences) {
      return nullptr;
    }
    return std::unique_ptr<DecodeFirstScheduler>(new DecodeFirstScheduler(
        max_batch_tokens, max_decode_sequences, max_prefill_chunk_size));
  }

  // Rejects an empty task, a decode wider than one token, an oversized chunk,
  // and a tid already queued, including one repeated inside this vector.
  bool submit(std::vector<Task> tasks) override {
    std::lock_guard<std::mutex> g(mutex_);
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      if (!admissible_(tasks[i])) {
        return false;
      }
      // admissible_ tests pending_, which this vector has not joined yet, so a
      // repeat within one submit has to be caught here. Pairwise because a
      // vector holds one prompt's chunks: small, and it allocates nothing.
      for (std::size_t j = 0; j < i; ++j) {
        if (tasks[j].tid == tasks[i].tid) {
          return false;
        }
      }
    }
    for (Task& t : tasks) {
      enqueue_(std::move(t));
    }
    return true;
  }

  bool has_work() const override {
    std::lock_guard<std::mutex> g(mutex_);
    return !pending_.empty();
  }

  // Spending accumulates instead of counting a budget down, so no arithmetic
  // here can wrap below zero.
  std::vector<Task> get_work() override {
    std::vector<Task> taken;
    std::lock_guard<std::mutex> g(mutex_);
    std::size_t spent = 0;
    // Sessions that already hold a decode in this batch. The executor is
    // promised consecutive ranges and one produce_output per session, so a
    // session's second decode, or a prefill chunk beside its decode, waits for
    // the next batch. Local to the call, so unlike a persistent index it
    // cannot fall out of sync with the queues.
    std::unordered_set<SessionId> decoding;

    take_decodes_(taken, spent, decoding);
    while (spent < max_batch_tokens_ &&
           take_prefill_pass_(taken, spent, decoding)) {
    }
    return taken;
  }

  std::vector<Task> cancel(SessionId sid) override {
    std::vector<Task> dropped;
    std::lock_guard<std::mutex> g(mutex_);

    auto prefills = prefill_by_session_.find(sid);
    if (prefills != prefill_by_session_.end()) {
      for (const TaskPtr& t : prefills->second) {
        if (release_(t)) {
          dropped.push_back(std::move(*t));
        }
      }
      // The entry stays behind, empty, because it holds the session's place in
      // the rotation. take_prefill_pass_ retires the two together.
      prefills->second.clear();
    }
    // Marked in place rather than erased: the queues already tolerate
    // cancelled entries, and drop_cancelled_ removes them as they surface.
    for (const TaskPtr& t : decode_queue_) {
      if (t->input.sid == sid && release_(t)) {
        dropped.push_back(std::move(*t));
      }
    }
    return dropped;
  }

  std::vector<Task> clear() override {
    std::vector<Task> dropped;
    std::lock_guard<std::mutex> g(mutex_);

    dropped.reserve(pending_.size());
    for (auto& entry : pending_) {
      entry.second->cancelled = true;
      dropped.push_back(std::move(*entry.second));
    }
    pending_.clear();
    decode_queue_.clear();
    prefill_by_session_.clear();
    prefill_rotation_.clear();
    return dropped;
  }

  // The whole budget one batch may spend, in tokens.
  std::size_t max_batch_tokens() const {
    return max_batch_tokens_;
  }
  // Decodes admitted per batch. The rest wait.
  std::size_t max_decode_sequences() const {
    return max_decode_sequences_;
  }
  // Largest chunk accepted. A larger submit is rejected, not split.
  std::size_t max_prefill_chunk_size() const {
    return max_prefill_chunk_size_;
  }

 private:
  DecodeFirstScheduler(
      std::size_t max_batch_tokens,
      std::size_t max_decode_sequences,
      std::size_t max_prefill_chunk_size)
      : max_batch_tokens_(max_batch_tokens),
        max_decode_sequences_(max_decode_sequences),
        max_prefill_chunk_size_(max_prefill_chunk_size) {}

  // Caller holds mutex_.
  bool admissible_(const Task& t) const {
    if (t.input.size == 0) {
      return false;
    }
    if (t.is_decode) {
      // take_decodes_ spends one token per decode, and create()'s floor is
      // written in those terms, so a wider decode would overspend the budget.
      if (t.input.size != 1) {
        return false;
      }
    } else if (t.input.size > max_prefill_chunk_size_) {
      return false;
    }
    return pending_.find(t.tid) == pending_.end();
  }

  // Caller holds mutex_, having already checked admissible_.
  void enqueue_(Task&& task) {
    const bool decode = task.is_decode;
    const SessionId sid = task.input.sid;
    const TaskId tid = task.tid;

    auto t = std::make_shared<Task>(std::move(task));
    t->cancelled = false;
    pending_.emplace(tid, t);

    if (decode) {
      decode_queue_.push_back(t);
      return;
    }
    // A session is in the rotation exactly when it has an entry here, so the
    // insertion, not the emptiness of the queue, is what decides whether it
    // joins. Testing for emptiness would give a second slot to a session whose
    // entry cancel() emptied, and it would then take two turns per pass.
    auto entry = prefill_by_session_.try_emplace(sid);
    if (entry.second) {
      prefill_rotation_.push_back(sid);
    }
    entry.first->second.push_back(t);
  }

  // pending_ tracks only tasks still waiting, so a dispatched one leaves it.
  // Caller holds mutex_.
  void dispatch_(TaskId tid) {
    pending_.erase(tid);
  }

  // Returns whether this call was the one that dropped the task, so that a
  // double cancel reports it once. Caller holds mutex_.
  bool release_(const TaskPtr& t) {
    if (pending_.erase(t->tid) == 0) {
      return false; // already handed out, or already dropped
    }
    t->cancelled = true;
    return true;
  }

  // Discards entries release_ already dropped. They are still in the queue
  // only because a deque cannot erase from the middle.
  static void drop_cancelled_(TaskQueue& q) {
    while (!q.empty() && q.front()->cancelled) {
      q.pop_front();
    }
  }

  // Takes decodes in arrival order, at most one per session. create() keeps
  // max_batch_tokens above max_decode_sequences, so this cannot exhaust the
  // budget. Caller holds mutex_.
  void take_decodes_(
      std::vector<Task>& taken,
      std::size_t& spent,
      std::unordered_set<SessionId>& decoding) {
    std::vector<TaskPtr> deferred; // session already decoding in this batch
    std::size_t n = 0;
    while (n < max_decode_sequences_) {
      drop_cancelled_(decode_queue_);
      if (decode_queue_.empty()) {
        break;
      }
      const TaskPtr t = decode_queue_.front();
      decode_queue_.pop_front();
      if (!decoding.insert(t->input.sid).second) {
        deferred.push_back(t);
        continue;
      }
      // Read the id before moving the task out of the shared entry.
      const TaskId tid = t->tid;
      taken.push_back(std::move(*t));
      dispatch_(tid);
      ++n;
      spent += 1;
    }
    // Returned to the head in arrival order, so a session passed over here
    // still leads the queue next time.
    for (auto it = deferred.rbegin(); it != deferred.rend(); ++it) {
      decode_queue_.push_front(*it);
    }
  }

  // One turn each, in rotation order. Returns whether anything was taken.
  // Caller holds mutex_.
  bool take_prefill_pass_(
      std::vector<Task>& taken,
      std::size_t& spent,
      const std::unordered_set<SessionId>& decoding) {
    bool progress = false;
    std::vector<SessionId> deferred; // passed over, took nothing
    std::vector<SessionId> served;

    // Nothing rejoins the rotation inside this loop, so each session is
    // visited at most once per pass.
    while (spent < max_batch_tokens_ && !prefill_rotation_.empty()) {
      const SessionId sid = prefill_rotation_.front();
      prefill_rotation_.pop_front();
      // A rotation slot exists exactly when the map entry does. enqueue_ adds
      // the pair, and every path below retires the pair.
      auto it = prefill_by_session_.find(sid);
      assert(
          it != prefill_by_session_.end() &&
          "prefill_rotation_ names a session with no queue");
      TaskQueue& dq = it->second;
      drop_cancelled_(dq);
      if (dq.empty()) {
        prefill_by_session_.erase(it);
        continue;
      }
      if (decoding.count(sid) != 0) {
        // The session's decode is already in this batch. Adding a chunk beside
        // it would hand the executor two ranges that do not adjoin, both
        // asking to produce output.
        deferred.push_back(sid);
        continue;
      }
      const std::size_t n = dq.front()->input.size;
      // The loop condition keeps spent below the budget, so the remaining room
      // is positive.
      if (n > max_batch_tokens_ - spent) {
        // Pass the turn on rather than stop, since a smaller chunk behind this
        // one may still fit.
        deferred.push_back(sid);
        continue;
      }
      const TaskPtr t = dq.front();
      const TaskId tid = t->tid;
      taken.push_back(std::move(*t));
      dispatch_(tid);
      dq.pop_front();
      spent += n;
      progress = true;
      if (dq.empty()) {
        prefill_by_session_.erase(it);
      } else {
        served.push_back(sid);
      }
    }

    // Sessions that took nothing keep their order and rank ahead of those that
    // did. They were popped before whatever remains in the rotation, so
    // restoring them at the head preserves arrival order.
    for (auto it = deferred.rbegin(); it != deferred.rend(); ++it) {
      prefill_rotation_.push_front(*it);
    }
    for (SessionId sid : served) {
      prefill_rotation_.push_back(sid);
    }
    return progress;
  }

  mutable std::mutex mutex_;
  std::size_t max_batch_tokens_;
  std::size_t max_decode_sequences_;
  std::size_t max_prefill_chunk_size_;

  TaskQueue decode_queue_;

  // A session is in prefill_rotation_ exactly when it has an entry here. That
  // pairing is what keeps the rotation free of duplicates.
  std::unordered_map<SessionId, TaskQueue> prefill_by_session_;
  std::deque<SessionId> prefill_rotation_;

  // Membership is what "still queued" means, so no separate count can fall out
  // of sync with it.
  std::unordered_map<TaskId, TaskPtr> pending_;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
