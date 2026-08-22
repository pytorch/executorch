/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Orders steps into batches. Decodes first, up to max_decode_sequences, then
// the rest of max_batch_size on prefill, so prefill never delays a queued
// decode. Decode is one arrival-order queue. Prefill is a FIFO per session
// plus a rotation over sessions, and a pass takes at most one chunk from each,
// so a long prompt advances a chunk at a time instead of monopolising the
// batch.
//
// The scheduler reads exactly three things from a Request:
//
//   tokens.size()   budget arithmetic, and decode vs prefill routing
//   session_id      which prefill FIFO, and a slot in the rotation
//   request_id      the key it is tracked and cancelled under
//
// Everything else is carried from submit() to Batch, one way, without being
// inspected -- the executor reads it, not the scheduler -- so payload fields
// can be added without touching any scheduling logic.
//
// Results do not come back through here, and a step stops being tracked the
// moment it is handed to a Batch: whoever drains get_work() owns it from then
// on. cancel() therefore only reaches steps still waiting for a batch.
//
// Every public method is guarded by mutex_. The Scheduler must outlive every
// thread using it; params() hands out a reference into it.

#include <atomic>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <executorch/extension/llm/batching/step.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

class SchedulerParams {
 public:
  explicit SchedulerParams(
      std::int32_t max_decode_sequences = 32,
      std::int32_t max_prefill_chunk_size = 256)
      : max_decode_sequences_(max_decode_sequences),
        max_prefill_chunk_size_(max_prefill_chunk_size) {
    if (max_decode_sequences_ <= 0 || max_prefill_chunk_size_ <= 0) {
      throw std::invalid_argument("SchedulerParams: limits must be positive");
    }
    // max_batch_size() is derived, so the combination has to be representable.
    // Silently wrapping would hand get_work() a negative budget, which admits
    // nothing and leaves every request unresolved.
    const std::int64_t budget =
        2 * static_cast<std::int64_t>(max_prefill_chunk_size_) +
        max_decode_sequences_;
    if (budget > std::numeric_limits<std::int32_t>::max()) {
      throw std::invalid_argument(
          "SchedulerParams: 2 * max_prefill_chunk_size + max_decode_sequences "
          "overflows int32");
    }
  }

  // Decodes admitted per batch; the rest wait. Bounds the working set the
  // planner must hold at once.
  std::int32_t max_decode_sequences() const {
    return max_decode_sequences_;
  }
  // Largest prefill chunk accepted. A larger submit is rejected, not split.
  std::int32_t max_prefill_chunk_size() const {
    return max_prefill_chunk_size_;
  }
  // Room for two full prefill chunks beside a saturated decode batch. Exceeding
  // max_decode_sequences is load-bearing: it means a full-size chunk still fits
  // once decodes are taken.
  std::int32_t max_batch_size() const {
    return 2 * max_prefill_chunk_size_ + max_decode_sequences_;
  }

 private:
  std::int32_t max_decode_sequences_;
  std::int32_t max_prefill_chunk_size_;
};

// Held by shared_ptr so a queue entry and the index name the same step.
struct PendingRequest {
  Request request;
  // A deque cannot cheaply erase from the middle, so dropping a step marks it
  // and the queue skips it on the way past.
  bool cancelled = false;
  // False once dispatched. Only a waiting step is counted by queued_.
  bool queued = true;
};

using PendingPtr = std::shared_ptr<PendingRequest>;
using PendingQueue = std::deque<PendingPtr>;

class Scheduler {
 public:
  explicit Scheduler(SchedulerParams params) : params_(params) {}

  // Unique within this Scheduler, which is the scope request_id must be unique
  // over. Callable from any thread.
  RequestId next_request_id() {
    return next_request_id_.fetch_add(1, std::memory_order_relaxed);
  }

  // Queue a step. False = rejected and nothing changed: no tokens, a prefill
  // above max_prefill_chunk_size, or a request_id already waiting.
  //
  // Results are not delivered here: whoever drains get_work() runs the batch
  // and already holds them, so a step carries no return channel of its own.
  bool submit(Request request) {
    auto p = std::make_shared<PendingRequest>();
    p->request = std::move(request);
    return admit_(p);
  }

  // Whether get_work() would return a non-empty batch.
  bool has_work() const {
    std::lock_guard<std::mutex> g(mutex_);
    return queued_ > 0;
  }

  // Decodes first, up to max_decode_sequences, then prefill round-robin across
  // sessions until max_batch_size is spent. Empty when nothing is queued.
  Batch get_work() {
    Batch batch;
    std::lock_guard<std::mutex> g(mutex_);
    std::int32_t budget = params_.max_batch_size();

    take_decodes_(batch, budget);
    while (budget > 0 && take_prefill_pass_(batch, budget)) {
    }
    return batch;
  }

  // Drop a step that is still queued, so it is never handed to a Batch. A step
  // already dispatched is not here to drop -- get_work() released it when it
  // put it in the batch -- so this is a no-op for one in flight, and for an id
  // that never existed. Callers that abandon a step can therefore call it
  // without first knowing which state it is in.
  void cancel(RequestId request_id) {
    std::lock_guard<std::mutex> g(mutex_);
    (void)take_(request_id);
  }

  // Drop every queued step. For shutdown.
  void clear() {
    std::lock_guard<std::mutex> g(mutex_);
    for (auto& entry : pending_requests_) {
      entry.second->cancelled = true;
    }
    pending_requests_.clear();
    queued_ = 0;
    decode_queue_.clear();
    prefill_by_session_.clear();
    prefill_rotation_.clear();
  }

  // True while the step is waiting for a batch. A dispatched step is no longer
  // tracked, so this does not mean "submitted and unfinished".
  bool pending(RequestId request_id) const {
    std::lock_guard<std::mutex> g(mutex_);
    return pending_requests_.find(request_id) != pending_requests_.end();
  }

  // Steps waiting in a queue, excluding those in flight.
  std::size_t queued() const {
    std::lock_guard<std::mutex> g(mutex_);
    return queued_;
  }

  const SchedulerParams& params() const {
    return params_;
  }

 private:
  // Validate, register and queue. False = rejected and nothing changed.
  bool admit_(const PendingPtr& p) {
    const std::int32_t n = p->request.n_tokens();
    if (n == 0) {
      return false;
    }
    if (!p->request.is_decode() && n > params_.max_prefill_chunk_size()) {
      return false;
    }

    std::lock_guard<std::mutex> g(mutex_);
    // emplace drops a duplicate silently, which would leave the step
    // registered under an id its owner already uses.
    auto [slot, inserted] = pending_requests_.emplace(p->request.request_id, p);
    if (!inserted) {
      return false;
    }

    try {
      if (p->request.is_decode()) {
        decode_queue_.push_back(p);
      } else {
        const SessionId sid = p->request.session_id;
        // Rotation before map: a rotation entry with no session is harmless,
        // since get_work() drops it, whereas a session missing from the
        // rotation is never visited and its chunks never run.
        if (prefill_by_session_.find(sid) == prefill_by_session_.end()) {
          prefill_rotation_.push_back(sid);
        }
        prefill_by_session_[sid].push_back(p);
      }
    } catch (...) {
      // Registered but unqueued would be unschedulable: it would never appear
      // in a batch, yet block its id. Undo and reject instead.
      pending_requests_.erase(slot);
      return false;
    }
    queued_ += 1;
    return true;
  }

  // Hand a step to a Batch: it leaves the queue and pending_requests_ at once,
  // since that index only exists to find a step still waiting. Takes the id
  // separately because the caller has already moved the request into the
  // batch. Caller holds mutex_.
  void dispatch_(const PendingPtr& p, RequestId request_id) {
    p->queued = false;
    queued_ -= 1;
    pending_requests_.erase(request_id);
  }

  // Remove a waiting step and return it, or null if it is not waiting -- which
  // includes one already dispatched. Caller holds mutex_.
  PendingPtr take_(RequestId request_id) {
    auto it = pending_requests_.find(request_id);
    if (it == pending_requests_.end()) {
      return nullptr;
    }
    PendingPtr p = it->second;
    // Marked so the queue drops it on the way past, since a deque cannot erase
    // from the middle.
    p->cancelled = true;
    p->queued = false;
    queued_ -= 1;
    pending_requests_.erase(it);
    return p;
  }

  // cancel() already dropped these and decremented queued_; they are only
  // still here because a deque cannot erase from the middle.
  static void drop_cancelled_(PendingQueue& q) {
    while (!q.empty() && q.front()->cancelled) {
      q.pop_front();
    }
  }

  // Decodes in arrival order. A decode is one token and max_batch_size exceeds
  // max_decode_sequences, so the budget cannot run out here.
  // Caller holds mutex_.
  void take_decodes_(Batch& batch, std::int32_t& budget) {
    for (std::int32_t n = 0; n < params_.max_decode_sequences(); ++n) {
      drop_cancelled_(decode_queue_);
      if (decode_queue_.empty()) {
        return;
      }
      // Move rather than copy: dispatch_ releases the step on the next line,
      // so the source is discarded either way, and a prefill chunk carries up
      // to max_prefill_chunk_size tokens. Request's move is noexcept, so
      // push_back still gives the strong guarantee -- nothing is lost if it
      // throws. The id is read first, since the request is gone after.
      const PendingPtr p = decode_queue_.front();
      const RequestId id = p->request.request_id;
      batch.requests.push_back(std::move(p->request));
      dispatch_(p, id);
      decode_queue_.pop_front();
      budget -= 1;
    }
  }

  // At most one chunk per session. Returns whether anything was taken.
  // Caller holds mutex_.
  bool take_prefill_pass_(Batch& batch, std::int32_t& budget) {
    bool progress = false;
    std::vector<SessionId> deferred; // head did not fit
    std::vector<SessionId> served;

    // Nothing is put back into the rotation here, so it shrinks by one per
    // iteration and each session is visited at most once.
    while (budget > 0 && !prefill_rotation_.empty()) {
      const SessionId sid = prefill_rotation_.front();
      prefill_rotation_.pop_front();
      auto it = prefill_by_session_.find(sid);
      if (it == prefill_by_session_.end()) {
        continue;
      }
      PendingQueue& dq = it->second;
      drop_cancelled_(dq);
      if (dq.empty()) {
        prefill_by_session_.erase(it);
        continue;
      }
      const std::int32_t n = dq.front()->request.n_tokens();
      if (n > budget) {
        // Pass the turn on rather than stop, so a smaller chunk behind can
        // still use the budget.
        deferred.push_back(sid);
        continue;
      }
      // Moved, not copied; see take_decodes_.
      const PendingPtr p = dq.front();
      const RequestId id = p->request.request_id;
      batch.requests.push_back(std::move(p->request));
      dispatch_(p, id);
      dq.pop_front();
      budget -= n;
      progress = true;
      if (dq.empty()) {
        prefill_by_session_.erase(it);
      } else {
        served.push_back(sid);
      }
    }

    // Deferred sessions were popped from the front, so they belong ahead of
    // whatever the pass never reached; served sessions go to the back. Everyone
    // who got nothing keeps their order and outranks everyone who did.
    for (auto it = deferred.rbegin(); it != deferred.rend(); ++it) {
      prefill_rotation_.push_front(*it);
    }
    for (SessionId sid : served) {
      prefill_rotation_.push_back(sid);
    }
    return progress;
  }

  SchedulerParams params_;
  mutable std::mutex mutex_;

  PendingQueue decode_queue_;
  // A session is in prefill_rotation_ exactly when it has an entry here, and
  // the entry is erased as soon as its deque drains, so the rotation cannot
  // accumulate duplicates.
  std::unordered_map<SessionId, PendingQueue> prefill_by_session_;
  std::deque<SessionId> prefill_rotation_;

  // Every step waiting for a batch, from submit() until it is dispatched or
  // cancelled; the queues hold the same shared_ptr.
  std::unordered_map<RequestId, PendingPtr> pending_requests_;
  std::size_t queued_ = 0;
  std::atomic<RequestId> next_request_id_{1};
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
