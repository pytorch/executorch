/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Step scheduler for batched LLM serving. A Request is one step for one
// sequence -- a single decode token, or one prefill chunk -- not a whole
// generation. Callers submit a step, await its sampled token, and submit the
// next one.
//
// get_work() takes decodes first, up to max_decode_sequences, then spends the
// rest of max_batch_size on prefill, so prefill never delays a queued decode.
// Decode is one arrival-order queue. Prefill is a FIFO per session plus a
// rotation over sessions, and a pass takes at most one chunk from each, so a
// long prompt advances a chunk at a time instead of monopolising the batch.
//
// No tensors, cache, or model here: where a step's KV lives belongs to the
// planner. Callers split prefill into chunks of at most max_prefill_chunk_size,
// which needs the session's position sequence and so is caller state.
//
// The scheduler reads exactly three things from a Request:
//
//   tokens.size()   budget arithmetic, and decode vs prefill routing
//   session_id      which prefill FIFO, and a slot in the rotation
//   request_id      the key its result is settled through
//
// Everything else lives in RequestParams / ResponsePayload and is carried from
// submit()
// to Batch and back without being inspected, so payload fields can be added
// without touching any scheduling logic.
//
// Every public method is guarded by mutex_. The Scheduler must outlive every
// thread using it; params() hands out a reference into it.

#include <atomic>
#include <cstdint>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

using Token = std::int64_t;
using RequestId = std::int64_t;
using SessionId = std::int64_t;
using Position = std::int32_t;

// --- Payload: carried through the scheduler untouched ----------------------

// Carried per step, so a caller may change sampling between turns.
struct SamplingParams {
  float temperature = 0.0f;
  float top_p = 1.0f;
  std::int32_t top_k = 0;
  std::uint64_t seed = 0;
};

// Which positions of a step the model must produce output for. A verify step
// needs every row -- it inspects the prediction at each drafted position --
// while a decode or prefill chunk only needs the last. A sampled step yields
// one token per row; an unsampled one yields one distribution per row.
enum class OutputRows {
  Last,
  All,
};

// Raw model output for a step whose Request asked not to sample
struct LogitsBlock {
  std::vector<float> data;
  std::int32_t n_rows = 0;
  std::int32_t vocab = 0;
};
using LogitsPtr = std::shared_ptr<const LogitsBlock>;

// How to execute a step. Promote to a variant if a second step kind ever needs
// different fields rather than different values.
struct RequestParams {
  Position position = 0;
  // Absent means: do not sample, return the raw distribution. Rejection
  // sampling and constrained decoding need the distribution; greedy
  // speculative verification does not, since argmax at every row is just
  // sampling at temperature zero.
  std::optional<SamplingParams> sampling;
  OutputRows output_rows = OutputRows::Last;
};

// Which alternative is held follows from whether the Request asked to sample.
// A sampled step yields one token per output row, so a greedy verify round
// returns the prediction at every drafted position; deciding how many to
// accept, and where the session therefore continues, is the caller's job.
using ResponsePayload = std::variant<std::vector<Token>, LogitsPtr>;

// --- Scheduling ------------------------------------------------------------

// One step for one sequence. request_id names the submission and is always
// unique, so the same work submitted twice is two requests that both run and
// both get answered. Only request_id, session_id, and tokens.size() are read
// by the scheduler; `params` is carried.
struct Request {
  RequestId request_id = 0;
  SessionId session_id = 0;
  std::vector<Token> tokens;
  RequestParams params;

  std::int32_t n_tokens() const {
    return static_cast<std::int32_t>(tokens.size());
  }

  // A one-token prefill is indistinguishable from a decode and is treated as
  // one: same single row of output, one decode slot.
  bool is_decode() const {
    return tokens.size() == 1;
  }
};

struct Response {
  RequestId request_id = 0;
  SessionId session_id = 0;
  ResponsePayload payload;
};

// One batch of steps for a single forward. Decodes first, then prefills, in
// admission order.
struct Batch {
  std::vector<Request> requests;

  bool empty() const {
    return requests.empty();
  }
  std::int32_t n_tokens() const {
    std::int32_t n = 0;
    for (const Request& r : requests) {
      n += r.n_tokens();
    }
    return n;
  }
};

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

// Held by shared_ptr because std::promise is move-only.
struct PendingRequest {
  Request request;
  // A deque cannot cheaply erase from the middle, so fail() marks and the
  // queues drop on the way past.
  bool cancelled = false;
  // Still waiting in a queue, as opposed to handed to a Batch. Only a queued
  // step is counted by queued_, so this is what fail() checks before
  // decrementing it.
  bool queued = true;
  std::promise<Response> promise;
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

  // The future carries the sampled token, or an exception if the step is
  // rejected: no tokens, a prefill above max_prefill_chunk_size, or a
  // request_id already outstanding.
  std::future<Response> submit(Request request) {
    auto p = std::make_shared<PendingRequest>();
    p->request = std::move(request);
    std::future<Response> fut = p->promise.get_future();

    const std::int32_t n = p->request.n_tokens();
    if (n == 0) {
      set_error_(*p, "step carries no tokens");
      return fut;
    }
    if (!p->request.is_decode() && n > params_.max_prefill_chunk_size()) {
      set_error_(*p, "prefill chunk exceeds max_prefill_chunk_size");
      return fut;
    }

    std::lock_guard<std::mutex> g(mutex_);
    // emplace drops a duplicate silently, which would leave this promise never
    // settled and the caller's future blocked forever.
    auto [slot, inserted] = pending_requests_.emplace(p->request.request_id, p);
    if (!inserted) {
      set_error_(*p, "request_id already outstanding");
      return fut;
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
      // Registered but unqueued would be unschedulable, so its future would
      // never settle. Undo and reject instead.
      pending_requests_.erase(slot);
      set_error_(*p, "failed to queue step");
      return fut;
    }
    queued_ += 1;
    return fut;
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

  // Settle a sampled step: one token per output row, so a greedy verify round
  // reports the target's prediction at every drafted position. An unknown id is
  // ignored, so a completion racing fail_all() is not an error.
  void complete(RequestId request_id, std::vector<Token> tokens) {
    Response r;
    r.payload = std::move(tokens);
    settle_(request_id, std::move(r));
  }

  // Settle an unsampled step with its raw distribution.
  void complete(RequestId request_id, LogitsPtr logits) {
    Response r;
    r.payload = std::move(logits);
    settle_(request_id, std::move(r));
  }

  // Fails one step, queued or in flight. Cancelling a whole generation is the
  // caller's loop over its outstanding ids.
  void fail(RequestId request_id, const std::string& what) {
    PendingPtr p;
    {
      std::lock_guard<std::mutex> g(mutex_);
      auto it = pending_requests_.find(request_id);
      if (it == pending_requests_.end()) {
        return;
      }
      p = it->second;
      p->cancelled = true;
      // An in-flight step was already uncounted by get_work(); decrementing
      // again would underflow queued_ or mask other queued work.
      if (p->queued) {
        p->queued = false;
        queued_ -= 1;
      }
      pending_requests_.erase(it);
    }
    set_error_(*p, what);
  }

  // For shutdown.
  void fail_all(const std::string& what) {
    std::vector<PendingPtr> doomed;
    {
      std::lock_guard<std::mutex> g(mutex_);
      for (auto& entry : pending_requests_) {
        entry.second->cancelled = true;
        doomed.push_back(entry.second);
      }
      pending_requests_.clear();
      queued_ = 0;
      decode_queue_.clear();
      prefill_by_session_.clear();
      prefill_rotation_.clear();
    }
    for (PendingPtr& p : doomed) {
      set_error_(*p, what);
    }
  }

  // Whether `request_id` is still queued or in flight.
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
  // Fills in identity and hands `r` to the waiter. Erasing under the lock
  // before settling is what stops complete() and fail() both resolving the
  // same promise.
  void settle_(RequestId request_id, Response&& r) {
    PendingPtr p;
    {
      std::lock_guard<std::mutex> g(mutex_);
      auto it = pending_requests_.find(request_id);
      if (it == pending_requests_.end()) {
        return;
      }
      p = it->second;
      pending_requests_.erase(it);
      r.request_id = request_id;
      r.session_id = p->request.session_id;
    }
    p->promise.set_value(std::move(r)); // outside the lock: wakes the waiter
  }

  static void set_error_(PendingRequest& p, const std::string& what) {
    p.promise.set_exception(
        std::make_exception_ptr(std::runtime_error("scheduler: " + what)));
  }

  // fail() already settled these and decremented queued_; they are only still
  // here because a deque cannot erase from the middle.
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
      // Copy into the batch first: it is the only step here that can throw, and
      // popping first would lose the request entirely.
      batch.requests.push_back(decode_queue_.front()->request);
      decode_queue_.front()->queued = false;
      decode_queue_.pop_front();
      budget -= 1;
      queued_ -= 1;
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
      // Copy into the batch before mutating anything else; see take_decodes_.
      batch.requests.push_back(dq.front()->request);
      dq.front()->queued = false;
      dq.pop_front();
      budget -= n;
      queued_ -= 1;
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

  // Index from submit() until complete()/fail(); the queues hold the same
  // shared_ptr.
  std::unordered_map<RequestId, PendingPtr> pending_requests_;
  std::size_t queued_ = 0;
  std::atomic<RequestId> next_request_id_{1};
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
