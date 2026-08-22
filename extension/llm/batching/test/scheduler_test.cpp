/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/scheduler.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <thread>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

using executorch::extension::llm::batching::Batch;
using executorch::extension::llm::batching::LogitsBlock;
using executorch::extension::llm::batching::LogitsPtr;
using executorch::extension::llm::batching::OutputRows;
using executorch::extension::llm::batching::Position;
using executorch::extension::llm::batching::Request;
using executorch::extension::llm::batching::RequestId;
using executorch::extension::llm::batching::RequestParams;
using executorch::extension::llm::batching::SamplingParams;
using executorch::extension::llm::batching::Scheduler;
using executorch::extension::llm::batching::SchedulerParams;
using executorch::extension::llm::batching::SessionId;
using executorch::extension::llm::batching::Token;

namespace {

// A step of `n_tokens` for `session` at `position`, sampled by default. Token
// values are irrelevant to scheduling, so they are all the same.
Request make_request(
    RequestId request_id,
    SessionId session,
    int n_tokens,
    Position position,
    OutputRows rows = OutputRows::Last,
    bool sample = true) {
  Request r;
  r.request_id = request_id;
  r.session_id = session;
  r.tokens.assign(static_cast<std::size_t>(n_tokens), 7);
  r.params.position = position;
  r.params.output_rows = rows;
  if (sample) {
    r.params.sampling = SamplingParams{};
  }
  return r;
}

std::vector<RequestId> ids(const Batch& b) {
  std::vector<RequestId> out;
  out.reserve(b.requests.size());
  for (const Request& r : b.requests) {
    out.push_back(r.request_id);
  }
  return out;
}

} // namespace

// Guards every get() whose absence would block instead of failing.

namespace {} // namespace

// --- SchedulerParams -------------------------------------------------------

TEST(SchedulerParamsTest, DefaultsAndDerivedBatchSize) {
  SchedulerParams p;
  EXPECT_EQ(p.max_decode_sequences(), 32);
  EXPECT_EQ(p.max_prefill_chunk_size(), 256);
  EXPECT_EQ(p.max_batch_size(), 2 * 256 + 32);
}

TEST(SchedulerParamsTest, BatchSizeIsDerivedNotStored) {
  SchedulerParams p(2, 4);
  EXPECT_EQ(p.max_batch_size(), 2 * 4 + 2);
}

// The decode loop omits a budget check because this always holds.
TEST(SchedulerParamsTest, BatchSizeAlwaysExceedsDecodeCap) {
  for (std::int32_t d = 1; d <= 8; ++d) {
    for (std::int32_t c = 1; c <= 8; ++c) {
      SchedulerParams p(d, c);
      EXPECT_GT(p.max_batch_size(), p.max_decode_sequences());
      EXPECT_GE(p.max_batch_size() - p.max_decode_sequences(), c);
    }
  }
}

TEST(SchedulerParamsTest, RejectsNonPositiveLimits) {
  EXPECT_THROW(SchedulerParams(0, 4), std::invalid_argument);
  EXPECT_THROW(SchedulerParams(2, 0), std::invalid_argument);
  EXPECT_THROW(SchedulerParams(-1, 4), std::invalid_argument);
  EXPECT_THROW(SchedulerParams(2, -1), std::invalid_argument);
}

// A wrapped max_batch_size() would be a negative budget: get_work() would admit
// nothing and every request would sit unresolved forever.
TEST(SchedulerParamsTest, RejectsCombinationsThatOverflowBatchSize) {
  constexpr std::int32_t kMax = std::numeric_limits<std::int32_t>::max();
  EXPECT_THROW(SchedulerParams(1, kMax), std::invalid_argument);
  EXPECT_THROW(SchedulerParams(kMax, kMax), std::invalid_argument);
  EXPECT_THROW(SchedulerParams(kMax, kMax / 2), std::invalid_argument);
}

TEST(SchedulerParamsTest, AcceptsTheLargestRepresentableCombination) {
  constexpr std::int32_t kMax = std::numeric_limits<std::int32_t>::max();
  // 2 * chunk + decodes == kMax exactly.
  const std::int32_t chunk = (kMax - 1) / 2;
  SchedulerParams p(1, chunk);
  EXPECT_GT(p.max_batch_size(), 0);
  EXPECT_EQ(p.max_batch_size(), 2 * chunk + 1);
  EXPECT_THROW(SchedulerParams(2, chunk), std::invalid_argument);
}

// --- Request / Batch shape -------------------------------------------------

TEST(RequestTest, OneTokenStepIsADecode) {
  EXPECT_TRUE(make_request(1, 10, 1, 0).is_decode());
  EXPECT_FALSE(make_request(1, 10, 2, 0).is_decode());
}

TEST(RequestTest, OutputRowsIsCarriedNotInterpreted) {
  EXPECT_EQ(
      make_request(1, 10, 8, 0, OutputRows::Last).params.output_rows,
      OutputRows::Last);
  EXPECT_EQ(
      make_request(1, 10, 5, 0, OutputRows::All).params.output_rows,
      OutputRows::All);
}

TEST(RequestTest, PayloadSupportsAllFourQuadrants) {
  EXPECT_TRUE(
      make_request(1, 10, 1, 0, OutputRows::Last, true).params.sampling);
  EXPECT_TRUE(make_request(2, 10, 5, 0, OutputRows::All, true).params.sampling);
  EXPECT_FALSE(
      make_request(3, 10, 1, 0, OutputRows::Last, false).params.sampling);
  EXPECT_FALSE(
      make_request(4, 10, 5, 0, OutputRows::All, false).params.sampling);
}

TEST(BatchTest, EmptyBatchReportsZeros) {
  Scheduler s{SchedulerParams(1, 4)};
  Batch b = s.get_work();
  EXPECT_TRUE(b.empty());
  EXPECT_EQ(b.n_tokens(), 0);
}

TEST(BatchTest, SumsTokensAcrossSteps) {
  Scheduler s{SchedulerParams(2, 8)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(3, 30, 6, 0)));
  EXPECT_TRUE(s.submit(make_request(4, 40, 5, 100, OutputRows::All)));

  Batch b = s.get_work();
  EXPECT_EQ(ids(b), (std::vector<RequestId>{1, 2, 3, 4}));
  EXPECT_EQ(b.n_tokens(), 1 + 1 + 6 + 5);
}

TEST(BatchTest, CarriesPayloadUninspected) {
  Scheduler s{SchedulerParams(1, 8)};
  EXPECT_TRUE(
      s.submit(make_request(1, 40, 5, 100, OutputRows::All, /*sample=*/false)));

  Batch b = s.get_work();
  ASSERT_EQ(b.requests.size(), 1u);
  EXPECT_EQ(b.requests[0].params.position, 100);
  EXPECT_EQ(b.requests[0].params.output_rows, OutputRows::All);
  EXPECT_FALSE(b.requests[0].params.sampling.has_value());
}

// --- submit ----------------------------------------------------------------

TEST(SubmitTest, RejectsEmptyStep) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_FALSE(s.submit(make_request(1, 10, 0, 0)));
  EXPECT_EQ(s.queued(), 0u);
}

TEST(SubmitTest, RejectsPrefillAboveChunkSize) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_FALSE(s.submit(make_request(1, 10, 9, 0)));
  EXPECT_EQ(s.queued(), 0u);
}

TEST(SubmitTest, RejectsDuplicateRequestId) {
  Scheduler s{SchedulerParams(2, 4)};
  EXPECT_TRUE(s.submit(make_request(7, 10, 1, 0)));
  EXPECT_FALSE(s.submit(make_request(7, 20, 1, 0)));

  EXPECT_EQ(s.queued(), 1u) << "rejected step must not inflate the count";
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{7}));
}

TEST(SubmitTest, DuplicateWorkWithDistinctIdsBothRun) {
  Scheduler s{SchedulerParams(2, 4)};
  Request a = make_request(0, 10, 1, 5);
  Request b = make_request(0, 10, 1, 5);
  a.request_id = s.next_request_id();
  b.request_id = s.next_request_id();

  EXPECT_TRUE(s.submit(a));
  EXPECT_TRUE(s.submit(b));
  EXPECT_EQ(s.get_work().requests.size(), 2u);

  // Dispatch released them, so neither is tracked any more.
  EXPECT_FALSE(s.pending(a.request_id));
  EXPECT_FALSE(s.pending(b.request_id));
}

TEST(SubmitTest, RejectedStepEntersNoQueue) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_FALSE(s.submit(make_request(1, 10, 0, 0)));
  EXPECT_FALSE(s.submit(make_request(2, 10, 9, 0)));

  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
  EXPECT_FALSE(s.pending(1));
  EXPECT_FALSE(s.pending(2));
  EXPECT_TRUE(s.get_work().empty());
}

TEST(SubmitTest, NextRequestIdIsUnique) {
  Scheduler s{SchedulerParams(2, 4)};
  std::set<RequestId> seen;
  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(seen.insert(s.next_request_id()).second);
  }
}

TEST(SubmitTest, NextRequestIdIsUniqueAcrossThreads) {
  Scheduler s{SchedulerParams(2, 4)};
  constexpr int kThreads = 8;
  constexpr int kPer = 500;
  std::vector<std::vector<RequestId>> out(kThreads);
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kPer; ++i) {
        out[t].push_back(s.next_request_id());
      }
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }
  std::set<RequestId> seen;
  for (const auto& v : out) {
    for (RequestId id : v) {
      EXPECT_TRUE(seen.insert(id).second);
    }
  }
  EXPECT_EQ(seen.size(), static_cast<std::size_t>(kThreads * kPer));
}

// --- decode scheduling -----------------------------------------------------

TEST(DecodeTest, ServedInArrivalOrderUpToTheCap) {
  Scheduler s{SchedulerParams(2, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(3, 30, 1, 0)));

  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 2}));
  EXPECT_EQ(s.queued(), 1u);
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3}));
}

// A shorter queue must not give a later arrival a head start.
TEST(DecodeTest, StaysFifoAcrossDrainAndRefill) {
  Scheduler s{SchedulerParams(2, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(3, 30, 1, 0)));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 2}));

  EXPECT_TRUE(s.submit(make_request(4, 40, 1, 0)));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3, 4}));
}

TEST(DecodeTest, BeatsPrefillAndStillLeavesRoomForAFullChunk) {
  Scheduler s{SchedulerParams(3, 4)}; // batch = 11
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(3, 30, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(50, 90, 4, 0)));

  Batch b = s.get_work();
  EXPECT_EQ(ids(b), (std::vector<RequestId>{1, 2, 3, 50}));
  EXPECT_EQ(b.n_tokens(), 3 + 4);
}

// --- prefill scheduling ----------------------------------------------------

TEST(PrefillTest, ChunksOfOneSessionStayInOrder) {
  Scheduler s{SchedulerParams(1, 2)};
  EXPECT_TRUE(s.submit(make_request(1, 7, 2, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 7, 2, 2)));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 2}));
}

TEST(PrefillTest, LongPromptCannotHogTheBatch) {
  Scheduler s{SchedulerParams(2, 4)};
  for (int c = 0; c < 4; ++c) {
    EXPECT_TRUE(s.submit(make_request(1 + c, 10, 4, c * 4)));
  }
  EXPECT_TRUE(s.submit(make_request(5, 20, 4, 0)));
  EXPECT_TRUE(s.submit(make_request(6, 20, 4, 4)));

  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 5}));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2, 6}));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3, 4}))
      << "session 20 drained, so session 10 may take two chunks";
  EXPECT_EQ(s.queued(), 0u);
}

TEST(PrefillTest, LoneSessionFillsTheBatchAcrossPasses) {
  Scheduler s{SchedulerParams(1, 4)}; // batch = 9
  EXPECT_TRUE(s.submit(make_request(1, 10, 4, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 10, 4, 4)));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 2}));
}

TEST(PrefillTest, StopsWhenTheNextChunkDoesNotFit) {
  Scheduler s{SchedulerParams(1, 4)}; // batch = 9
  EXPECT_TRUE(s.submit(make_request(1, 10, 4, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 4, 0)));
  EXPECT_TRUE(s.submit(make_request(3, 30, 4, 0)));

  Batch b = s.get_work();
  EXPECT_EQ(b.requests.size(), 2u);
  EXPECT_EQ(s.queued(), 1u);
}

// A session skipped on size was reached; one the pass never got to was not.
// Both got nothing, so they must keep their original relative order.
TEST(PrefillTest, DeferredSessionOutranksOneNeverReached) {
  Scheduler s{SchedulerParams(1, 4)}; // batch = 9
  s.submit(make_request(1, 10, 4, 0)); // served, 9 -> 5
  s.submit(make_request(2, 20, 3, 0)); // served, 5 -> 2
  s.submit(make_request(3, 30, 4, 0)); // 4 > 2, deferred
  s.submit(make_request(4, 40, 2, 0)); // served, 2 -> 0, pass exits
  s.submit(make_request(5, 50, 4, 0)); // never reached

  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 2, 4}));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3, 5}));
}

TEST(PrefillTest, DeferredSessionsKeepTheirOrder) {
  Scheduler s{SchedulerParams(1, 4)}; // batch = 9
  EXPECT_TRUE(s.submit(make_request(1, 10, 4, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 4, 0)));
  EXPECT_TRUE(s.submit(make_request(3, 30, 4, 0)));
  EXPECT_TRUE(s.submit(make_request(4, 40, 3, 0)));

  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 2}));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3, 4}));
}

// Two long prompts must not starve a third: served sessions rotate to the back.
TEST(PrefillTest, RotationIsFairAcrossCalls) {
  Scheduler s{SchedulerParams(1, 4)};
  RequestId id = 1;
  for (SessionId session : {10, 20, 30}) {
    for (int c = 0; c < 8; ++c) {
      EXPECT_TRUE(s.submit(make_request(id++, session, 4, c * 4)));
    }
  }

  std::map<SessionId, int> served;
  for (int call = 0; call < 9; ++call) {
    for (const Request& r : s.get_work().requests) {
      served[r.session_id]++;
    }
  }
  EXPECT_EQ(served[10], 6);
  EXPECT_EQ(served[20], 6);
  EXPECT_EQ(served[30], 6);
}

// --- cancelling a queued step ----------------------------------------------

TEST(CancelStepTest, DropsAQueuedStepBeforeItRuns) {
  Scheduler s{SchedulerParams(2, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 77, 1, 12)));
  EXPECT_TRUE(s.pending(1));
  EXPECT_EQ(s.queued(), 1u);

  s.cancel(1);
  EXPECT_FALSE(s.pending(1));
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
  EXPECT_TRUE(s.get_work().empty()) << "the queued entry must be dropped";
}

TEST(CancelStepTest, UnknownIdIsIgnored) {
  Scheduler s{SchedulerParams(1, 4)};
  s.cancel(404);
  EXPECT_EQ(s.queued(), 0u);
  SUCCEED();
}

// Dispatch releases a step, so cancelling one already in a batch has nothing
// to do. Callers abandoning a step can call this without knowing which state
// it is in.
TEST(CancelStepTest, DispatchedStepIsNoLongerTracked) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_TRUE(s.submit(make_request(9, 77, 4, 12)));
  EXPECT_TRUE(s.pending(9));

  ASSERT_EQ(ids(s.get_work()), (std::vector<RequestId>{9}));
  EXPECT_FALSE(s.pending(9)) << "get_work released it";
  EXPECT_EQ(s.queued(), 0u);

  s.cancel(9); // no-op, and must not disturb the count
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
}

// The id is free for reuse the moment the step is dispatched, since nothing
// tracks it any more.
TEST(CancelStepTest, DispatchFreesTheIdForReuse) {
  Scheduler s{SchedulerParams(2, 4)};
  EXPECT_TRUE(s.submit(make_request(7, 10, 1, 0)));
  ASSERT_EQ(ids(s.get_work()), (std::vector<RequestId>{7}));

  EXPECT_TRUE(s.submit(make_request(7, 10, 1, 1)));
  EXPECT_TRUE(s.pending(7));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{7}));
}

// --- cancellation
// ---------------------------------------------------------------

TEST(CancelTest, CancelsOneStepAndLeavesOthersRunnable) {
  Scheduler s{SchedulerParams(2, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 1, 0)));

  s.cancel(1);
  EXPECT_FALSE(s.pending(1));
  EXPECT_EQ(s.queued(), 1u);
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2}));
}

TEST(CancelTest, AllCancelledDecodesDrainToAnEmptyBatch) {
  Scheduler s{SchedulerParams(2, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 1, 0)));
  s.cancel(1);
  s.cancel(2);

  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
  EXPECT_TRUE(s.get_work().empty());
}

TEST(CancelTest, CancelledPrefillLeavesTheRotation) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 4, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 4, 0)));
  s.cancel(1);

  EXPECT_EQ(s.queued(), 1u);
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2}));
}

TEST(CancelTest, ClearEmptiesEveryQueue) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 4, 0)));

  s.clear();
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
  EXPECT_TRUE(s.get_work().empty());
  EXPECT_FALSE(s.pending(1));
  EXPECT_FALSE(s.pending(2));
}

TEST(CancelTest, ClearSweepsQueuedAndInFlightTogether) {
  Scheduler s{SchedulerParams(2, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 1, 0)));
  ASSERT_EQ(s.get_work().requests.size(), 2u);
  ASSERT_EQ(s.queued(), 0u);

  EXPECT_TRUE(s.submit(make_request(3, 30, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(4, 40, 4, 0)));
  ASSERT_EQ(s.queued(), 2u);

  s.clear();
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
  for (RequestId id = 1; id <= 4; ++id) {
    EXPECT_FALSE(s.pending(id));
  }
  s.cancel(1); // cancelling after the sweep is ignored, not an error
  SUCCEED();
}

TEST(CancelTest, UnknownIdIsIgnored) {
  Scheduler s{SchedulerParams(1, 4)};
  s.cancel(404);
  EXPECT_EQ(s.queued(), 0u);
  SUCCEED();
}

// An in-flight step was already uncounted by get_work(). Decrementing again
// would wrap queued_ and leave has_work() permanently true.
TEST(CancelTest, CancellingInFlightStepDoesNotUnderflowQueued) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  ASSERT_EQ(s.get_work().requests.size(), 1u);
  ASSERT_EQ(s.queued(), 0u);

  s.cancel(1);
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
}

// ...and the other direction: it must not drop a count that belongs to work
// still waiting, or the engine would sleep with a full queue.
TEST(CancelTest, CancellingInFlightStepDoesNotHideQueuedWork) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  ASSERT_EQ(s.get_work().requests.size(), 1u);
  EXPECT_TRUE(s.submit(make_request(2, 20, 1, 0)));
  ASSERT_EQ(s.queued(), 1u);

  s.cancel(1);
  EXPECT_EQ(s.queued(), 1u) << "the queued step is still queued";
  EXPECT_TRUE(s.has_work());
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2}));
}

TEST(CancelTest, CancellingAQueuedStepStillDecrementsOnce) {
  Scheduler s{SchedulerParams(2, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 1, 0)));
  ASSERT_EQ(s.queued(), 2u);

  s.cancel(1);
  EXPECT_EQ(s.queued(), 1u);
  s.cancel(1); // already gone
  EXPECT_EQ(s.queued(), 1u);
}

TEST(CancelTest, CancellingEveryInFlightStepLeavesQueuedCountAtZero) {
  Scheduler s{SchedulerParams(4, 4)};
  for (RequestId id = 1; id <= 4; ++id) {
    EXPECT_TRUE(s.submit(make_request(id, id, 1, 0)));
  }
  ASSERT_EQ(s.get_work().requests.size(), 4u);
  ASSERT_EQ(s.queued(), 0u);

  for (RequestId id = 1; id <= 4; ++id) {
    s.cancel(id);
  }
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
}

// A cancelled entry is only dropped once it reaches the head of its queue, so
// one buried behind live steps has to survive until then without disturbing
// them or the count.
TEST(CancelTest, CancelledDecodeInTheMiddleOfTheQueue) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 20, 1, 0)));
  EXPECT_TRUE(s.submit(make_request(3, 30, 1, 0)));

  s.cancel(2);
  EXPECT_EQ(s.queued(), 2u);

  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1}));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3}));
  EXPECT_EQ(s.queued(), 0u);
}

TEST(CancelTest, CancelledPrefillChunkInTheMiddleOfASession) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 7, 4, 0)));
  EXPECT_TRUE(s.submit(make_request(2, 7, 4, 4)));
  EXPECT_TRUE(s.submit(make_request(3, 7, 4, 8)));

  s.cancel(2);
  EXPECT_EQ(s.queued(), 2u);

  // Surviving chunks of the session keep their relative order.
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 3}));
  EXPECT_EQ(s.queued(), 0u);
}

TEST(PrefillTest, SessionRejoinsTheRotationAfterDraining) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 4, 0)));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1}));
  EXPECT_TRUE(s.get_work().empty());

  EXPECT_TRUE(s.submit(make_request(2, 10, 4, 4)));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2}));

  // And it interleaves normally with a second session afterwards.
  EXPECT_TRUE(s.submit(make_request(3, 10, 4, 8)));
  EXPECT_TRUE(s.submit(make_request(4, 20, 4, 0)));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3, 4}));
}

TEST(PrefillTest, SessionWhoseOnlyChunkIsCancelledLeavesNoStaleRotationEntry) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_TRUE(s.submit(make_request(1, 10, 4, 0)));
  s.cancel(1);
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_TRUE(s.get_work().empty());

  // The stale rotation entry, if any, must not swallow the session's next turn.
  EXPECT_TRUE(s.submit(make_request(2, 10, 4, 0)));
  EXPECT_TRUE(s.submit(make_request(3, 20, 4, 0)));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2, 3}));
}

// --- has_work --------------------------------------------------------------

TEST(HasWorkTest, TracksQueuedStepsOnly) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_FALSE(s.has_work());

  EXPECT_TRUE(s.submit(make_request(1, 10, 1, 0)));
  EXPECT_TRUE(s.has_work());

  s.get_work();
  EXPECT_FALSE(s.has_work()) << "in flight is not queued";
  s.cancel(1);
  EXPECT_FALSE(s.has_work());
}

// --- randomized invariants -------------------------------------------------

// queued() must equal exactly the steps submitted but not yet taken into a
// batch. An upper bound alone would miss under-counting, which is the failure
// mode that silently hides work from an engine waiting on has_work().
TEST(InvariantTest, QueuedCountStaysConsistentUnderRandomOps) {
  std::mt19937 rng(1234);
  Scheduler s{SchedulerParams(3, 4)};

  // The scheduler tracks exactly the steps waiting for a batch: dispatch and
  // cancel both release one.
  std::set<RequestId> waiting;
  int submitted = 0, dispatched = 0, cancelled = 0;

  auto expect_exact_count = [&](int op) {
    EXPECT_EQ(s.queued(), waiting.size())
        << "queued() diverged from the model at op " << op;
  };

  for (int i = 0; i < 4000; ++i) {
    expect_exact_count(i);
    if (::testing::Test::HasFailure()) {
      return;
    }

    switch (rng() % 3) {
      case 0: { // submit
        const RequestId id = s.next_request_id();
        ASSERT_TRUE(s.submit(make_request(
            id,
            static_cast<SessionId>(rng() % 4),
            static_cast<int>(1 + rng() % 4),
            static_cast<Position>(rng() % 100))));
        waiting.insert(id);
        submitted++;
        break;
      }
      case 1: { // dispatch a batch
        for (const Request& r : s.get_work().requests) {
          EXPECT_EQ(waiting.erase(r.request_id), 1u)
              << "a batch contained a step the model did not have queued";
          EXPECT_FALSE(s.pending(r.request_id)) << "dispatch must release it";
          dispatched++;
        }
        break;
      }
      case 2: { // cancel something still waiting
        if (waiting.empty()) {
          break;
        }
        auto it = waiting.begin();
        std::advance(it, static_cast<std::ptrdiff_t>(rng() % waiting.size()));
        s.cancel(*it);
        waiting.erase(it);
        cancelled++;
        break;
      }
      default:
        break;
    }
  }

  // Drain: every remaining step must reach a batch, and the count must land
  // exactly on zero rather than wrapping past it.
  while (!waiting.empty()) {
    Batch b = s.get_work();
    if (b.empty()) {
      break;
    }
    for (const Request& r : b.requests) {
      waiting.erase(r.request_id);
      dispatched++;
    }
  }

  EXPECT_TRUE(waiting.empty());
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
  EXPECT_EQ(dispatched + cancelled, submitted);
}

// --- concurrency -----------------------------------------------------------

TEST(ConcurrencyTest, ProducersAndEngineMakeProgressWithoutLoss) {
  Scheduler s{SchedulerParams(4, 8)};
  constexpr int kProducers = 4;
  constexpr int kPer = 250;
  std::atomic<bool> stop{false};
  std::atomic<int> accepted{0};
  std::atomic<int> dispatched{0};

  std::thread engine([&] {
    while (!stop.load()) {
      Batch b = s.get_work();
      dispatched += static_cast<int>(b.requests.size());
      if (b.empty()) {
        std::this_thread::yield();
      }
    }
    // Drain after stop: work submitted between the last check and the store
    // would otherwise never be dispatched.
    Batch b = s.get_work();
    dispatched += static_cast<int>(b.requests.size());
  });

  std::vector<std::thread> producers;
  for (int t = 0; t < kProducers; ++t) {
    producers.emplace_back([&, t] {
      for (int i = 0; i < kPer; ++i) {
        if (s.submit(make_request(
                s.next_request_id(),
                (t * 7 + i) % 5,
                (i % 3 == 0) ? 1 : 4,
                static_cast<Position>(i)))) {
          accepted++;
        }
      }
    });
  }
  for (std::thread& t : producers) {
    t.join();
  }
  // Let the engine catch up before stopping it, so the count is meaningful.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (dispatched.load() < accepted.load() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  stop.store(true);
  engine.join();

  EXPECT_EQ(accepted.load(), kProducers * kPer) << "a submit was rejected";
  EXPECT_EQ(dispatched.load(), accepted.load())
      << "a step was never dispatched";
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
}

TEST(ConcurrencyTest, ObserversAreSafeDuringScheduling) {
  Scheduler s{SchedulerParams(2, 4)};
  std::atomic<bool> stop{false};
  std::atomic<bool> observing{false};
  std::atomic<long> observations{0};

  std::thread observer([&] {
    observing.store(true);
    while (!stop.load()) {
      (void)s.has_work();
      (void)s.queued();
      (void)s.pending(1);
      (void)s.params().max_batch_size();
      observations++;
    }
  });
  // Without this the main loop can finish before the observer starts, leaving
  // the test passing vacuously.
  while (!observing.load()) {
    std::this_thread::yield();
  }

  for (int i = 0; i < 500; ++i) {
    const RequestId id = s.next_request_id();
    ASSERT_TRUE(s.submit(make_request(id, 10, 1, 0)));
    (void)s.get_work(); // dispatch releases each step
  }
  stop.store(true);
  observer.join();

  EXPECT_GT(observations.load(), 0) << "observer never ran";
  EXPECT_EQ(s.queued(), 0u);
}
