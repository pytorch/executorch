/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/scheduler/scheduler.h>

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

using executorch::extension::llm::scheduler::Batch;
using executorch::extension::llm::scheduler::LogitsBlock;
using executorch::extension::llm::scheduler::LogitsPtr;
using executorch::extension::llm::scheduler::OutputRows;
using executorch::extension::llm::scheduler::Position;
using executorch::extension::llm::scheduler::Request;
using executorch::extension::llm::scheduler::RequestId;
using executorch::extension::llm::scheduler::RequestParams;
using executorch::extension::llm::scheduler::Response;
using executorch::extension::llm::scheduler::SamplingParams;
using executorch::extension::llm::scheduler::Scheduler;
using executorch::extension::llm::scheduler::SchedulerParams;
using executorch::extension::llm::scheduler::SessionId;
using executorch::extension::llm::scheduler::Token;

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

bool settled(std::future<Response>& f) {
  return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

bool holds_tokens(const Response& r) {
  return std::holds_alternative<std::vector<Token>>(r.payload);
}

const std::vector<Token>& tokens_of(const Response& r) {
  return std::get<std::vector<Token>>(r.payload);
}

LogitsPtr logits_of(const Response& r) {
  return std::get<LogitsPtr>(r.payload);
}

// Bounded wait. A regression that leaves a future unresolved must fail
// diagnostically rather than hang the suite -- several tests here exist
// precisely to catch non-settlement.
constexpr std::chrono::seconds kSettleTimeout{5};

testing::AssertionResult Settles(std::future<Response>& f) {
  if (f.wait_for(kSettleTimeout) != std::future_status::ready) {
    return testing::AssertionFailure()
        << "future did not settle within " << kSettleTimeout.count() << "s";
  }
  return testing::AssertionSuccess();
}

} // namespace

// Guards every get() whose absence would block instead of failing.
#define ASSERT_SETTLES(f) ASSERT_TRUE(Settles(f))

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
  s.submit(make_request(1, 10, 1, 0));
  s.submit(make_request(2, 20, 1, 0));
  s.submit(make_request(3, 30, 6, 0));
  s.submit(make_request(4, 40, 5, 100, OutputRows::All));

  Batch b = s.get_work();
  EXPECT_EQ(ids(b), (std::vector<RequestId>{1, 2, 3, 4}));
  EXPECT_EQ(b.n_tokens(), 1 + 1 + 6 + 5);
}

TEST(BatchTest, CarriesPayloadUninspected) {
  Scheduler s{SchedulerParams(1, 8)};
  s.submit(make_request(1, 40, 5, 100, OutputRows::All, /*sample=*/false));

  Batch b = s.get_work();
  ASSERT_EQ(b.requests.size(), 1u);
  EXPECT_EQ(b.requests[0].params.position, 100);
  EXPECT_EQ(b.requests[0].params.output_rows, OutputRows::All);
  EXPECT_FALSE(b.requests[0].params.sampling.has_value());
}

// --- submit ----------------------------------------------------------------

TEST(SubmitTest, RejectsEmptyStep) {
  Scheduler s{SchedulerParams(1, 4)};
  auto f = s.submit(make_request(1, 10, 0, 0));
  ASSERT_SETTLES(f);
  EXPECT_THROW((void)f.get(), std::runtime_error);
  EXPECT_EQ(s.queued(), 0u);
}

TEST(SubmitTest, RejectsPrefillAboveChunkSize) {
  Scheduler s{SchedulerParams(1, 4)};
  auto f = s.submit(make_request(1, 10, 9, 0));
  ASSERT_SETTLES(f);
  EXPECT_THROW((void)f.get(), std::runtime_error);
  EXPECT_EQ(s.queued(), 0u);
}

TEST(SubmitTest, RejectsDuplicateRequestIdRatherThanHanging) {
  Scheduler s{SchedulerParams(2, 4)};
  auto first = s.submit(make_request(7, 10, 1, 0));
  auto dup = s.submit(make_request(7, 20, 1, 0));

  ASSERT_SETTLES(dup);
  EXPECT_THROW((void)dup.get(), std::runtime_error);
  EXPECT_EQ(s.queued(), 1u) << "rejected step must not inflate the count";
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{7}));
  s.complete(7, std::vector<Token>{42});
  ASSERT_SETTLES(first);
  EXPECT_EQ(tokens_of(first.get()), (std::vector<Token>{42}));
}

TEST(SubmitTest, RequestIdIsReusableOnceSettled) {
  Scheduler s{SchedulerParams(2, 4)};
  auto first = s.submit(make_request(7, 10, 1, 0));
  s.get_work();
  s.complete(7, std::vector<Token>{1});
  ASSERT_SETTLES(first);
  EXPECT_EQ(tokens_of(first.get())[0], 1);
  EXPECT_FALSE(s.pending(7));

  auto second = s.submit(make_request(7, 10, 1, 1));
  s.get_work();
  s.complete(7, std::vector<Token>{2});
  ASSERT_SETTLES(second);
  EXPECT_EQ(tokens_of(second.get())[0], 2);
}

TEST(SubmitTest, DuplicateWorkWithDistinctIdsBothRun) {
  Scheduler s{SchedulerParams(2, 4)};
  Request a = make_request(0, 10, 1, 5);
  Request b = make_request(0, 10, 1, 5);
  a.request_id = s.next_request_id();
  b.request_id = s.next_request_id();

  auto fa = s.submit(a);
  auto fb = s.submit(b);
  EXPECT_EQ(s.get_work().requests.size(), 2u);

  s.complete(a.request_id, std::vector<Token>{111});
  s.complete(b.request_id, std::vector<Token>{222});
  ASSERT_SETTLES(fa);
  EXPECT_EQ(tokens_of(fa.get())[0], 111);
  ASSERT_SETTLES(fb);
  EXPECT_EQ(tokens_of(fb.get())[0], 222);
}

TEST(SubmitTest, RejectedStepEntersNoQueue) {
  Scheduler s{SchedulerParams(1, 4)};
  auto empty = s.submit(make_request(1, 10, 0, 0));
  auto toobig = s.submit(make_request(2, 10, 9, 0));
  ASSERT_SETTLES(empty);
  EXPECT_THROW((void)empty.get(), std::runtime_error);
  ASSERT_SETTLES(toobig);
  EXPECT_THROW((void)toobig.get(), std::runtime_error);

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
  s.submit(make_request(1, 10, 1, 0));
  s.submit(make_request(2, 20, 1, 0));
  s.submit(make_request(3, 30, 1, 0));

  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 2}));
  EXPECT_EQ(s.queued(), 1u);
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3}));
}

// A shorter queue must not give a later arrival a head start.
TEST(DecodeTest, StaysFifoAcrossDrainAndRefill) {
  Scheduler s{SchedulerParams(2, 4)};
  s.submit(make_request(1, 10, 1, 0));
  s.submit(make_request(2, 20, 1, 0));
  s.submit(make_request(3, 30, 1, 0));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 2}));

  s.submit(make_request(4, 40, 1, 0));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3, 4}));
}

TEST(DecodeTest, BeatsPrefillAndStillLeavesRoomForAFullChunk) {
  Scheduler s{SchedulerParams(3, 4)}; // batch = 11
  s.submit(make_request(1, 10, 1, 0));
  s.submit(make_request(2, 20, 1, 0));
  s.submit(make_request(3, 30, 1, 0));
  s.submit(make_request(50, 90, 4, 0));

  Batch b = s.get_work();
  EXPECT_EQ(ids(b), (std::vector<RequestId>{1, 2, 3, 50}));
  EXPECT_EQ(b.n_tokens(), 3 + 4);
}

// --- prefill scheduling ----------------------------------------------------

TEST(PrefillTest, ChunksOfOneSessionStayInOrder) {
  Scheduler s{SchedulerParams(1, 2)};
  s.submit(make_request(1, 7, 2, 0));
  s.submit(make_request(2, 7, 2, 2));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 2}));
}

TEST(PrefillTest, LongPromptCannotHogTheBatch) {
  Scheduler s{SchedulerParams(2, 4)};
  for (int c = 0; c < 4; ++c) {
    s.submit(make_request(1 + c, 10, 4, c * 4));
  }
  s.submit(make_request(5, 20, 4, 0));
  s.submit(make_request(6, 20, 4, 4));

  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 5}));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2, 6}));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3, 4}))
      << "session 20 drained, so session 10 may take two chunks";
  EXPECT_EQ(s.queued(), 0u);
}

TEST(PrefillTest, LoneSessionFillsTheBatchAcrossPasses) {
  Scheduler s{SchedulerParams(1, 4)}; // batch = 9
  s.submit(make_request(1, 10, 4, 0));
  s.submit(make_request(2, 10, 4, 4));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 2}));
}

TEST(PrefillTest, StopsWhenTheNextChunkDoesNotFit) {
  Scheduler s{SchedulerParams(1, 4)}; // batch = 9
  s.submit(make_request(1, 10, 4, 0));
  s.submit(make_request(2, 20, 4, 0));
  s.submit(make_request(3, 30, 4, 0));

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
  s.submit(make_request(1, 10, 4, 0));
  s.submit(make_request(2, 20, 4, 0));
  s.submit(make_request(3, 30, 4, 0));
  s.submit(make_request(4, 40, 3, 0));

  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 2}));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3, 4}));
}

// Two long prompts must not starve a third: served sessions rotate to the back.
TEST(PrefillTest, RotationIsFairAcrossCalls) {
  Scheduler s{SchedulerParams(1, 4)};
  RequestId id = 1;
  for (SessionId session : {10, 20, 30}) {
    for (int c = 0; c < 8; ++c) {
      s.submit(make_request(id++, session, 4, c * 4));
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

// --- completion ------------------------------------------------------------

TEST(CompleteTest, SampledStepReturnsTokensAndNoLogits) {
  Scheduler s{SchedulerParams(2, 4)};
  auto f = s.submit(make_request(1, 77, 1, 12));
  s.get_work();
  s.complete(1, std::vector<Token>{999});

  ASSERT_SETTLES(f);
  Response r = f.get();
  EXPECT_EQ(r.request_id, 1);
  EXPECT_EQ(r.session_id, 77);
  EXPECT_EQ(tokens_of(r), (std::vector<Token>{999}));
  EXPECT_TRUE(holds_tokens(r));
}

// Greedy verification: one token per drafted position. How many are accepted,
// and where the session continues, is the caller's business.
TEST(CompleteTest, GreedyVerifyReturnsOneTokenPerOutputRow) {
  Scheduler s{SchedulerParams(1, 8)};
  auto f = s.submit(make_request(2, 77, 5, 100, OutputRows::All));

  Batch b = s.get_work();
  ASSERT_EQ(b.requests.size(), 1u);
  EXPECT_EQ(b.requests[0].params.output_rows, OutputRows::All);
  s.complete(2, std::vector<Token>{11, 22, 33, 44, 55});

  ASSERT_SETTLES(f);
  Response r = f.get();
  EXPECT_EQ(tokens_of(r), (std::vector<Token>{11, 22, 33, 44, 55}))
      << "a verify round must report the prediction at each drafted position, "
         "in order";
  EXPECT_TRUE(holds_tokens(r));
}

TEST(CompleteTest, UnsampledStepReturnsLogitsAndNoTokens) {
  Scheduler s{SchedulerParams(1, 8)};
  auto f =
      s.submit(make_request(3, 77, 4, 200, OutputRows::All, /*sample=*/false));
  s.get_work();

  auto block = std::make_shared<LogitsBlock>();
  block->n_rows = 4;
  block->vocab = 3;
  block->data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  s.complete(3, LogitsPtr(block));

  ASSERT_SETTLES(f);
  Response r = f.get();
  EXPECT_FALSE(holds_tokens(r));
  LogitsPtr got = logits_of(r);
  ASSERT_NE(got, nullptr);
  EXPECT_EQ(got->n_rows, 4);
  EXPECT_EQ(got->vocab, 3);
  ASSERT_EQ(got->data.size(), 12u);
  EXPECT_FLOAT_EQ(got->data[11], 11.0f);
}

TEST(CompleteTest, PendingIsTrueUntilSettled) {
  Scheduler s{SchedulerParams(1, 4)};
  auto f = s.submit(make_request(9, 77, 4, 12));
  EXPECT_TRUE(s.pending(9));
  s.get_work();
  EXPECT_TRUE(s.pending(9)) << "in flight still counts as pending";
  s.complete(9, std::vector<Token>{1});
  EXPECT_FALSE(s.pending(9));
  EXPECT_FALSE(s.pending(12345));
}

TEST(CompleteTest, UnknownIdIsIgnoredByBothOverloads) {
  Scheduler s{SchedulerParams(1, 4)};
  s.complete(999, std::vector<Token>{1});
  s.complete(999, LogitsPtr{});
  SUCCEED();
}

// --- failure ---------------------------------------------------------------

TEST(FailTest, FailsOneStepAndLeavesOthersRunnable) {
  Scheduler s{SchedulerParams(2, 4)};
  auto doomed = s.submit(make_request(1, 10, 1, 0));
  auto ok = s.submit(make_request(2, 10, 1, 1));

  s.fail(1, "cancelled");
  EXPECT_EQ(s.queued(), 1u);
  EXPECT_FALSE(s.pending(1));
  EXPECT_TRUE(s.pending(2));
  ASSERT_SETTLES(doomed);
  EXPECT_THROW((void)doomed.get(), std::runtime_error);

  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2}));
  s.complete(2, std::vector<Token>{5});
  ASSERT_SETTLES(ok);
  EXPECT_EQ(tokens_of(ok.get())[0], 5);
}

TEST(FailTest, AllCancelledDecodesDrainToAnEmptyBatch) {
  Scheduler s{SchedulerParams(2, 4)};
  auto a = s.submit(make_request(1, 10, 1, 0));
  auto b = s.submit(make_request(2, 20, 1, 0));
  s.fail(1, "x");
  s.fail(2, "x");

  EXPECT_TRUE(s.get_work().empty());
  EXPECT_EQ(s.queued(), 0u);
  ASSERT_SETTLES(a);
  EXPECT_THROW((void)a.get(), std::runtime_error);
  ASSERT_SETTLES(b);
  EXPECT_THROW((void)b.get(), std::runtime_error);
}

TEST(FailTest, CancelledPrefillLeavesTheRotation) {
  Scheduler s{SchedulerParams(1, 2)};
  auto doomed = s.submit(make_request(1, 7, 2, 0));
  s.submit(make_request(2, 8, 2, 0));

  s.fail(1, "cancelled");
  EXPECT_EQ(s.queued(), 1u);
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2}));
  ASSERT_SETTLES(doomed);
  EXPECT_THROW((void)doomed.get(), std::runtime_error);
}

TEST(FailTest, FailAllSettlesEverythingAndEmptiesTheQueues) {
  Scheduler s{SchedulerParams(1, 4)};
  auto a = s.submit(make_request(1, 10, 1, 0));
  auto b = s.submit(make_request(2, 20, 4, 0));

  s.fail_all("shutdown");
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
  EXPECT_TRUE(s.get_work().empty());
  ASSERT_SETTLES(a);
  EXPECT_THROW((void)a.get(), std::runtime_error);
  ASSERT_SETTLES(b);
  EXPECT_THROW((void)b.get(), std::runtime_error);
}

// fail_all() assigns queued_ = 0 rather than decrementing, so the mix cannot
// underflow -- but it must still settle in-flight steps, not just queued ones.
TEST(FailTest, FailAllSettlesQueuedAndInFlightTogether) {
  Scheduler s{SchedulerParams(2, 4)};
  auto inflight_a = s.submit(make_request(1, 10, 1, 0));
  auto inflight_b = s.submit(make_request(2, 20, 1, 0));
  ASSERT_EQ(s.get_work().requests.size(), 2u);
  ASSERT_EQ(s.queued(), 0u);

  auto queued_a = s.submit(make_request(3, 30, 1, 0));
  auto queued_b = s.submit(make_request(4, 40, 4, 0));
  ASSERT_EQ(s.queued(), 2u);

  s.fail_all("shutdown");

  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
  EXPECT_TRUE(s.get_work().empty());
  for (auto* f : {&inflight_a, &inflight_b, &queued_a, &queued_b}) {
    ASSERT_SETTLES(*f);
    EXPECT_THROW((void)f->get(), std::runtime_error);
  }
  EXPECT_FALSE(s.pending(1));
  EXPECT_FALSE(s.pending(3));

  // A completion arriving after the sweep is ignored, not a double-settle.
  s.complete(1, std::vector<Token>{1});
  SUCCEED();
}

TEST(FailTest, UnknownIdIsIgnored) {
  Scheduler s{SchedulerParams(1, 4)};
  s.fail(999, "nobody");
  SUCCEED();
}

// get_work() already uncounted the step, so fail() must not decrement again.
TEST(FailTest, FailingInFlightStepDoesNotUnderflowQueued) {
  Scheduler s{SchedulerParams(2, 4)};
  auto a = s.submit(make_request(1, 10, 1, 0));
  auto b = s.submit(make_request(2, 20, 1, 0));
  ASSERT_EQ(s.queued(), 2u);
  s.get_work();
  ASSERT_EQ(s.queued(), 0u);

  s.fail(1, "in-flight fault");
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
  ASSERT_SETTLES(a);
  EXPECT_THROW((void)a.get(), std::runtime_error);

  s.complete(2, std::vector<Token>{5});
  ASSERT_SETTLES(b);
  EXPECT_EQ(tokens_of(b.get())[0], 5);
}

// The same double-decrement would otherwise hide genuinely queued work and
// stall an engine that waits on has_work().
TEST(FailTest, FailingInFlightStepDoesNotHideQueuedWork) {
  Scheduler s{SchedulerParams(1, 4)};
  auto inflight = s.submit(make_request(1, 10, 1, 0));
  s.get_work();
  auto waiting = s.submit(make_request(2, 20, 1, 0));
  ASSERT_EQ(s.queued(), 1u);

  s.fail(1, "in-flight fault");
  EXPECT_EQ(s.queued(), 1u);
  EXPECT_TRUE(s.has_work());
  ASSERT_SETTLES(inflight);
  EXPECT_THROW((void)inflight.get(), std::runtime_error);

  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2}));
  s.complete(2, std::vector<Token>{7});
  ASSERT_SETTLES(waiting);
  EXPECT_EQ(tokens_of(waiting.get())[0], 7);
}

TEST(FailTest, FailingAQueuedStepStillDecrementsOnce) {
  Scheduler s{SchedulerParams(2, 4)};
  auto a = s.submit(make_request(1, 10, 1, 0));
  auto b = s.submit(make_request(2, 20, 1, 0));
  ASSERT_EQ(s.queued(), 2u);

  s.fail(1, "cancelled");
  EXPECT_EQ(s.queued(), 1u);
  ASSERT_SETTLES(a);
  EXPECT_THROW((void)a.get(), std::runtime_error);
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2}));
  EXPECT_EQ(s.queued(), 0u);
  s.complete(2, std::vector<Token>{1});
  ASSERT_SETTLES(b);
  EXPECT_EQ(tokens_of(b.get())[0], 1);
}

TEST(FailTest, FailingEveryInFlightStepLeavesQueuedCountAtZero) {
  Scheduler s{SchedulerParams(4, 4)};
  std::vector<std::future<Response>> futures;
  for (RequestId id = 1; id <= 4; ++id) {
    futures.push_back(s.submit(make_request(id, id, 1, 0)));
  }
  ASSERT_EQ(s.get_work().requests.size(), 4u);
  ASSERT_EQ(s.queued(), 0u);

  for (RequestId id = 1; id <= 4; ++id) {
    s.fail(id, "fault");
  }
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
  for (auto& f : futures) {
    ASSERT_SETTLES(f);
    EXPECT_THROW((void)f.get(), std::runtime_error);
  }
}

// A cancelled entry is only dropped once it reaches the head of its queue, so
// one buried behind live steps has to survive until then without disturbing
// them or the count.
TEST(FailTest, CancelledDecodeInTheMiddleOfTheQueue) {
  Scheduler s{SchedulerParams(1, 4)};
  auto a = s.submit(make_request(1, 10, 1, 0));
  auto b = s.submit(make_request(2, 20, 1, 0));
  auto c = s.submit(make_request(3, 30, 1, 0));

  s.fail(2, "cancelled");
  EXPECT_EQ(s.queued(), 2u);
  ASSERT_SETTLES(b);
  EXPECT_THROW((void)b.get(), std::runtime_error);

  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1}));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3}));
  EXPECT_EQ(s.queued(), 0u);

  s.complete(1, std::vector<Token>{1});
  s.complete(3, std::vector<Token>{3});
  ASSERT_SETTLES(a);
  EXPECT_EQ(tokens_of(a.get())[0], 1);
  ASSERT_SETTLES(c);
  EXPECT_EQ(tokens_of(c.get())[0], 3);
}

TEST(FailTest, CancelledPrefillChunkInTheMiddleOfASession) {
  Scheduler s{SchedulerParams(1, 4)};
  auto a = s.submit(make_request(1, 7, 4, 0));
  auto b = s.submit(make_request(2, 7, 4, 4));
  auto c = s.submit(make_request(3, 7, 4, 8));

  s.fail(2, "cancelled");
  EXPECT_EQ(s.queued(), 2u);
  ASSERT_SETTLES(b);
  EXPECT_THROW((void)b.get(), std::runtime_error);

  // Surviving chunks of the session keep their relative order.
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1, 3}));
  EXPECT_EQ(s.queued(), 0u);
}

// A session is erased from the rotation when its last chunk is taken; a later
// submit has to put it back.
TEST(PrefillTest, SessionRejoinsTheRotationAfterDraining) {
  Scheduler s{SchedulerParams(1, 4)};
  s.submit(make_request(1, 10, 4, 0));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{1}));
  EXPECT_TRUE(s.get_work().empty());

  s.submit(make_request(2, 10, 4, 4));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2}));

  // And it interleaves normally with a second session afterwards.
  s.submit(make_request(3, 10, 4, 8));
  s.submit(make_request(4, 20, 4, 0));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{3, 4}));
}

TEST(PrefillTest, SessionWhoseOnlyChunkIsCancelledLeavesNoStaleRotationEntry) {
  Scheduler s{SchedulerParams(1, 4)};
  auto doomed = s.submit(make_request(1, 10, 4, 0));
  s.fail(1, "cancelled");
  ASSERT_SETTLES(doomed);
  EXPECT_THROW((void)doomed.get(), std::runtime_error);
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_TRUE(s.get_work().empty());

  // The stale rotation entry, if any, must not swallow the session's next turn.
  s.submit(make_request(2, 10, 4, 0));
  s.submit(make_request(3, 20, 4, 0));
  EXPECT_EQ(ids(s.get_work()), (std::vector<RequestId>{2, 3}));
}

// --- has_work --------------------------------------------------------------

TEST(HasWorkTest, TracksQueuedStepsOnly) {
  Scheduler s{SchedulerParams(1, 4)};
  EXPECT_FALSE(s.has_work());

  auto f = s.submit(make_request(1, 10, 1, 0));
  EXPECT_TRUE(s.has_work());

  s.get_work();
  EXPECT_FALSE(s.has_work()) << "in flight is not queued";
  s.complete(1, std::vector<Token>{1});
  EXPECT_FALSE(s.has_work());
  EXPECT_TRUE(settled(f));
}

// --- randomized invariants -------------------------------------------------

// queued() must equal exactly the steps submitted but not yet taken into a
// batch. An upper bound alone would miss under-counting, which is the failure
// mode that silently hides work from an engine waiting on has_work().
TEST(InvariantTest, QueuedCountStaysConsistentUnderRandomOps) {
  std::mt19937 rng(1234);
  Scheduler s{SchedulerParams(3, 4)};

  std::map<RequestId, std::future<Response>> outstanding;
  std::vector<RequestId> in_flight;
  int submitted = 0, completed = 0, failed = 0;

  // Every outstanding step is either still queued or in flight, so the
  // scheduler's count must be exactly the difference.
  auto expect_exact_count = [&](int op) {
    ASSERT_GE(outstanding.size(), in_flight.size()) << "model broken at " << op;
    EXPECT_EQ(s.queued(), outstanding.size() - in_flight.size())
        << "queued() diverged from the model at op " << op;
  };

  for (int i = 0; i < 4000; ++i) {
    expect_exact_count(i);
    if (::testing::Test::HasFailure()) {
      return;
    }

    switch (rng() % 4) {
      case 0: { // submit
        RequestId id = s.next_request_id();
        SessionId session = static_cast<SessionId>(rng() % 4);
        int n = static_cast<int>(1 + rng() % 4);
        outstanding.emplace(
            id,
            s.submit(make_request(
                id, session, n, static_cast<Position>(rng() % 100))));
        submitted++;
        break;
      }
      case 1: { // drain a batch
        for (const Request& r : s.get_work().requests) {
          in_flight.push_back(r.request_id);
        }
        break;
      }
      case 2: { // complete something in flight
        if (in_flight.empty()) {
          break;
        }
        std::size_t k = rng() % in_flight.size();
        RequestId id = in_flight[k];
        in_flight.erase(in_flight.begin() + static_cast<std::ptrdiff_t>(k));
        s.complete(id, std::vector<Token>{1});
        auto it = outstanding.find(id);
        if (it != outstanding.end()) {
          ASSERT_SETTLES(it->second);
          (void)it->second.get();
          outstanding.erase(it);
          completed++;
        }
        break;
      }
      case 3: { // fail something, queued or in flight
        if (outstanding.empty()) {
          break;
        }
        auto it = outstanding.begin();
        std::advance(
            it, static_cast<std::ptrdiff_t>(rng() % outstanding.size()));
        RequestId id = it->first;
        s.fail(id, "random");
        ASSERT_SETTLES(it->second);
        EXPECT_THROW((void)it->second.get(), std::runtime_error);
        outstanding.erase(it);
        in_flight.erase(
            std::remove(in_flight.begin(), in_flight.end(), id),
            in_flight.end());
        failed++;
        break;
      }
      default:
        break;
    }
  }

  // Drain: everything still outstanding must settle, and the count must land
  // exactly on zero rather than wrapping past it.
  while (!outstanding.empty()) {
    Batch b = s.get_work();
    if (b.empty()) {
      break;
    }
    for (const Request& r : b.requests) {
      s.complete(r.request_id, std::vector<Token>{1});
      auto it = outstanding.find(r.request_id);
      if (it != outstanding.end()) {
        ASSERT_SETTLES(it->second);
        (void)it->second.get();
        outstanding.erase(it);
        completed++;
      }
    }
  }

  EXPECT_TRUE(outstanding.empty());
  EXPECT_EQ(s.queued(), 0u);
  EXPECT_FALSE(s.has_work());
  EXPECT_EQ(completed + failed, submitted);
}

// --- concurrency -----------------------------------------------------------

// Producers submit and await; one engine thread drains and completes.
TEST(ConcurrencyTest, ProducersAndEngineMakeProgressWithoutLoss) {
  Scheduler s{SchedulerParams(4, 8)};
  constexpr int kProducers = 4;
  constexpr int kPer = 250;
  std::atomic<bool> stop{false};
  std::atomic<int> completed{0};
  std::atomic<int> lost{0};

  std::thread engine([&] {
    while (!stop.load()) {
      Batch b = s.get_work();
      for (const Request& r : b.requests) {
        s.complete(r.request_id, std::vector<Token>{42});
      }
      if (b.empty()) {
        std::this_thread::yield();
      }
    }
    Batch b = s.get_work();
    for (const Request& r : b.requests) {
      s.complete(r.request_id, std::vector<Token>{42});
    }
  });

  std::vector<std::thread> producers;
  for (int t = 0; t < kProducers; ++t) {
    producers.emplace_back([&, t] {
      for (int i = 0; i < kPer; ++i) {
        Request r = make_request(
            s.next_request_id(),
            (t * 7 + i) % 5,
            (i % 3 == 0) ? 1 : 4,
            static_cast<Position>(i));
        auto f = s.submit(std::move(r));
        // Bounded: a lost request must end this producer rather than block it,
        // or the join below would never return and the engine never stop.
        if (f.wait_for(kSettleTimeout) != std::future_status::ready) {
          lost++;
          return;
        }
        if (tokens_of(f.get()).size() == 1u) {
          completed++;
        }
      }
    });
  }
  for (std::thread& t : producers) {
    t.join();
  }
  stop.store(true);
  engine.join();

  EXPECT_EQ(lost.load(), 0) << "a request was never settled";
  EXPECT_EQ(completed.load(), kProducers * kPer);
  EXPECT_EQ(s.queued(), 0u);
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
    auto f = s.submit(make_request(s.next_request_id(), 10, 1, 0));
    Batch b = s.get_work();
    for (const Request& r : b.requests) {
      s.complete(r.request_id, std::vector<Token>{1});
    }
    ASSERT_SETTLES(f);
    (void)f.get();
  }
  stop.store(true);
  observer.join();

  EXPECT_GT(observations.load(), 0) << "observer never ran";
  EXPECT_EQ(s.queued(), 0u);
}
