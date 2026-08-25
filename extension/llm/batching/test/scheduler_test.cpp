/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/scheduler.h>
#include <executorch/extension/llm/batching/types.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using executorch::extension::llm::batching::BatchInput;
using executorch::extension::llm::batching::DecodeFirstScheduler;
using executorch::extension::llm::batching::Input;
using executorch::extension::llm::batching::Position;
using executorch::extension::llm::batching::SamplingParams;
using executorch::extension::llm::batching::Scheduler;
using executorch::extension::llm::batching::SessionId;
using executorch::extension::llm::batching::Task;
using executorch::extension::llm::batching::TaskId;
using executorch::extension::llm::batching::to_batch_input;
using executorch::extension::llm::batching::Token;

namespace {

using SchedulerPtr = std::unique_ptr<DecodeFirstScheduler>;

// Sizes a scheduler by decode cap and chunk size, giving it room for two full
// chunks beside a saturated decode batch. Every case below states its limits
// this way, so batch sizes stay comparable across tests. Always valid: the
// floor is decodes + chunk, and 2 * chunk + decodes clears it for any
// non-zero pair.
SchedulerPtr make_scheduler(
    std::size_t max_decode_sequences,
    std::size_t max_prefill_chunk_size) {
  return DecodeFirstScheduler::create(
      2 * max_prefill_chunk_size + max_decode_sequences,
      max_decode_sequences,
      max_prefill_chunk_size);
}

// Token values are irrelevant to scheduling, so they are all the same.
Task make_task(
    TaskId task_id,
    SessionId session,
    std::size_t n_tokens,
    Position position,
    bool is_decode,
    bool produce_output = true) {
  auto tokens = std::make_shared<std::vector<Token>>(n_tokens, 7);
  Task task;
  task.tid = task_id;
  task.cancelled = false;
  task.is_decode = is_decode;
  task.input = Input{
      session,
      produce_output,
      0,
      n_tokens,
      std::move(tokens),
      position,
      SamplingParams{}};
  return task;
}

// One chunk of a prompt. A chunk of one token is still prefill, which is the
// distinction is_decode exists to make.
Task prefill(
    TaskId task_id,
    SessionId session,
    std::size_t n_tokens,
    Position position,
    bool produce_output = true) {
  return make_task(
      task_id,
      session,
      n_tokens,
      position,
      /*is_decode=*/false,
      produce_output);
}

// One decode step, always exactly one token.
Task decode(TaskId task_id, SessionId session, Position position = 0) {
  return make_task(
      task_id, session, /*n_tokens=*/1, position, /*is_decode=*/true);
}

bool submit(DecodeFirstScheduler& scheduler, Task task) {
  std::vector<Task> tasks;
  tasks.push_back(std::move(task));
  return scheduler.submit(std::move(tasks));
}

std::vector<TaskId> ids(const std::vector<Task>& tasks) {
  std::vector<TaskId> out;
  out.reserve(tasks.size());
  for (const Task& task : tasks) {
    out.push_back(task.tid);
  }
  return out;
}

std::vector<TaskId> sorted_ids(const std::vector<Task>& tasks) {
  std::vector<TaskId> out = ids(tasks);
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<SessionId> sessions(const std::vector<Task>& tasks) {
  std::vector<SessionId> out;
  out.reserve(tasks.size());
  for (const Task& task : tasks) {
    out.push_back(task.input.sid);
  }
  return out;
}

std::size_t token_count(const std::vector<Task>& tasks) {
  std::size_t count = 0;
  for (const Task& task : tasks) {
    count += task.input.size;
  }
  return count;
}

// The executor is promised at most one produce_output per session. A session
// may still appear more than once, because several consecutive prefill chunks
// in one batch form one wider prefill, and only the last asks for output.
bool at_most_one_output_per_session(const std::vector<Task>& tasks) {
  std::set<SessionId> producing;
  for (const Task& task : tasks) {
    if (task.input.produce_output && !producing.insert(task.input.sid).second) {
      return false;
    }
  }
  return true;
}

// Stricter, for batches whose tasks are all decodes. A decode adjoins nothing,
// so a session may hold only one slot.
bool at_most_one_task_per_session(const std::vector<Task>& tasks) {
  std::set<SessionId> seen;
  for (const Task& task : tasks) {
    if (!seen.insert(task.input.sid).second) {
      return false;
    }
  }
  return true;
}

} // namespace

static_assert(
    std::is_abstract<Scheduler>::value,
    "Scheduler is an interface and tests must use an implementation");
static_assert(
    std::is_base_of<Scheduler, DecodeFirstScheduler>::value,
    "DecodeFirstScheduler must implement Scheduler");

// --- construction ----------------------------------------------------------

TEST(CreateTest, Defaults) {
  SchedulerPtr scheduler = DecodeFirstScheduler::create();
  ASSERT_NE(scheduler, nullptr);
  EXPECT_EQ(scheduler->max_batch_tokens(), 544u);
  EXPECT_EQ(scheduler->max_decode_sequences(), 32u);
  EXPECT_EQ(scheduler->max_prefill_chunk_size(), 256u);
}

TEST(CreateTest, RejectsZeroLimits) {
  EXPECT_EQ(DecodeFirstScheduler::create(0, 32, 256), nullptr);
  EXPECT_EQ(DecodeFirstScheduler::create(544, 0, 256), nullptr);
  EXPECT_EQ(DecodeFirstScheduler::create(544, 32, 0), nullptr);
}

// Below this floor a full-size chunk could be admitted and then never fit in
// any batch, leaving the task queued forever rather than merely delayed.
TEST(CreateTest, RequiresRoomForDecodesPlusAFullChunk) {
  EXPECT_NE(DecodeFirstScheduler::create(288, 32, 256), nullptr)
      << "exactly the floor must be accepted";
  EXPECT_EQ(DecodeFirstScheduler::create(287, 32, 256), nullptr);
  EXPECT_EQ(DecodeFirstScheduler::create(100, 32, 256), nullptr);
}

TEST(CreateTest, RejectsDecodeCapThatLeavesNoRoomForPrefill) {
  EXPECT_EQ(DecodeFirstScheduler::create(544, 544, 256), nullptr);
  EXPECT_EQ(DecodeFirstScheduler::create(544, 600, 256), nullptr);
}

// The limits are size_t, so an absurd chunk size is representable and must be
// caught by the floor rather than wrapping into something that passes.
TEST(CreateTest, RejectsChunkLargerThanTheBudget) {
  constexpr std::size_t kHuge = std::size_t{1} << 40;
  EXPECT_EQ(DecodeFirstScheduler::create(544, 32, kHuge), nullptr);
  EXPECT_NE(
      DecodeFirstScheduler::create(
          std::numeric_limits<std::size_t>::max(), 32, 256),
      nullptr)
      << "only the floor constrains the budget; spending accumulates toward it "
         "and never wraps, so a large one is not itself an error";
}

// --- decode and prefill are distinguished by is_decode, not by size ---------

// Inferring the kind from input.size would send a one-token final chunk to the
// decode queue, ahead of queued decodes and outside the prefill rotation.
TEST(ClassificationTest, OneTokenPrefillIsNotADecode) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  ASSERT_TRUE(submit(*scheduler, prefill(1, 10, 1, 0)));
  ASSERT_TRUE(submit(*scheduler, decode(2, 20)));

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{2, 1}))
      << "the real decode must be taken before the one-token prefill";
}

// take_decodes_ spends one token per decode, and the create() floor is written
// in those terms, so a wider decode would overspend the budget.
TEST(ClassificationTest, RejectsDecodeWiderThanOneToken) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  EXPECT_FALSE(submit(*scheduler, make_task(1, 10, 2, 0, /*is_decode=*/true)));
  EXPECT_FALSE(submit(*scheduler, make_task(2, 10, 4, 0, /*is_decode=*/true)));
  EXPECT_FALSE(scheduler->has_work());

  EXPECT_TRUE(submit(*scheduler, decode(3, 10)));
}

// --- admission -------------------------------------------------------------

TEST(SubmitTest, RejectsEmptyStep) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  EXPECT_FALSE(submit(*scheduler, prefill(1, 10, 0, 0)));
  EXPECT_FALSE(scheduler->has_work());
  EXPECT_TRUE(scheduler->get_work().empty());
}

TEST(SubmitTest, RejectsPrefillAboveChunkSize) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  EXPECT_FALSE(submit(*scheduler, prefill(1, 10, 5, 0)));
  EXPECT_FALSE(scheduler->has_work());
}

TEST(SubmitTest, RejectsDuplicateTaskIdAlreadyWaiting) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  EXPECT_TRUE(submit(*scheduler, decode(7, 10)));
  EXPECT_FALSE(submit(*scheduler, decode(7, 20)));

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{7}));
  EXPECT_FALSE(scheduler->has_work()) << "rejection must not add a task";
}

// admissible_ tests pending_, which the vector being submitted has not joined
// yet, so a repeat inside one submit has to be caught separately. Accepting it
// would queue a task that pending_ never recorded.
TEST(SubmitTest, RejectsTaskIdRepeatedWithinOneSubmit) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  std::vector<Task> tasks;
  tasks.push_back(prefill(5, 10, 4, 0, /*produce_output=*/false));
  tasks.push_back(prefill(5, 10, 4, 4));

  EXPECT_FALSE(scheduler->submit(std::move(tasks)));
  EXPECT_FALSE(scheduler->has_work());
  EXPECT_TRUE(scheduler->get_work().empty());
  EXPECT_TRUE(scheduler->clear().empty()) << "neither copy may be retained";
}

TEST(SubmitTest, RejectedGroupEntersNoQueue) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  std::vector<Task> tasks;
  tasks.push_back(prefill(1, 10, 4, 0));
  tasks.push_back(prefill(2, 10, 0, 4));
  tasks.push_back(prefill(3, 10, 4, 4));

  EXPECT_FALSE(scheduler->submit(std::move(tasks)));
  EXPECT_FALSE(scheduler->has_work());
  EXPECT_TRUE(scheduler->get_work().empty());
}

TEST(SubmitTest, DuplicatePendingIdRejectsTheWholeGroup) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  EXPECT_TRUE(submit(*scheduler, decode(7, 10)));

  std::vector<Task> tasks;
  tasks.push_back(prefill(8, 20, 4, 0));
  tasks.push_back(prefill(7, 20, 4, 4));
  EXPECT_FALSE(scheduler->submit(std::move(tasks)));

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{7}));
  EXPECT_FALSE(scheduler->has_work())
      << "task 8 must not be partially accepted";
}

// Ids must be unique among queued tasks, not for all time.
TEST(SubmitTest, DispatchFreesTheTaskIdForReuse) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  EXPECT_TRUE(submit(*scheduler, decode(7, 10)));
  ASSERT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{7}));

  EXPECT_TRUE(submit(*scheduler, decode(7, 10, 1)));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{7}));
}

TEST(SubmitTest, CancelFreesTheTaskIdForReuse) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  EXPECT_TRUE(submit(*scheduler, decode(7, 10)));
  ASSERT_EQ(ids(scheduler->cancel(10)), (std::vector<TaskId>{7}));

  EXPECT_TRUE(submit(*scheduler, decode(7, 10, 1)));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{7}));
}

// --- payload passthrough ---------------------------------------------------

TEST(PayloadTest, CarriesPayloadUninspected) {
  SchedulerPtr scheduler = make_scheduler(1, 8);
  auto tokens = std::make_shared<std::vector<Token>>(
      std::initializer_list<Token>{10, 11, 12, 13, 14, 15});
  Task task = prefill(1, 40, 4, 90, /*produce_output=*/false);
  task.input.tokens = tokens;
  task.input.offset = 1;
  task.input.sampling_params.temperature = 0.7f;
  task.input.sampling_params.top_p = 0.8f;
  task.input.sampling_params.top_k = 9;
  EXPECT_TRUE(submit(*scheduler, std::move(task)));

  std::vector<Task> work = scheduler->get_work();
  ASSERT_EQ(work.size(), 1u);
  EXPECT_EQ(work[0].input.sid, 40);
  EXPECT_FALSE(work[0].input.produce_output);
  EXPECT_EQ(work[0].input.offset, 1u);
  EXPECT_EQ(work[0].input.size, 4u);
  EXPECT_EQ(work[0].input.tokens.get(), tokens.get());
  EXPECT_EQ(work[0].input.position, 90);
  EXPECT_FLOAT_EQ(work[0].input.sampling_params.temperature, 0.7f);
  EXPECT_FLOAT_EQ(work[0].input.sampling_params.top_p, 0.8f);
  EXPECT_EQ(work[0].input.sampling_params.top_k, 9);
}

TEST(PayloadTest, BatchInputKeepsTaskOrderAndSlices) {
  SchedulerPtr scheduler = make_scheduler(2, 8);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10)));
  EXPECT_TRUE(submit(*scheduler, decode(2, 20)));
  EXPECT_TRUE(submit(*scheduler, prefill(3, 30, 6, 0)));
  EXPECT_TRUE(submit(*scheduler, prefill(4, 40, 5, 100)));

  std::vector<Task> work = scheduler->get_work();
  EXPECT_EQ(ids(work), (std::vector<TaskId>{1, 2, 3, 4}));
  EXPECT_EQ(token_count(work), 1u + 1u + 6u + 5u);

  BatchInput batch = to_batch_input(work);
  EXPECT_EQ(batch.size(), 1u + 1u + 6u + 5u);
  EXPECT_EQ(batch.inputs.size(), 4u);
}

// --- decode scheduling -----------------------------------------------------

TEST(DecodeTest, ServedInArrivalOrderUpToTheCap) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10)));
  EXPECT_TRUE(submit(*scheduler, decode(2, 20)));
  EXPECT_TRUE(submit(*scheduler, decode(3, 30)));

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1, 2}));
  EXPECT_TRUE(scheduler->has_work());
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{3}));
  EXPECT_FALSE(scheduler->has_work());
}

// A shorter queue must not give a later arrival a head start.
TEST(DecodeTest, StaysFifoAcrossDrainAndRefill) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10)));
  EXPECT_TRUE(submit(*scheduler, decode(2, 20)));
  EXPECT_TRUE(submit(*scheduler, decode(3, 30)));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1, 2}));

  EXPECT_TRUE(submit(*scheduler, decode(4, 40)));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{3, 4}));
}

TEST(DecodeTest, BeatsPrefillAndStillLeavesRoomForAFullChunk) {
  SchedulerPtr scheduler = make_scheduler(3, 4); // batch = 11
  EXPECT_TRUE(submit(*scheduler, decode(1, 10)));
  EXPECT_TRUE(submit(*scheduler, decode(2, 20)));
  EXPECT_TRUE(submit(*scheduler, decode(3, 30)));
  EXPECT_TRUE(submit(*scheduler, prefill(50, 90, 4, 0)));

  std::vector<Task> work = scheduler->get_work();
  EXPECT_EQ(ids(work), (std::vector<TaskId>{1, 2, 3, 50}));
  EXPECT_EQ(token_count(work), 3u + 4u);
}

// --- prefill scheduling ----------------------------------------------------

TEST(PrefillTest, SubmittedPromptChunksStayInOrder) {
  SchedulerPtr scheduler = make_scheduler(1, 2);
  std::vector<Task> prompt;
  prompt.push_back(prefill(1, 7, 2, 0, /*produce_output=*/false));
  prompt.push_back(prefill(2, 7, 2, 2));
  ASSERT_TRUE(scheduler->submit(std::move(prompt)));

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1, 2}));
}

TEST(PrefillTest, LongPromptCannotHogTheBatch) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  for (int chunk = 0; chunk < 4; ++chunk) {
    EXPECT_TRUE(submit(
        *scheduler,
        prefill(1 + chunk, 10, 4, static_cast<Position>(chunk * 4))));
  }
  EXPECT_TRUE(submit(*scheduler, prefill(5, 20, 4, 0)));
  EXPECT_TRUE(submit(*scheduler, prefill(6, 20, 4, 4)));

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1, 5}));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{2, 6}));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{3, 4}))
      << "session 20 drained, so session 10 may take two chunks";
  EXPECT_FALSE(scheduler->has_work());
}

TEST(PrefillTest, LoneSessionFillsTheBatchAcrossPasses) {
  SchedulerPtr scheduler = make_scheduler(1, 4); // batch = 9
  EXPECT_TRUE(submit(*scheduler, prefill(1, 10, 4, 0, false)));
  EXPECT_TRUE(submit(*scheduler, prefill(2, 10, 4, 4)));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1, 2}));
}

TEST(PrefillTest, StopsWhenTheNextChunkDoesNotFit) {
  SchedulerPtr scheduler = make_scheduler(1, 4); // batch = 9
  EXPECT_TRUE(submit(*scheduler, prefill(1, 10, 4, 0)));
  EXPECT_TRUE(submit(*scheduler, prefill(2, 20, 4, 0)));
  EXPECT_TRUE(submit(*scheduler, prefill(3, 30, 4, 0)));

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1, 2}));
  EXPECT_TRUE(scheduler->has_work());
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{3}));
}

// A session skipped on size was reached; one the pass never got to was not.
// Both took nothing, so they must keep their original relative order.
TEST(PrefillTest, DeferredSessionOutranksOneNeverReached) {
  SchedulerPtr scheduler = make_scheduler(1, 4); // batch = 9
  EXPECT_TRUE(submit(*scheduler, prefill(1, 10, 4, 0))); // 9 -> 5
  EXPECT_TRUE(submit(*scheduler, prefill(2, 20, 3, 0))); // 5 -> 2
  EXPECT_TRUE(submit(*scheduler, prefill(3, 30, 4, 0))); // deferred
  EXPECT_TRUE(submit(*scheduler, prefill(4, 40, 2, 0))); // 2 -> 0
  EXPECT_TRUE(submit(*scheduler, prefill(5, 50, 4, 0))); // not reached

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1, 2, 4}));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{3, 5}));
}

TEST(PrefillTest, DeferredSessionsKeepTheirOrder) {
  SchedulerPtr scheduler = make_scheduler(1, 4); // batch = 9
  EXPECT_TRUE(submit(*scheduler, prefill(1, 10, 4, 0)));
  EXPECT_TRUE(submit(*scheduler, prefill(2, 20, 4, 0)));
  EXPECT_TRUE(submit(*scheduler, prefill(3, 30, 4, 0)));
  EXPECT_TRUE(submit(*scheduler, prefill(4, 40, 3, 0)));

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1, 2}));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{3, 4}));
}

// Two long prompts must not starve a third: served sessions rotate to the back.
TEST(PrefillTest, RotationIsFairAcrossCalls) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  TaskId task_id = 1;
  for (SessionId session : {10, 20, 30}) {
    for (int chunk = 0; chunk < 8; ++chunk) {
      EXPECT_TRUE(submit(
          *scheduler,
          prefill(task_id++, session, 4, static_cast<Position>(chunk * 4))));
    }
  }

  std::map<SessionId, int> served;
  for (int call = 0; call < 9; ++call) {
    for (const Task& task : scheduler->get_work()) {
      served[task.input.sid]++;
    }
  }
  EXPECT_EQ(served[10], 6);
  EXPECT_EQ(served[20], 6);
  EXPECT_EQ(served[30], 6);
}

TEST(PrefillTest, SessionRejoinsTheRotationAfterDraining) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  EXPECT_TRUE(submit(*scheduler, prefill(1, 10, 4, 0)));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1}));
  EXPECT_TRUE(scheduler->get_work().empty());

  EXPECT_TRUE(submit(*scheduler, prefill(2, 10, 4, 4)));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{2}));

  EXPECT_TRUE(submit(*scheduler, prefill(3, 10, 4, 8)));
  EXPECT_TRUE(submit(*scheduler, prefill(4, 20, 4, 0)));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{3, 4}));
}

// --- what one batch may hold for one session -------------------------------

// The executor is promised ranges that adjoin and a single produce_output per
// session. Consecutive prefill chunks satisfy that; two decodes do not.
TEST(BatchCompositionTest, OneDecodePerSessionPerBatch) {
  SchedulerPtr scheduler = make_scheduler(8, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10, 0)));
  EXPECT_TRUE(submit(*scheduler, decode(2, 10, 1)));
  EXPECT_TRUE(submit(*scheduler, decode(3, 10, 2)));
  EXPECT_TRUE(submit(*scheduler, decode(4, 20, 0)));

  std::vector<Task> first = scheduler->get_work();
  EXPECT_EQ(ids(first), (std::vector<TaskId>{1, 4}));
  EXPECT_TRUE(at_most_one_task_per_session(first));

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{2}))
      << "deferred decodes keep their arrival order";
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{3}));
  EXPECT_FALSE(scheduler->has_work());
}

TEST(BatchCompositionTest, SecondDecodeForASessionRunsInTheNextBatch) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10, 5)));
  EXPECT_TRUE(submit(*scheduler, decode(2, 10, 5)));

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1}));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{2}));
  EXPECT_FALSE(scheduler->has_work());
}

// A decode and a prefill chunk for one session are two ranges that do not
// adjoin, and both ask to produce output.
TEST(BatchCompositionTest, PrefillWaitsWhileTheSessionHasADecode) {
  SchedulerPtr scheduler = make_scheduler(8, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10, 0)));
  EXPECT_TRUE(submit(*scheduler, prefill(2, 10, 4, 8)));
  EXPECT_TRUE(submit(*scheduler, prefill(3, 20, 4, 0)));

  std::vector<Task> first = scheduler->get_work();
  EXPECT_EQ(ids(first), (std::vector<TaskId>{1, 3}));
  EXPECT_TRUE(at_most_one_task_per_session(first));

  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{2}));
}

// Several chunks of one prompt are allowed together, because they adjoin and
// only the last asks for output.
TEST(BatchCompositionTest, ConsecutivePrefillChunksMaySharePlaces) {
  SchedulerPtr scheduler = make_scheduler(1, 4); // batch = 9
  std::vector<Task> prompt;
  prompt.push_back(prefill(1, 10, 4, 0, /*produce_output=*/false));
  prompt.push_back(prefill(2, 10, 4, 4));
  ASSERT_TRUE(scheduler->submit(std::move(prompt)));

  std::vector<Task> work = scheduler->get_work();
  EXPECT_EQ(ids(work), (std::vector<TaskId>{1, 2}));
  EXPECT_TRUE(at_most_one_output_per_session(work));
}

// --- cancellation ----------------------------------------------------------

TEST(CancelTest, DropsEveryQueuedTaskForTheSessionAndReturnsThem) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  std::vector<Task> prompt;
  prompt.push_back(prefill(1, 77, 4, 0, /*produce_output=*/false));
  prompt.push_back(prefill(2, 77, 4, 4));
  ASSERT_TRUE(scheduler->submit(std::move(prompt)));

  std::vector<Task> dropped = scheduler->cancel(77);
  EXPECT_EQ(ids(dropped), (std::vector<TaskId>{1, 2}));
  for (const Task& task : dropped) {
    EXPECT_TRUE(task.cancelled);
  }
  EXPECT_FALSE(scheduler->has_work());
  EXPECT_TRUE(scheduler->get_work().empty());
}

TEST(CancelTest, DropsEveryQueuedDecodeForTheSession) {
  SchedulerPtr scheduler = make_scheduler(4, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10, 0)));
  EXPECT_TRUE(submit(*scheduler, decode(2, 20, 0)));
  EXPECT_TRUE(submit(*scheduler, decode(3, 10, 1)));

  EXPECT_EQ(sorted_ids(scheduler->cancel(10)), (std::vector<TaskId>{1, 3}))
      << "every queued decode for the session must be dropped, not just one";
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{2}));
  EXPECT_FALSE(scheduler->has_work());
}

TEST(CancelTest, LeavesOtherSessionsRunnable) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10)));
  EXPECT_TRUE(submit(*scheduler, decode(2, 20)));

  EXPECT_EQ(ids(scheduler->cancel(10)), (std::vector<TaskId>{1}));
  EXPECT_TRUE(scheduler->has_work());
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{2}));
}

TEST(CancelTest, UnknownSessionIsIgnored) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  EXPECT_TRUE(scheduler->cancel(404).empty());
  EXPECT_FALSE(scheduler->has_work());
}

TEST(CancelTest, DoubleCancelReportsEachTaskOnce) {
  SchedulerPtr scheduler = make_scheduler(4, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10, 0)));
  EXPECT_TRUE(submit(*scheduler, prefill(2, 10, 4, 8)));

  EXPECT_EQ(sorted_ids(scheduler->cancel(10)), (std::vector<TaskId>{1, 2}));
  EXPECT_TRUE(scheduler->cancel(10).empty());
  EXPECT_FALSE(scheduler->has_work());
}

// A task handed out by get_work() is no longer owned by the scheduler.
TEST(CancelTest, DispatchedTaskIsNoLongerTracked) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  EXPECT_TRUE(submit(*scheduler, prefill(9, 77, 4, 12)));
  ASSERT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{9}));

  EXPECT_TRUE(scheduler->cancel(77).empty());
  EXPECT_FALSE(scheduler->has_work());
}

TEST(CancelTest, CancellingDispatchedTaskDoesNotHideQueuedWork) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10)));
  ASSERT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1}));
  EXPECT_TRUE(submit(*scheduler, decode(2, 20)));

  EXPECT_TRUE(scheduler->cancel(10).empty());
  EXPECT_TRUE(scheduler->has_work());
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{2}));
}

// A cancelled entry stays in its deque until scheduling reaches it. Skipping it
// must not disturb live tasks before or after it.
TEST(CancelTest, CancelledDecodeInTheMiddleOfTheQueue) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10)));
  EXPECT_TRUE(submit(*scheduler, decode(2, 20)));
  EXPECT_TRUE(submit(*scheduler, decode(3, 30)));

  EXPECT_EQ(ids(scheduler->cancel(20)), (std::vector<TaskId>{2}));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1}));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{3}));
  EXPECT_FALSE(scheduler->has_work());
}

TEST(CancelTest, CancelledPrefillSessionLeavesTheRotation) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  std::vector<Task> first_prompt;
  first_prompt.push_back(prefill(1, 10, 4, 0, /*produce_output=*/false));
  first_prompt.push_back(prefill(2, 10, 4, 4));
  ASSERT_TRUE(scheduler->submit(std::move(first_prompt)));
  EXPECT_TRUE(submit(*scheduler, prefill(3, 20, 4, 0)));

  EXPECT_EQ(ids(scheduler->cancel(10)), (std::vector<TaskId>{1, 2}));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{3}));
}

// The batch in between retires the cancelled session's rotation slot, so
// rejoining after one is the easy case.
TEST(CancelTest, SessionRejoinsAfterAnInterveningBatch) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  EXPECT_TRUE(submit(*scheduler, prefill(1, 10, 4, 0)));
  ASSERT_EQ(ids(scheduler->cancel(10)), (std::vector<TaskId>{1}));
  EXPECT_TRUE(scheduler->get_work().empty());

  EXPECT_TRUE(submit(*scheduler, prefill(2, 10, 4, 0)));
  EXPECT_TRUE(submit(*scheduler, prefill(3, 20, 4, 0)));
  EXPECT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{2, 3}));
}

// Regression: cancelling used to erase the session's queue while leaving its
// slot in the rotation, so rejoining took a second slot and the session got two
// turns per pass, starving everyone else. There must be no get_work() between
// the cancel and the resubmit, since that would retire the stale slot and hide
// the bug.
TEST(CancelTest, RejoiningAfterCancelDoesNotTakeTwoTurnsPerPass) {
  SchedulerPtr scheduler = make_scheduler(1, 4); // batch = 9
  EXPECT_TRUE(submit(*scheduler, prefill(1, 10, 4, 0)));
  ASSERT_EQ(ids(scheduler->cancel(10)), (std::vector<TaskId>{1}));

  EXPECT_TRUE(submit(*scheduler, prefill(2, 10, 4, 0, false)));
  EXPECT_TRUE(submit(*scheduler, prefill(3, 10, 4, 4)));
  EXPECT_TRUE(submit(*scheduler, prefill(4, 20, 4, 0, false)));
  EXPECT_TRUE(submit(*scheduler, prefill(5, 20, 4, 4)));

  std::vector<Task> work = scheduler->get_work();
  EXPECT_EQ(sessions(work), (std::vector<SessionId>{10, 20}))
      << "one chunk each, not two for the session that rejoined";
  EXPECT_EQ(ids(work), (std::vector<TaskId>{2, 4}));
}

// --- clear -----------------------------------------------------------------

TEST(ClearTest, ReturnsAndRemovesAllQueuedTasks) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10)));
  EXPECT_TRUE(submit(*scheduler, prefill(2, 20, 4, 0)));

  std::vector<Task> dropped = scheduler->clear();
  EXPECT_EQ(sorted_ids(dropped), (std::vector<TaskId>{1, 2}));
  for (const Task& task : dropped) {
    EXPECT_TRUE(task.cancelled);
  }
  EXPECT_FALSE(scheduler->has_work());
  EXPECT_TRUE(scheduler->get_work().empty());
  EXPECT_TRUE(scheduler->clear().empty());
}

TEST(ClearTest, DoesNotClaimTasksAlreadyHandedToTheCaller) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  EXPECT_TRUE(submit(*scheduler, decode(1, 10)));
  EXPECT_TRUE(submit(*scheduler, decode(2, 20)));
  ASSERT_EQ(ids(scheduler->get_work()), (std::vector<TaskId>{1, 2}));

  EXPECT_TRUE(submit(*scheduler, decode(3, 30)));
  EXPECT_TRUE(submit(*scheduler, prefill(4, 40, 4, 0)));
  EXPECT_EQ(sorted_ids(scheduler->clear()), (std::vector<TaskId>{3, 4}));
  EXPECT_TRUE(scheduler->cancel(10).empty());
  EXPECT_TRUE(scheduler->cancel(20).empty());
  EXPECT_FALSE(scheduler->has_work());
}

// --- has_work --------------------------------------------------------------

TEST(HasWorkTest, EmptySchedulerHasNothingToDo) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  EXPECT_FALSE(scheduler->has_work());
  EXPECT_TRUE(scheduler->get_work().empty());
}

TEST(HasWorkTest, TracksQueuedTasksOnly) {
  SchedulerPtr scheduler = make_scheduler(1, 4);
  EXPECT_FALSE(scheduler->has_work());

  EXPECT_TRUE(submit(*scheduler, decode(1, 10)));
  EXPECT_TRUE(scheduler->has_work());

  scheduler->get_work();
  EXPECT_FALSE(scheduler->has_work()) << "handed-out work is caller-owned";
  EXPECT_TRUE(scheduler->cancel(10).empty());
  EXPECT_FALSE(scheduler->has_work());
}

// --- randomized invariants -------------------------------------------------

// Every accepted task must be returned exactly once, either from get_work() or
// from cancel(). The public has_work() state must agree with that model, and
// no batch may owe one session two outputs.
TEST(InvariantTest, AcceptedTasksAreNeverLostOrReturnedTwice) {
  std::mt19937 rng(1234);
  SchedulerPtr scheduler = make_scheduler(3, 4);

  std::map<TaskId, SessionId> waiting;
  TaskId next_task = 1;
  SessionId next_session = 1;
  int submitted = 0;
  int dispatched = 0;
  int cancelled = 0;

  for (int operation = 0; operation < 4000; ++operation) {
    ASSERT_EQ(scheduler->has_work(), !waiting.empty())
        << "has_work() diverged from the model at operation " << operation;

    switch (rng() % 3) {
      case 0: { // Submit one decode or a group of prompt chunks.
        const SessionId session = next_session++;
        const bool is_decode = rng() % 3 == 0;
        const int count = is_decode ? 1 : static_cast<int>(1 + rng() % 3);
        std::vector<Task> tasks;
        for (int i = 0; i < count; ++i) {
          const TaskId task_id = next_task++;
          const auto position = static_cast<Position>(i * 4);
          // Only a prompt's last chunk asks for output, as the runner builds
          // them, so a batch may legitimately hold several of one session's
          // chunks while still owing it a single output.
          tasks.push_back(
              is_decode ? decode(task_id, session, position)
                        : prefill(
                              task_id,
                              session,
                              2 + rng() % 3,
                              position,
                              /*produce_output=*/i == count - 1));
          waiting.emplace(task_id, session);
          submitted++;
        }
        ASSERT_TRUE(scheduler->submit(std::move(tasks)));
        break;
      }
      case 1: { // Dispatch a batch.
        std::vector<Task> work = scheduler->get_work();
        EXPECT_TRUE(at_most_one_output_per_session(work))
            << "a batch owed one session two outputs at operation "
            << operation;
        for (const Task& task : work) {
          EXPECT_FALSE(task.cancelled);
          EXPECT_EQ(waiting.erase(task.tid), 1u)
              << "a batch returned a task not present in the model";
          dispatched++;
        }
        break;
      }
      case 2: { // Cancel a session with work still waiting.
        if (waiting.empty()) {
          break;
        }
        auto selected = waiting.begin();
        std::advance(
            selected, static_cast<std::ptrdiff_t>(rng() % waiting.size()));
        const SessionId session = selected->second;
        std::vector<TaskId> expected;
        for (const auto& entry : waiting) {
          if (entry.second == session) {
            expected.push_back(entry.first);
          }
        }

        std::vector<Task> dropped = scheduler->cancel(session);
        EXPECT_EQ(sorted_ids(dropped), expected);
        for (const Task& task : dropped) {
          EXPECT_TRUE(task.cancelled);
          EXPECT_EQ(waiting.erase(task.tid), 1u);
          cancelled++;
        }
        break;
      }
      default:
        break;
    }
  }

  while (scheduler->has_work()) {
    std::vector<Task> work = scheduler->get_work();
    ASSERT_FALSE(work.empty()) << "has_work() but get_work() made no progress";
    for (const Task& task : work) {
      EXPECT_EQ(waiting.erase(task.tid), 1u);
      dispatched++;
    }
  }

  EXPECT_TRUE(waiting.empty());
  EXPECT_FALSE(scheduler->has_work());
  EXPECT_EQ(dispatched + cancelled, submitted);
}

// --- concurrency -----------------------------------------------------------

TEST(ConcurrencyTest, ProducersAndConsumerMakeProgressWithoutLoss) {
  SchedulerPtr scheduler = make_scheduler(4, 8);
  constexpr int kProducers = 4;
  constexpr int kPerProducer = 250;
  std::atomic<bool> stop{false};
  std::atomic<TaskId> next_task{1};
  std::atomic<int> accepted{0};
  std::atomic<int> dispatched{0};

  std::thread consumer([&] {
    while (!stop.load() || scheduler->has_work()) {
      std::vector<Task> work = scheduler->get_work();
      dispatched += static_cast<int>(work.size());
      if (work.empty()) {
        std::this_thread::yield();
      }
    }
  });

  std::vector<std::thread> producers;
  for (int producer = 0; producer < kProducers; ++producer) {
    producers.emplace_back([&] {
      for (int i = 0; i < kPerProducer; ++i) {
        const TaskId task_id = next_task.fetch_add(1);
        const auto position = static_cast<Position>(i);
        Task task = (i % 3 == 0) ? decode(task_id, task_id, position)
                                 : prefill(task_id, task_id, 4, position);
        if (submit(*scheduler, std::move(task))) {
          accepted++;
        }
      }
    });
  }
  for (std::thread& producer : producers) {
    producer.join();
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (dispatched.load() < accepted.load() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  stop.store(true);
  consumer.join();

  EXPECT_EQ(accepted.load(), kProducers * kPerProducer)
      << "a valid, uniquely identified task was rejected";
  EXPECT_EQ(dispatched.load(), accepted.load())
      << "an accepted task was never dispatched";
  EXPECT_FALSE(scheduler->has_work());
}

TEST(ConcurrencyTest, ObserversAreSafeDuringScheduling) {
  SchedulerPtr scheduler = make_scheduler(2, 4);
  std::atomic<bool> stop{false};
  std::atomic<bool> observing{false};
  std::atomic<long> observations{0};

  std::thread observer([&] {
    observing.store(true);
    while (!stop.load()) {
      (void)scheduler->has_work();
      (void)scheduler->max_batch_tokens();
      observations++;
    }
  });
  while (!observing.load()) {
    std::this_thread::yield();
  }

  for (TaskId task_id = 1; task_id <= 500; ++task_id) {
    ASSERT_TRUE(submit(*scheduler, decode(task_id, task_id)));
    (void)scheduler->get_work();
  }
  stop.store(true);
  observer.join();

  EXPECT_GT(observations.load(), 0) << "observer never ran";
  EXPECT_FALSE(scheduler->has_work());
}
