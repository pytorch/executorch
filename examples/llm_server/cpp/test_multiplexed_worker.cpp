/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/llm_server/cpp/multiplexed_worker.h>
#include <executorch/examples/llm_server/cpp/multiplexed_worker_test.h>

#include <poll.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <thread>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/tokenizer.h>

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>

namespace {
namespace batching = executorch::extension::llm::batching;
namespace serving = executorch::extension::llm::serving;
using executorch::examples::llm_server::MultiplexedWorkerConfig;
using executorch::examples::llm_server::testing::Checkpoint;
using executorch::examples::llm_server::testing::run_multiplexed_worker;
using executorch::examples::llm_server::testing::WorkerTestHooks;
using Json = nlohmann::json;
using namespace std::chrono_literals;

class Tokenizer : public tokenizers::Tokenizer {
 public:
  std::string piece = "x\n\"";
  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }
  tokenizers::Result<std::string> decode(uint64_t, uint64_t, bool)
      const override {
    return piece;
  }
  tokenizers::Result<std::vector<uint64_t>>
  encode(const std::string& text, int8_t, int8_t) const override {
    return std::vector<uint64_t>(text.begin(), text.end());
  }
  tokenizers::Result<std::string> id_to_piece(uint64_t) const override {
    return piece;
  }
  tokenizers::Result<uint64_t> piece_to_id(const std::string&) const override {
    return uint64_t{1};
  }
};

class Executor : public batching::testing::FakeExecutor {
 public:
  bool execute(const batching::BatchInput& batch, batching::BatchOutput& out)
      override {
    std::this_thread::sleep_for(2ms);
    return FakeExecutor::execute(batch, out);
  }
};

class CheckpointGate {
 public:
  void signal() {
    std::lock_guard<std::mutex> lock(mutex_);
    reached_ = true;
    cv_.notify_all();
  }

  void pause() {
    signal();
    std::unique_lock<std::mutex> lock(mutex_);
    EXPECT_TRUE(cv_.wait_for(lock, 5s, [this] { return released_; }));
  }

  bool wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, 5s, [this] { return reached_; });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool reached_ = false;
  bool released_ = false;
};

class ProtocolTest : public ::testing::Test {
 protected:
  void start(
      MultiplexedWorkerConfig config = {},
      std::size_t max_sessions = 4,
      std::size_t max_requests = 8) {
    signal(SIGPIPE, SIG_IGN);
    ASSERT_EQ(pipe(input_), 0);
    ASSERT_EQ(pipe(output_), 0);
    serving::ServingRuntimeConfig runtime_config;
    runtime_config.max_sessions = max_sessions;
    runtime_config.max_context_length = 8192;
    runtime_config.max_requests = max_requests;
    runtime_config.max_events_per_request = 128;
    runtime_config.max_tokens_per_request = 8192;
    runtime_ = std::make_unique<serving::ServingRuntime>(
        executor_,
        batching::DecodeFirstScheduler::create(16, 8, 8),
        tokenizer_,
        runtime_config);
    result_ = std::async(std::launch::async, [&, config] {
      return run_multiplexed_worker(
          *runtime_, input_[0], output_[1], config, hooks_);
    });
    auto ready = receive();
    ASSERT_TRUE(ready.value("ready", false));
    ASSERT_TRUE(ready.value("multiplexed", false));
    ASSERT_FALSE(ready.contains("request_id"));
    ASSERT_EQ(ready.at("max_named_sessions"), max_sessions);
    ASSERT_EQ(ready.at("max_inflight_requests"), config.max_inflight_requests);
  }

  std::shared_ptr<CheckpointGate> gate() {
    auto result = std::make_shared<CheckpointGate>();
    gates_.push_back(result);
    return result;
  }

  void TearDown() override {
    for (const auto& checkpoint : gates_)
      checkpoint->release();
    executor_.release();
    if (input_[1] >= 0) {
      close(input_[1]);
      input_[1] = -1;
    }
    // Unblock a stalled writer even after an assertion fails.
    if (output_[0] >= 0) {
      close(output_[0]);
      output_[0] = -1;
    }
    if (result_.valid())
      (void)result_.get();
    runtime_.reset();
    for (int fd : {input_[0], input_[1], output_[0], output_[1]})
      if (fd >= 0)
        close(fd);
    EXPECT_EQ(executor_.open_count(), 0);
    EXPECT_EQ(executor_.opened().size(), executor_.closed().size());
  }

  void send_raw(const std::string& bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
      auto count =
          write(input_[1], bytes.data() + offset, bytes.size() - offset);
      ASSERT_GT(count, 0);
      offset += count;
    }
  }
  void send(Json message) {
    send_raw(message.dump() + "\n");
  }
  Json generate(uint64_t id, int count = 3) {
    return {
        {"op", "generate"},
        {"request_id", id},
        {"prompt", "hi"},
        {"max_new_tokens", count}};
  }
  Json receive() {
    std::string frame;
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
      pollfd fd{output_[0], POLLIN, 0};
      if (poll(&fd, 1, 20) <= 0)
        continue;
      char c;
      if (read(output_[0], &c, 1) != 1)
        break;
      if (c == '\n')
        return Json::parse(frame);
      frame += c;
    }
    ADD_FAILURE() << "timed out waiting for complete JSONL record: " << frame;
    return Json::object();
  }
  bool wait_execute() {
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!executor_.in_execute() &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    return executor_.in_execute();
  }
  int finish() {
    close(input_[1]);
    input_[1] = -1;
    if (result_.wait_for(5s) != std::future_status::ready) {
      ADD_FAILURE() << "worker failed to stop";
      return -1;
    }
    return result_.get();
  }

  WorkerTestHooks hooks_;
  std::vector<std::shared_ptr<CheckpointGate>> gates_;
  Tokenizer tokenizer_;
  Executor executor_;
  std::unique_ptr<serving::ServingRuntime> runtime_;
  std::future<int> result_;
  int input_[2]{-1, -1}, output_[2]{-1, -1};
};

TEST_F(ProtocolTest, InterleavedIdsAndOutOfOrderCompletion) {
  start();
  send(generate(20, 40));
  send(generate(10, 2)); // Reserved earlier, arriving later is legal.
  std::vector<uint64_t> completed;
  std::map<uint64_t, std::string> text;
  while (completed.size() != 2) {
    auto msg = receive();
    ASSERT_TRUE(msg.contains("request_id"));
    const auto id = msg.at("request_id").get<uint64_t>();
    ASSERT_TRUE(id == 20 || id == 10);
    ASSERT_FALSE(msg.contains("error")) << msg;
    if (msg.contains("token"))
      text[id] += msg.at("token").get<std::string>();
    else {
      ASSERT_TRUE(msg.value("done", false));
      EXPECT_EQ(msg.at("finish_reason"), "length");
      EXPECT_EQ(msg.at("completion_tokens"), id == 10 ? 2 : 40);
      EXPECT_TRUE(msg.contains("generated_token_ids"));
      completed.push_back(id);
    }
  }
  EXPECT_EQ(completed, (std::vector<uint64_t>{10, 20}));
  EXPECT_EQ(text[10], tokenizer_.piece + tokenizer_.piece);
  EXPECT_EQ(text[20].size(), tokenizer_.piece.size() * 40);
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, LifecycleAndTargetedCancellationWhileGenerationActive) {
  start();
  executor_.hold();
  send(generate(1, 100));
  ASSERT_TRUE(wait_execute());
  send({{"op", "close"}, {"request_id", 2}, {"session_id", "absent"}});
  auto closed = receive();
  EXPECT_EQ(closed.at("request_id"), 2);
  EXPECT_TRUE(closed.value("closed", false));
  send({{"op", "cancel"}, {"request_id", 3}, {"target_request_id", 1}});
  auto ack = receive();
  EXPECT_EQ(ack.at("request_id"), 3);
  EXPECT_TRUE(ack.value("cancelled", false));
  send(generate(6, 2));
  executor_.release();
  int completed = 0;
  while (completed != 2) {
    auto done = receive();
    if (done.contains("token"))
      continue;
    ASSERT_TRUE(done.value("done", false)) << done;
    const bool cancelled = done.at("request_id") == 1;
    EXPECT_EQ(done.value("cancelled", false), cancelled);
    EXPECT_EQ(done.at("finish_reason"), cancelled ? "stop" : "length");
    ++completed;
  }
  send({{"op", "cancel"}, {"request_id", 4}, {"target_request_id", 1}});
  EXPECT_TRUE(receive().value("cancelled", false));
  send({{"op", "cancel"}, {"request_id", 5}, {"target_request_id", 999}});
  EXPECT_TRUE(receive().value("cancelled", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, BusyAndMissingSessionErrorsKeepTheirWireCodes) {
  start();
  executor_.hold();
  auto active = generate(1, 100);
  active["session_id"] = "owned";
  send(active);
  ASSERT_TRUE(wait_execute());
  auto overlap = generate(2);
  overlap["session_id"] = "owned";
  send(overlap);
  auto busy = receive();
  EXPECT_EQ(busy.at("request_id"), 2);
  EXPECT_EQ(busy.at("code"), "session_busy");
  send({{"op", "reset"}, {"request_id", 3}, {"session_id", "absent"}});
  auto missing = receive();
  EXPECT_EQ(missing.at("request_id"), 3);
  EXPECT_EQ(missing.at("code"), "session_not_found");
  send({{"op", "cancel"}, {"request_id", 4}, {"target_request_id", 1}});
  EXPECT_TRUE(receive().value("cancelled", false));
  executor_.release();
  Json done;
  do {
    done = receive();
  } while (done.contains("token"));
  EXPECT_TRUE(done.value("cancelled", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, EofSettlesOutstandingGenerationAndLifecycleCommands) {
  start();
  executor_.hold();
  send(generate(1, 100));
  ASSERT_TRUE(wait_execute());
  send({{"op", "open"}, {"request_id", 2}, {"session_id", "other"}});
  send({{"op", "reset"}, {"request_id", 3}, {"session_id", "other"}});
  send({{"op", "close"}, {"request_id", 4}, {"session_id", "other"}});
  close(input_[1]);
  input_[1] = -1;
  executor_.release();
  ASSERT_EQ(result_.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(result_.get(), 0);
  std::map<uint64_t, int> terminals;
  while (terminals.size() != 4) {
    const auto message = receive();
    ASSERT_TRUE(message.contains("request_id")) << message;
    if (message.contains("token"))
      continue;
    const auto id = message.at("request_id").get<uint64_t>();
    ASSERT_GE(id, 1u);
    ASSERT_LE(id, 4u);
    EXPECT_EQ(++terminals[id], 1);
    EXPECT_TRUE(
        message.contains("error") || message.value("done", false) ||
        message.value("opened", false) || message.value("reset", false) ||
        message.value("closed", false))
        << message;
  }
  EXPECT_EQ(executor_.open_count(), 0);
}

TEST_F(ProtocolTest, IncompleteUtf8IsReplacedWithoutFailingOtherRequests) {
  tokenizer_.piece = "\xE2";
  start();
  send(generate(1, 1));
  send(generate(2, 1));
  std::map<uint64_t, std::string> text;
  int completed = 0;
  while (completed != 2) {
    auto msg = receive();
    ASSERT_TRUE(msg.contains("request_id"));
    ASSERT_FALSE(msg.contains("error")) << msg;
    const auto id = msg.at("request_id").get<uint64_t>();
    if (msg.contains("token"))
      text[id] += msg.at("token").get<std::string>();
    else {
      ASSERT_TRUE(msg.value("done", false));
      ++completed;
    }
  }
  EXPECT_EQ(text[1], "\xEF\xBF\xBD");
  EXPECT_EQ(text[2], "\xEF\xBF\xBD");
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, AdmissionFailureDoesNotPoisonOtherRequests) {
  start({}, 1);
  send({{"op", "open"}, {"request_id", 1}, {"session_id", "owned"}});
  EXPECT_TRUE(receive().value("opened", false));
  send(generate(2));
  auto error = receive();
  EXPECT_EQ(error.at("request_id"), 2);
  EXPECT_EQ(error.at("code"), "capacity_exhausted");
  send({{"op", "reset"}, {"request_id", 3}, {"session_id", "owned"}});
  EXPECT_TRUE(receive().value("reset", false));
  auto request = generate(4);
  request["session_id"] = "owned";
  request.erase("prompt");
  request["prompt_segments"] =
      Json::array({{{"text", "h"}}, {{"ids", {7, 8}}}});
  send(request);
  Json done;
  do {
    done = receive();
  } while (done.contains("token"));
  EXPECT_TRUE(done.value("done", false));
  EXPECT_EQ(done.at("prompt_tokens"), 3);
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, StrictFieldsAndUnsupportedModalitiesAreIsolated) {
  start();
  uint64_t id = 1;
  std::vector<Json> invalid;
  for (auto field :
       {"seed", "temperature", "top_p", "top_k", "max_new_tokens"}) {
    auto request = generate(id++);
    request[field] = true;
    invalid.push_back(request);
  }
  for (auto value :
       {Json(-2),
        Json(1.5),
        Json(uint64_t{1} << 40),
        Json(std::numeric_limits<uint64_t>::max())}) {
    auto request = generate(id++);
    request["max_new_tokens"] = value;
    invalid.push_back(request);
  }
  auto image = generate(id++);
  image["image"] = "ignored?";
  invalid.push_back(image);
  image = generate(id++);
  image.erase("prompt");
  image["prompt_segments"] =
      Json::array({{{"type", "image"}, {"image", "data"}}});
  invalid.push_back(image);
  auto both = generate(id++);
  both["prompt_segments"] = Json::array();
  invalid.push_back(both);
  auto bad_ids = generate(id++);
  bad_ids.erase("prompt");
  bad_ids["prompt_segments"] = Json::array({{{"ids", {1, -1}}}});
  invalid.push_back(bad_ids);
  auto ambiguous = generate(id++);
  ambiguous.erase("prompt");
  ambiguous["prompt_segments"] = Json::array({{{"text", "h"}, {"ids", {1}}}});
  invalid.push_back(ambiguous);
  auto hidden_image = generate(id++);
  hidden_image.erase("prompt");
  hidden_image["prompt_segments"] =
      Json::array({{{"text", "h"}, {"image", "data"}}});
  invalid.push_back(hidden_image);
  invalid.push_back(
      {{"op", "open"}, {"request_id", id++}, {"session_id", false}});
  invalid.push_back(
      {{"op", "cancel"}, {"request_id", id++}, {"target_request_id", 1.5}});
  for (const auto& request : invalid) {
    send(request);
    const auto error = receive();
    EXPECT_EQ(error.at("request_id"), request.at("request_id"));
    EXPECT_EQ(error.at("code"), "invalid_argument") << error;
  }
  send({{"op", "close"}, {"request_id", id}, {"session_id", "absent"}});
  EXPECT_TRUE(receive().value("closed", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, ZeroSeedRemainsUnsetAndPositiveSeedIsPreserved) {
  start();
  uint64_t request_id = 1;
  for (uint64_t seed : {uint64_t{0}, std::numeric_limits<uint64_t>::max()}) {
    auto request = generate(request_id++, 1);
    request["seed"] = seed;
    send(request);
    Json done;
    do {
      done = receive();
    } while (done.contains("token"));
    ASSERT_TRUE(done.value("done", false)) << done;
    const auto seen = executor_.seen();
    ASSERT_FALSE(seen.empty());
    EXPECT_EQ(
        seen.back().sampling_seed,
        seed == 0 ? std::nullopt : std::optional<uint64_t>(seed));
  }
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, OversizedInputFailsBoundedly) {
  MultiplexedWorkerConfig config;
  config.max_frame_bytes = 1024;
  start(config);
  send_raw(std::string(1025, 'x') + "\n");
  ASSERT_EQ(result_.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(result_.get(), 1);
}

class InvalidIdTest : public ProtocolTest,
                      public ::testing::WithParamInterface<const char*> {};

TEST_P(InvalidIdTest, RejectsUncorrelatableFrame) {
  start();
  send_raw(
      std::string("{\"request_id\":") + GetParam() + ",\"op\":\"close\"}\n");
  ASSERT_EQ(result_.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(result_.get(), 1);
}

INSTANTIATE_TEST_SUITE_P(
    StrictIds,
    InvalidIdTest,
    ::testing::Values(
        "true",
        "0",
        "-1",
        "1.5",
        "18446744073709551616",
        "null",
        "\"1\""));

TEST_F(ProtocolTest, IncompleteInputFailsAtEof) {
  start();
  send_raw("{\"request_id\":1");
  EXPECT_EQ(finish(), 1);
}

TEST_F(ProtocolTest, MaxUint64IdAndAutoBudgetAreAccepted) {
  start();
  auto request = generate(std::numeric_limits<uint64_t>::max(), -1);
  request["stop"] = Json::array({"x"});
  send(request);
  auto done = receive();
  EXPECT_EQ(done.at("request_id"), std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(done.value("done", false));
  EXPECT_EQ(done.at("finish_reason"), "stop");
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, OversizedTerminalIsAnExplicitError) {
  MultiplexedWorkerConfig config;
  config.max_frame_bytes = 1024;
  start(config);
  send(generate(1, 300));
  Json terminal;
  do {
    terminal = receive();
  } while (terminal.contains("token"));
  EXPECT_EQ(terminal.at("request_id"), 1);
  EXPECT_EQ(terminal.at("code"), "frame_too_large");
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, DuplicateInflightIdIsRejectedWithoutAnotherAdmission) {
  start();
  executor_.hold();
  send(generate(1));
  ASSERT_TRUE(wait_execute());
  send(generate(1));
  executor_.release();
  ASSERT_EQ(result_.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(result_.get(), 1);
  EXPECT_EQ(executor_.opened().size(), 1u);
}

TEST_F(ProtocolTest, BacklogCancelsOnlyAffectedRequestAndKeepsTerminalBudget) {
  MultiplexedWorkerConfig config;
  config.token_bytes_per_request = 1;
  start(config);
  send(generate(1, 100));
  send({{"op", "close"}, {"request_id", 2}, {"session_id", "absent"}});
  bool failed = false, closed = false;
  for (int i = 0; i < 2; ++i) {
    const auto msg = receive();
    if (msg.at("request_id") == 1) {
      EXPECT_EQ(msg.at("code"), "slow_consumer");
      failed = true;
    } else {
      EXPECT_TRUE(msg.value("closed", false));
      closed = true;
    }
  }
  EXPECT_TRUE(failed && closed);
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, PermanentlyUnreadOutputExpiresAndCleansUp) {
  MultiplexedWorkerConfig config;
  config.write_timeout = 100ms;
  config.token_frames_per_request = 128;
  config.token_bytes_per_request = 1024 * 1024;
  tokenizer_.piece = std::string(128 * 1024, 'x');
  start(config);
  send(generate(1, 100));
  ASSERT_EQ(result_.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(result_.get(), 1);
  EXPECT_EQ(executor_.open_count(), 0);
}

TEST_F(ProtocolTest, PublishedControlReleasesCapacityBeforeWriterContinues) {
  auto published = gate();
  auto admitted = gate();
  hooks_.checkpoint = [published, admitted](Checkpoint point, uint64_t id) {
    if (point == Checkpoint::TerminalPublished && id == 1)
      published->pause();
    if (point == Checkpoint::Admitted && id == 2)
      admitted->signal();
  };
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  start(config, 4, 1);
  send({{"op", "cancel"}, {"request_id", 1}, {"target_request_id", 99}});
  const auto first = receive();
  ASSERT_EQ(first.at("request_id"), 1);
  ASSERT_TRUE(first.value("cancelled", false));
  ASSERT_TRUE(published->wait());
  send({{"op", "cancel"}, {"request_id", 2}, {"target_request_id", 99}});
  ASSERT_TRUE(admitted->wait());
  published->release();
  const auto second = receive();
  EXPECT_EQ(second.at("request_id"), 2);
  EXPECT_TRUE(second.value("cancelled", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, DeferredGenerationTerminalLetsUnrelatedAckProgress) {
  auto sink = gate();
  auto bound = gate();
  auto deferred = gate();
  hooks_.checkpoint = [sink, bound, deferred](Checkpoint point, uint64_t id) {
    if (id != 1)
      return;
    if (point == Checkpoint::TerminalEnqueued)
      sink->pause();
    if (point == Checkpoint::Bound)
      bound->signal();
    if (point == Checkpoint::TerminalDeferred)
      deferred->signal();
  };
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  start(config, 4, 1);
  send(generate(1, 1));
  ASSERT_TRUE(sink->wait());
  ASSERT_TRUE(bound->wait());
  ASSERT_TRUE(deferred->wait());
  const auto text = receive();
  ASSERT_EQ(text.at("request_id"), 1);
  ASSERT_EQ(text.at("token"), tokenizer_.piece);
  send({{"op", "cancel"}, {"request_id", 2}, {"target_request_id", 99}});
  const auto ack = receive();
  ASSERT_EQ(ack.at("request_id"), 2);
  ASSERT_TRUE(ack.value("cancelled", false));
  sink->release();
  const auto done = receive();
  ASSERT_EQ(done.at("request_id"), 1);
  ASSERT_TRUE(done.value("done", false));
  send(generate(3, 1));
  EXPECT_EQ(receive().at("token"), tokenizer_.piece);
  const auto next = receive();
  EXPECT_EQ(next.at("request_id"), 3);
  EXPECT_TRUE(next.value("done", false)) << next;
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, TerminalBeforeBindingCannotUsePlaceholderHandle) {
  auto binding = gate();
  auto terminal = gate();
  auto deferred = gate();
  hooks_.checkpoint = [binding, terminal, deferred](
                          Checkpoint point, uint64_t id) {
    if (id != 1)
      return;
    if (point == Checkpoint::BeforeBind)
      binding->pause();
    if (point == Checkpoint::TerminalEnqueued)
      terminal->signal();
    if (point == Checkpoint::TerminalDeferred)
      deferred->signal();
  };
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  start(config, 4, 1);
  send(generate(1, 1));
  ASSERT_TRUE(binding->wait());
  ASSERT_TRUE(terminal->wait());
  ASSERT_TRUE(deferred->wait());
  ASSERT_EQ(receive().at("token"), tokenizer_.piece);
  pollfd descriptor{output_[0], POLLIN, 0};
  EXPECT_EQ(poll(&descriptor, 1, 0), 0);
  binding->release();
  const auto done = receive();
  ASSERT_EQ(done.at("request_id"), 1);
  ASSERT_TRUE(done.value("done", false));
  // Reusing the identity is legal only once its terminal is published.
  send(generate(1, 1));
  EXPECT_EQ(receive().at("token"), tokenizer_.piece);
  EXPECT_TRUE(receive().value("done", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, TerminalNewlineRetriesKeepAdmissionReserved) {
  auto retry = gate();
  auto admitted = gate();
  auto attempts = std::make_shared<std::atomic<int>>(0);
  hooks_.terminal_write_error = [retry, attempts](uint64_t id) {
    if (id != 1)
      return 0;
    const auto attempt = ++*attempts;
    if (attempt == 1)
      return EAGAIN;
    if (attempt == 2)
      return EINTR;
    retry->pause();
    return 0;
  };
  hooks_.checkpoint = [admitted](Checkpoint point, uint64_t id) {
    if (point == Checkpoint::Admitted && id == 2)
      admitted->signal();
  };
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  start(config, 4, 1);
  send(generate(1, 1));
  ASSERT_TRUE(retry->wait());
  // The terminal payload is written, but its LF has not been published.
  // This must still take the transport's overload path, not reach runtime.
  send(generate(2, 1));
  ASSERT_TRUE(admitted->wait());
  retry->release();
  EXPECT_EQ(receive().at("token"), tokenizer_.piece);
  const auto done = receive();
  EXPECT_EQ(done.at("request_id"), 1);
  EXPECT_TRUE(done.value("done", false));
  const auto overload = receive();
  EXPECT_EQ(overload.at("request_id"), 2);
  EXPECT_EQ(overload.at("code"), "capacity_exhausted");
  EXPECT_EQ(overload.at("error"), "worker operation capacity exhausted");
  EXPECT_EQ(attempts->load(), 3);
  send(generate(3, 1));
  EXPECT_EQ(receive().at("token"), tokenizer_.piece);
  EXPECT_TRUE(receive().value("done", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, OperationCapacityReservesCancellationResponses) {
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  start(config);
  executor_.hold();
  send(generate(1, 100));
  ASSERT_TRUE(wait_execute());
  send(generate(2));
  EXPECT_EQ(receive().at("code"), "capacity_exhausted");
  send({{"op", "cancel"}, {"request_id", 3}, {"target_request_id", 1}});
  EXPECT_TRUE(receive().value("cancelled", false));
  executor_.release();
  Json done;
  do {
    done = receive();
  } while (done.contains("token"));
  EXPECT_TRUE(done.value("cancelled", false));
  EXPECT_EQ(finish(), 0);
}
} // namespace
