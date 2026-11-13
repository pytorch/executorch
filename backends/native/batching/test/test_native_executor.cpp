// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/batching/native_executor.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <executorch/backends/native/runtime/engine/ggml/GgmlEngine.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

namespace {

namespace batching = executorch::extension::llm::batching;
using batching::Token;

class NativeExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
    const char* path = std::getenv("ET_NATIVE_BATCHING_PTN");
    if (path == nullptr) {
      GTEST_SKIP() << "ET_NATIVE_BATCHING_PTN names no package; see "
                      "test/export_toy_llm.py";
    }
    ptn::register_ggml_cache_builders();
    auto created = ptn::NativeExecutor::create(
        {.model_path = path,
         .make_host = [] { return ptn::GgmlEngineHost::create(); },
         .cache_backend_id = ptn::kGgmlCacheBackendId,
         .max_sessions = 4,
         .max_session_tokens = 64});
    ASSERT_TRUE(created.ok());
    executor_ = std::move(created.get());
    ASSERT_TRUE(executor_->initialize());
  }

  batching::SessionId open() {
    const auto session = executor_->open_session();
    EXPECT_TRUE(session.has_value());
    executor_->set_sampling(*session, batching::SamplingParams{}, 0);
    return *session;
  }

  static batching::Input input(
      batching::SessionId session,
      std::vector<Token> tokens,
      batching::Position position) {
    batching::Input in;
    in.sid = session;
    in.size = tokens.size();
    in.offset = 0;
    in.payload = batching::TokenInputPtr(
        std::make_shared<std::vector<Token>>(std::move(tokens)));
    in.position = position;
    in.produce_output = true;
    return in;
  }

  // Runs one batch and returns each input's next token.
  std::vector<Token> step(std::vector<batching::Input> inputs) {
    batching::BatchOutput out;
    EXPECT_TRUE(executor_->execute(batching::BatchInput{inputs}, out));
    std::vector<Token> tokens;
    for (const auto& output : out.outputs) {
      EXPECT_TRUE(output.has_value());
      tokens.push_back(output ? output->tokens.at(0) : 0);
    }
    return tokens;
  }

  // Greedy tokens after `prompt`, decoding `steps` more, in a session alone.
  std::vector<Token> generate_alone(
      const std::vector<Token>& prompt,
      int steps) {
    const batching::SessionId session = open();
    std::vector<Token> tokens = step({input(session, prompt, 0)});
    for (int i = 1; i < steps; ++i) {
      tokens.push_back(step({input(
          session,
          {tokens.back()},
          static_cast<batching::Position>(prompt.size()) + i - 1)})[0]);
    }
    executor_->close_session(session);
    return tokens;
  }

  std::unique_ptr<ptn::NativeExecutor> executor_;
};

TEST(NativeExecutorCreateTest, RefusesAMissingEngine) {
  EXPECT_EQ(
      ptn::NativeExecutor::create({.model_path = "unused.ptn"}).error(),
      executorch::runtime::Error::InvalidArgument);
}

// Two sessions decoded in one batch give each the tokens it gets alone.
TEST_F(NativeExecutorTest, BatchedSessionsMatchSessionsAlone) {
  const std::vector<Token> a = {3, 14, 15, 92, 65};
  const std::vector<Token> b = {27, 18, 28};
  const std::vector<Token> want_a = generate_alone(a, 6);
  const std::vector<Token> want_b = generate_alone(b, 6);

  const batching::SessionId sa = open();
  const batching::SessionId sb = open();
  std::vector<Token> got_a, got_b;
  std::vector<Token> next = step({input(sa, a, 0), input(sb, b, 0)});
  for (int i = 0; i < 6; ++i) {
    got_a.push_back(next[0]);
    got_b.push_back(next[1]);
    if (i + 1 < 6) {
      next = step(
          {input(sa, {next[0]}, static_cast<batching::Position>(a.size()) + i),
           input(
               sb, {next[1]}, static_cast<batching::Position>(b.size()) + i)});
    }
  }
  EXPECT_EQ(got_a, want_a);
  EXPECT_EQ(got_b, want_b);
}

// A clone continues exactly as its source does, and a prompt wider than the
// method runs as several forwards.
TEST_F(NativeExecutorTest, ClonesContinueLikeTheirSource) {
  std::vector<Token> prompt;
  for (Token t = 1;
       t <= static_cast<Token>(executor_->preferred_batch_tokens()) + 5;
       ++t) {
    prompt.push_back(t % 100);
  }
  const auto position = static_cast<batching::Position>(prompt.size());
  const batching::SessionId source = open();
  const Token first = step({input(source, prompt, 0)})[0];
  const auto clone = executor_->clone(source, position);
  ASSERT_TRUE(clone.has_value());
  executor_->set_sampling(*clone, batching::SamplingParams{}, 0);
  const std::vector<Token> next = step(
      {input(source, {first}, position), input(*clone, {first}, position)});
  EXPECT_EQ(next[0], next[1]);
}

// One batch carries a session's decode and another's prompt, wider than the
// method, so the prompt's last token is selected in a later forward.
TEST_F(NativeExecutorTest, MixedPrefillAndDecodeMatchSessionsAlone) {
  const std::vector<Token> a = {3, 14, 15, 92, 65};
  std::vector<Token> b;
  for (Token t = 1;
       t <= static_cast<Token>(executor_->preferred_batch_tokens()) + 5;
       ++t) {
    b.push_back(t % 100);
  }
  const std::vector<Token> want_a = generate_alone(a, 4);
  const std::vector<Token> want_b = generate_alone(b, 3);

  const batching::SessionId sa = open();
  const batching::SessionId sb = open();
  const auto la = static_cast<batching::Position>(a.size());
  const auto lb = static_cast<batching::Position>(b.size());
  std::vector<Token> got_a = step({input(sa, a, 0)});
  std::vector<Token> next =
      step({input(sa, {got_a.back()}, la), input(sb, b, 0)});
  got_a.push_back(next[0]);
  std::vector<Token> got_b = {next[1]};
  for (batching::Position i = 1; i < 3; ++i) {
    next = step(
        {input(sa, {got_a.back()}, la + i),
         input(sb, {got_b.back()}, lb + i - 1)});
    got_a.push_back(next[0]);
    got_b.push_back(next[1]);
  }
  EXPECT_EQ(got_a, want_a);
  EXPECT_EQ(got_b, want_b);
}

// A batch the cache refuses, here a position past the session's length, fails
// without changing the session, which then continues as it would have.
TEST_F(NativeExecutorTest, RefusedBatchLeavesTheSession) {
  const std::vector<Token> prompt = {3, 14, 15, 92, 65};
  const std::vector<Token> want = generate_alone(prompt, 3);
  const batching::SessionId session = open();
  const auto length = static_cast<batching::Position>(prompt.size());
  std::vector<Token> got = step({input(session, prompt, 0)});
  batching::BatchOutput out;
  EXPECT_FALSE(executor_->execute(
      batching::BatchInput{{input(session, {got.back()}, length + 1)}}, out));
  for (batching::Position i = 0; i < 2; ++i) {
    got.push_back(step({input(session, {got.back()}, length + i)})[0]);
  }
  EXPECT_EQ(got, want);
}

} // namespace
