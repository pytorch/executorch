/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Drives CudaExecutor through the batching Runner on the toy decoder that
// export_toy_decoder.py writes to $ET_CUDA_BATCHING_TOY_DIR, and checks every
// generation against the eager greedy continuation it recorded.

#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/runner.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace cb = ::executorch::backends::cuda::batching;
namespace batching = ::executorch::extension::llm::batching;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;

namespace {

// Must match export_toy_decoder.py.
constexpr int kMaxStep = 8;
constexpr int kMaxCells = 256;
constexpr int kMaxContext = 64;
constexpr int kNewTokens = 6;
constexpr int kBFloat16 = 15; // ScalarType::BFloat16

struct Case {
  std::vector<batching::Token> prompt;
  std::vector<batching::Token> expected;
};

std::vector<batching::Token> parse_tokens(const std::string& text) {
  std::vector<batching::Token> tokens;
  std::istringstream in(text);
  batching::Token token;
  while (in >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

class CudaExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      GTEST_SKIP() << "CUDA device required";
    }
    const char* dir = std::getenv("ET_CUDA_BATCHING_TOY_DIR");
    if (dir == nullptr || *dir == '\0') {
      GTEST_SKIP() << "ET_CUDA_BATCHING_TOY_DIR is not set; run "
                      "export_toy_decoder.py first";
    }
    dir_ = dir;
    std::ifstream in(dir_ + "/expected.txt");
    ASSERT_TRUE(in.is_open()) << dir_ << "/expected.txt";
    std::string line;
    while (std::getline(in, line)) {
      const auto split = line.find(';');
      ASSERT_NE(split, std::string::npos) << line;
      cases_.push_back(
          {parse_tokens(line.substr(0, split)),
           parse_tokens(line.substr(split + 1))});
    }
    ASSERT_FALSE(cases_.empty());
  }

  std::unique_ptr<Module> module() const {
    return std::make_unique<Module>(
        dir_ + "/model.pte",
        std::vector<std::string>{dir_ + "/aoti_cuda_blob.ptd"},
        Module::LoadMode::File,
        /*event_tracer=*/nullptr,
        /*memory_allocator=*/nullptr,
        /*temp_allocator=*/nullptr,
        /*share_memory_arenas=*/false);
  }

  std::unique_ptr<cb::CudaExecutor> executor(
      cb::CudaExecutorOptions options = {},
      int max_sessions = 4) const {
    auto created = cb::CudaExecutor::create(
        module(),
        max_sessions,
        kMaxCells / max_sessions > kMaxContext ? kMaxContext
                                               : kMaxCells / max_sessions,
        kBFloat16,
        /*initial_capacity=*/16,
        options);
    EXPECT_EQ(created.error(), Error::Ok);
    return created.ok() ? std::move(created.get()) : nullptr;
  }

  // Runs every prompt concurrently through one runner; returns each
  // generation's tokens and its finish reason.
  struct Generation {
    std::vector<batching::Token> tokens;
    std::optional<batching::FinishReason> reason;
  };
  static std::vector<Generation> generate(
      batching::Runner& runner,
      const std::vector<std::vector<batching::Token>>& prompts) {
    std::vector<Generation> generations(prompts.size());
    std::vector<batching::Session> sessions;
    std::vector<batching::GenerationHandle> handles;
    std::mutex mutex;
    for (size_t i = 0; i < prompts.size(); ++i) {
      auto session = runner.open_session_async().get();
      EXPECT_TRUE(session.has_value()) << i;
      if (!session) {
        return generations;
      }
      batching::GenConfig config;
      config.max_new_tokens = kNewTokens;
      config.sampling.temperature = 0.0f;
      config.seed = 0;
      handles.push_back(session->generate_async(
          prompts[i], config, [&, i](const batching::GenerationUpdate& update) {
            std::lock_guard<std::mutex> guard(mutex);
            auto& tokens = generations[i].tokens;
            tokens.insert(
                tokens.end(), update.tokens.begin(), update.tokens.end());
          }));
      sessions.push_back(std::move(*session));
    }
    for (size_t i = 0; i < handles.size(); ++i) {
      handles[i].wait();
      generations[i].reason = handles[i].finish_reason();
    }
    return generations;
  }

  std::vector<std::vector<batching::Token>> prompts() const {
    std::vector<std::vector<batching::Token>> out;
    for (const Case& c : cases_) {
      out.push_back(c.prompt);
    }
    return out;
  }

  static std::unique_ptr<batching::Scheduler> scheduler() {
    // Half the forward for decodes, half for prefill chunks, so decodes and
    // prefills share forwards and a long prompt runs in several chunks.
    return batching::DecodeFirstScheduler::create(kMaxStep, 4, kMaxStep / 2);
  }

  std::string dir_;
  std::vector<Case> cases_;
};

} // namespace

TEST_F(CudaExecutorTest, ConcurrentGenerationsMatchEagerGreedy) {
  auto exec = executor();
  ASSERT_NE(exec, nullptr);
  EXPECT_EQ(exec->preferred_batch_tokens(), static_cast<size_t>(kMaxStep));
  batching::Runner runner(*exec, scheduler());
  const auto generations = generate(runner, prompts());
  const auto kv = exec->kv_metrics();
  runner.shutdown();

  for (size_t i = 0; i < cases_.size(); ++i) {
    EXPECT_EQ(generations[i].reason, batching::FinishReason::NewTokenLimit)
        << i;
    EXPECT_EQ(generations[i].tokens, cases_[i].expected) << "prompt " << i;
  }
  // The pools grew from 16 rows to hold every sequence's tokens at once.
  size_t tokens = 0;
  for (const Case& c : cases_) {
    tokens += c.prompt.size() + kNewTokens;
  }
  EXPECT_GE(kv.flat_capacity, kv.logical_length);
  EXPECT_GE(kv.growth_count, 1);
  EXPECT_GT(kv.allocated_bytes, 0);
  EXPECT_LE(kv.logical_length, static_cast<int64_t>(tokens));

  // Batched: fewer forwards than running the generations one after another.
  const auto engine = runner.metrics();
  EXPECT_EQ(engine.steps_failed, 0u);
  EXPECT_GT(engine.decode_sessions_total, engine.steps / 2);
}

TEST_F(CudaExecutorTest, EagerDecodeMatchesTheCapturedGraph) {
  cb::CudaExecutorOptions options;
  options.cuda_graph_for_decode = false;
  auto exec = executor(options);
  ASSERT_NE(exec, nullptr);
  batching::Runner runner(*exec, scheduler());
  const auto generations = generate(runner, prompts());
  runner.shutdown();
  for (size_t i = 0; i < cases_.size(); ++i) {
    EXPECT_EQ(generations[i].tokens, cases_[i].expected) << "prompt " << i;
  }
}

TEST_F(CudaExecutorTest, SamePromptTwiceInOneBatchGeneratesTheSame) {
  auto exec = executor();
  ASSERT_NE(exec, nullptr);
  batching::Runner runner(*exec, scheduler());
  const auto& c = cases_.back();
  const auto generations = generate(runner, {c.prompt, c.prompt});
  runner.shutdown();
  EXPECT_EQ(generations[0].tokens, c.expected);
  EXPECT_EQ(generations[1].tokens, c.expected);
}

TEST_F(CudaExecutorTest, SessionsReuseCellsAcrossRounds) {
  auto exec = executor();
  ASSERT_NE(exec, nullptr);
  batching::Runner runner(*exec, scheduler());
  // Sessions close between rounds, so the second round refills freed cells
  // and must not read what the first left there.
  for (int round = 0; round < 2; ++round) {
    const auto generations = generate(runner, prompts());
    for (size_t i = 0; i < cases_.size(); ++i) {
      EXPECT_EQ(generations[i].tokens, cases_[i].expected)
          << "round " << round << " prompt " << i;
    }
  }
  runner.shutdown();
}

TEST_F(CudaExecutorTest, RefusesLimitsThePoolCannotHold) {
  // 8 sessions of the full 64-token context need 512 cells; the program has
  // 256.
  EXPECT_EQ(
      cb::CudaExecutor::create(module(), 8, kMaxContext, kBFloat16, 16, {})
          .error(),
      Error::InvalidArgument);
  // Longer than the model's context.
  EXPECT_EQ(
      cb::CudaExecutor::create(module(), 1, kMaxContext + 1, kBFloat16, 16, {})
          .error(),
      Error::InvalidArgument);
}
