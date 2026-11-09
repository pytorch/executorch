/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Drives CudaExecutor on the toy decoders that export_toy_decoder.py writes
// under $ET_CUDA_BATCHING_TOY_DIR, and checks every generation against the
// eager greedy continuation it recorded. Runs once per artifact: forward_{1,2,
// 4,8} + forward_others, and the sparser forward_{1,4} + forward_others with a
// five-row selector minimum, where steps pad across a gap. mg_widths carries
// Muse Glimmer's forward_{1,...,64} + forward_others and drives steps of every
// width up to 65.

#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/prefix_cache.h>
#include <executorch/extension/llm/batching/runner.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace cb = ::executorch::backends::cuda::batching;
namespace batching = ::executorch::extension::llm::batching;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;

namespace {

// Must match export_toy_decoder.py.
constexpr int kMaxStep = 32;
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

class CudaExecutorTest : public ::testing::TestWithParam<const char*> {
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
    dir_ = std::string(dir) + "/" + GetParam();
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
    return module_at(dir_);
  }

  static std::unique_ptr<Module> module_at(const std::string& dir) {
    return std::make_unique<Module>(
        dir + "/model.pte",
        std::vector<std::string>{dir + "/aoti_cuda_blob.ptd"},
        Module::LoadMode::File,
        /*event_tracer=*/nullptr,
        /*memory_allocator=*/nullptr,
        /*temp_allocator=*/nullptr,
        /*share_memory_arenas=*/false);
  }

  std::unique_ptr<cb::CudaExecutor> executor(
      cb::CudaExecutorOptions options = {},
      int max_sessions = 4) const {
    return executor_for(module(), options, max_sessions);
  }

  static std::unique_ptr<cb::CudaExecutor> executor_for(
      std::unique_ptr<Module> program,
      cb::CudaExecutorOptions options = {},
      int max_sessions = 4) {
    auto created = cb::CudaExecutor::create(
        std::move(program),
        max_sessions,
        // One cell is the padding scratch row.
        (kMaxCells - 1) / max_sessions > kMaxContext
            ? kMaxContext
            : (kMaxCells - 1) / max_sessions,
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

  // Generates `new_tokens` greedily from `delta` on `session`, which may hold
  // a reused prefix. A capture, if given, wraps the callback.
  static std::vector<batching::Token> continue_on(
      const batching::Session& session,
      std::vector<batching::Token> delta,
      int new_tokens,
      batching::PrefixCache::PromptCapture* capture = nullptr) {
    std::vector<batching::Token> tokens;
    std::mutex mutex;
    batching::GenConfig config;
    config.max_new_tokens = new_tokens;
    config.seed = 0;
    batching::GenerationCallback on_update =
        [&](const batching::GenerationUpdate& update) {
          std::lock_guard<std::mutex> guard(mutex);
          tokens.insert(
              tokens.end(), update.tokens.begin(), update.tokens.end());
        };
    if (capture != nullptr) {
      on_update = capture->wrap(std::move(on_update));
    }
    auto handle =
        session.generate_async(std::move(delta), config, std::move(on_update));
    handle.wait();
    EXPECT_EQ(handle.finish_reason(), batching::FinishReason::NewTokenLimit);
    return tokens;
  }

  static std::unique_ptr<batching::Scheduler> scheduler() {
    // Up to four decodes per forward and prefill chunks of eight, so decodes
    // and prefills share forwards and a long prompt runs in several chunks.
    return batching::DecodeFirstScheduler::create(kMaxStep, 4, 8);
  }

  // One step of a hand-written schedule: (prompt index, tokens to feed).
  using Step = std::vector<std::pair<int, int>>;

  // Drives the executor directly through `schedule`, then decodes every
  // unfinished generation together until each has kNewTokens. Session i
  // samples with `sampling` and seed i. Returns each generation.
  std::vector<std::vector<batching::Token>> run_schedule(
      cb::CudaExecutor& exec,
      const std::vector<Step>& schedule,
      const batching::SamplingParams& sampling = {}) const {
    const size_t n = cases_.size();
    std::vector<batching::SessionId> sids(n);
    std::vector<std::vector<batching::Token>> history(n);
    std::vector<size_t> fed(n, 0);
    std::vector<std::vector<batching::Token>> generated(n);
    for (size_t i = 0; i < n; ++i) {
      auto sid = exec.open_session();
      EXPECT_TRUE(sid.has_value()) << i;
      sids[i] = sid.value_or(-1);
      exec.set_sampling(sids[i], sampling, i);
      history[i] = cases_[i].prompt;
    }
    auto run = [&](const Step& step) {
      batching::BatchInput batch;
      for (const auto& [i, count] : step) {
        const size_t take = static_cast<size_t>(count);
        EXPECT_LE(fed[i] + take, history[i].size()) << i;
        batch.inputs.push_back(batching::Input{
            sids[i],
            /*produce_output=*/fed[i] + take == history[i].size(),
            fed[i],
            take,
            std::make_shared<const std::vector<batching::Token>>(history[i]),
            /*position=*/0});
      }
      batching::BatchOutput out;
      EXPECT_TRUE(exec.execute(batch, out));
      for (size_t j = 0; j < step.size(); ++j) {
        const int i = step[j].first;
        fed[i] += static_cast<size_t>(step[j].second);
        if (batch.inputs[j].produce_output) {
          ASSERT_TRUE(out.outputs[j].has_value()) << i;
          const batching::Token token = out.outputs[j]->tokens.at(0);
          generated[i].push_back(token);
          history[i].push_back(token);
        }
      }
    };
    for (const Step& step : schedule) {
      run(step);
    }
    for (;;) {
      Step step;
      for (size_t i = 0; i < n; ++i) {
        if (generated[i].size() < static_cast<size_t>(kNewTokens)) {
          step.push_back({static_cast<int>(i), 1});
        }
      }
      if (step.empty()) {
        break;
      }
      run(step);
    }
    for (size_t i = 0; i < n; ++i) {
      exec.close_session(sids[i]);
    }
    return generated;
  }

  std::string dir_;
  std::vector<Case> cases_;
};

} // namespace

TEST_P(CudaExecutorTest, ConcurrentGenerationsMatchEagerGreedy) {
  auto exec = executor();
  ASSERT_NE(exec, nullptr);
  EXPECT_EQ(exec->preferred_batch_tokens(), static_cast<size_t>(kMaxStep));
  // Every exported method, narrowest first, the dynamic one last.
  const auto methods = exec->method_calls();
  ASSERT_GE(methods.size(), 3u);
  EXPECT_EQ(methods.front().name, "forward_1");
  EXPECT_EQ(methods.back().name, "forward_others");
  EXPECT_EQ(
      exec->samples_on_device(), std::string(GetParam()) == "device_sampling");
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

TEST_P(CudaExecutorTest, EagerDecodeMatchesTheCapturedGraph) {
  cb::CudaExecutorOptions options;
  options.cuda_graph_for_static_methods = false;
  auto exec = executor(options);
  ASSERT_NE(exec, nullptr);
  batching::Runner runner(*exec, scheduler());
  const auto generations = generate(runner, prompts());
  runner.shutdown();
  for (size_t i = 0; i < cases_.size(); ++i) {
    EXPECT_EQ(generations[i].tokens, cases_[i].expected) << "prompt " << i;
  }
}

TEST_P(CudaExecutorTest, SamePromptTwiceInOneBatchGeneratesTheSame) {
  auto exec = executor();
  ASSERT_NE(exec, nullptr);
  batching::Runner runner(*exec, scheduler());
  const auto& c = cases_.back();
  const auto generations = generate(runner, {c.prompt, c.prompt});
  runner.shutdown();
  EXPECT_EQ(generations[0].tokens, c.expected);
  EXPECT_EQ(generations[1].tokens, c.expected);
}

TEST_P(CudaExecutorTest, SessionsReuseCellsAcrossRounds) {
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

TEST_P(
    CudaExecutorTest,
    InterleavedMethodsMatchEagerGreedyWithAndWithoutGraphs) {
  // Prompts: [3], [5 9 1], [7 2 2 8 4 6 1], and 20 tokens. Each step's width,
  // in order: 20, 1, 4, 2, 4 (a partial prefill), 7, 3 -- then all four
  // decode together, narrowing as generations finish. Every static method and
  // forward_others runs, out of order, across pool growth from 16 rows, and
  // steps of 2, 3 and 7 tokens pad. Graph and eager runs must both reproduce
  // each prompt's unbatched greedy continuation, so padding touched no one
  // else's history.
  ASSERT_EQ(cases_.size(), 4u);
  const std::vector<Step> schedule = {
      {{3, 20}},
      {{3, 1}},
      {{1, 3}, {3, 1}},
      {{0, 1}, {1, 1}},
      {{2, 3}, {3, 1}},
      {{2, 4}, {0, 1}, {1, 1}, {3, 1}},
      {{0, 1}, {1, 1}, {3, 1}},
  };
  for (const bool graphs : {true, false}) {
    cb::CudaExecutorOptions options;
    options.cuda_graph_for_static_methods = graphs;
    auto exec = executor(options);
    ASSERT_NE(exec, nullptr);
    ASSERT_TRUE(exec->initialize());
    const auto generated = run_schedule(*exec, schedule);
    for (size_t i = 0; i < cases_.size(); ++i) {
      EXPECT_EQ(generated[i], cases_[i].expected)
          << "prompt " << i << (graphs ? " with graphs" : " eager");
    }
    for (const auto& method : exec->method_calls()) {
      EXPECT_GT(method.calls, 0u) << method.name;
    }
    EXPECT_GE(exec->kv_metrics().growth_count, 1);
  }
}

// Only the device_sampling program has the device samplers, so this suite is
// instantiated for it alone: a skipped instance would read, in CI, as the toy
// programs missing.
class CudaExecutorDeviceSamplingTest : public CudaExecutorTest {};

TEST_P(
    CudaExecutorDeviceSamplingTest,
    DeviceSamplerDrawsWhatTheHostSamplerDraws) {
  // dense/ has the same forwards and samples on the host. With one seed per
  // session, both must draw every token alike, under every policy and mixed
  // greedy/stochastic rows.
  const std::string host_dir = dir_.substr(0, dir_.rfind('/')) + "/dense";
  const std::vector<Step> schedule = {{{3, 20}}, {{0, 1}, {1, 3}, {2, 7}}};
  std::vector<batching::SamplingParams> policies(3);
  policies[0].temperature = 1.0f;
  policies[1].temperature = 1.3f;
  policies[1].top_k = 3;
  policies[2].temperature = 0.8f;
  policies[2].top_p = 0.9f;
  for (const auto& policy : policies) {
    auto device = executor();
    auto host = executor_for(module_at(host_dir));
    ASSERT_NE(device, nullptr);
    ASSERT_NE(host, nullptr);
    ASSERT_TRUE(device->initialize());
    ASSERT_TRUE(host->initialize());
    ASSERT_TRUE(device->samples_on_device());
    ASSERT_FALSE(host->samples_on_device());
    const auto on_device = run_schedule(*device, schedule, policy);
    const auto on_host = run_schedule(*host, schedule, policy);
    EXPECT_EQ(on_device, on_host)
        << "temperature " << policy.temperature << " top_k " << policy.top_k
        << " top_p " << policy.top_p;
    // Sampling, not argmax: some generation leaves the greedy path.
    bool diverged = false;
    for (size_t i = 0; i < cases_.size(); ++i) {
      diverged |= on_device[i] != cases_[i].expected;
    }
    EXPECT_TRUE(diverged);
  }
}

TEST_P(CudaExecutorTest, PrefixCacheResumesFromACapturedPrompt) {
  // Each prompt's cold generation leaves a snapshot cloned at its prompt
  // boundary. That snapshot then serves the same prompt again and a prompt
  // extending it; both resume on the eager greedy path, so the clone carried
  // the prompt's exact KV and later writes did not disturb it.
  auto exec = executor();
  ASSERT_NE(exec, nullptr);
  batching::Runner runner(*exec, scheduler());
  batching::PrefixCache prefixes(2);
  constexpr int kKnown = 2;
  for (const Case& c : cases_) {
    if (c.prompt.size() < 2) {
      continue;
    }
    {
      auto cold = runner.open_session_async().get();
      ASSERT_TRUE(cold.has_value());
      auto capture = prefixes.capture_prompt(*cold, c.prompt);
      EXPECT_EQ(continue_on(*cold, c.prompt, kNewTokens, &capture), c.expected);
      EXPECT_TRUE(capture.collect());
    }

    auto same = prefixes.lookup(c.prompt);
    ASSERT_TRUE(same.has_value());
    EXPECT_EQ(same->matched_tokens, c.prompt.size() - 1);
    EXPECT_EQ(
        same->session.position(),
        static_cast<batching::Position>(c.prompt.size() - 1));
    EXPECT_EQ(
        continue_on(same->session, {c.prompt.back()}, kNewTokens), c.expected);

    std::vector<batching::Token> extended = c.prompt;
    extended.insert(
        extended.end(), c.expected.begin(), c.expected.begin() + kKnown);
    auto longer = prefixes.lookup(extended);
    ASSERT_TRUE(longer.has_value());
    EXPECT_EQ(longer->matched_tokens, c.prompt.size());
    const std::vector<batching::Token> suffix(
        extended.begin() + longer->matched_tokens, extended.end());
    EXPECT_EQ(
        continue_on(longer->session, suffix, kNewTokens - kKnown),
        std::vector<batching::Token>(
            c.expected.begin() + kKnown, c.expected.end()));
  }
  prefixes.clear();
  runner.shutdown();
  EXPECT_EQ(runner.metrics().steps_failed, 0u);
}

TEST_P(CudaExecutorTest, RefusesLimitsThePoolCannotHold) {
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

TEST(CudaExecutorConfigTest, InvalidConfigIsRejectedBeforeLoading) {
  cb::CudaExecutorConfig config;
  EXPECT_EQ(cb::CudaExecutor::create(config).error(), Error::InvalidArgument);

  // A missing program must not mask invalid session limits, even without CUDA.
  config.model_path = "/nonexistent/cuda_executor_test/model.pte";
  config.load_mode = Module::LoadMode::File;
  config.max_sessions = 0;
  EXPECT_EQ(cb::CudaExecutor::create(config).error(), Error::InvalidArgument);

  config.max_sessions = 1;
  config.max_session_tokens = -1;
  EXPECT_EQ(cb::CudaExecutor::create(config).error(), Error::InvalidArgument);
}

TEST_P(CudaExecutorTest, ConfigAutoBudgetRespectsContextAndPaddingCell) {
  const std::vector<std::pair<int, int>> limits = {
      {1, kMaxContext}, {4, 63}, {8, 31}};
  for (const auto& [sessions, expected_budget] : limits) {
    SCOPED_TRACE(sessions);
    cb::CudaExecutorConfig config;
    config.model_path = dir_ + "/model.pte";
    config.data_files = {dir_ + "/aoti_cuda_blob.ptd"};
    config.load_mode = Module::LoadMode::File;
    config.max_sessions = sessions;
    config.initial_capacity = 16;
    config.options.cuda_graph_for_static_methods = false;
    auto created = cb::CudaExecutor::create(config);
    ASSERT_EQ(created.error(), Error::Ok);
    auto exec = std::move(created.get());
    ASSERT_NE(exec, nullptr);
    EXPECT_EQ(exec->max_session_tokens(), expected_budget);
    EXPECT_LT(exec->max_session_tokens() * sessions, kMaxCells);
    // The toy export publishes a 32-entry vocabulary.
    EXPECT_EQ(exec->vocab_size(), 32);
  }
}

TEST_P(CudaExecutorTest, ConfiguredBudgetsMatchEagerGreedy) {
  ASSERT_EQ(cases_.size(), 4u);
  for (const int token_budget : {0, 32}) {
    SCOPED_TRACE(token_budget);
    cb::CudaExecutorConfig config;
    config.model_path = dir_ + "/model.pte";
    config.data_files = {dir_ + "/aoti_cuda_blob.ptd"};
    config.load_mode = Module::LoadMode::File;
    config.max_sessions = 4;
    config.max_session_tokens = token_budget;
    config.kv_dtype = kBFloat16;
    config.initial_capacity = 16;
    config.options.cuda_graph_for_static_methods = false;
    auto created = cb::CudaExecutor::create(config);
    ASSERT_EQ(created.error(), Error::Ok);
    auto exec = std::move(created.get());
    ASSERT_NE(exec, nullptr);
    EXPECT_EQ(exec->max_session_tokens(), token_budget == 0 ? 63 : 32);
    batching::Runner runner(*exec, scheduler());
    const auto generations = generate(runner, prompts());
    runner.shutdown();
    ASSERT_EQ(generations.size(), cases_.size());
    for (size_t i = 0; i < cases_.size(); ++i) {
      EXPECT_EQ(generations[i].reason, batching::FinishReason::NewTokenLimit)
          << "prompt " << i;
      EXPECT_EQ(generations[i].tokens, cases_[i].expected) << "prompt " << i;
    }
    EXPECT_EQ(runner.metrics().steps_failed, 0u);
  }
}

TEST_P(CudaExecutorTest, ConfigRefusesOvercommitAndLegacyZeroBudget) {
  cb::CudaExecutorConfig config;
  config.model_path = dir_ + "/model.pte";
  config.data_files = {dir_ + "/aoti_cuda_blob.ptd"};
  config.load_mode = Module::LoadMode::File;
  config.max_sessions = 8;
  config.max_session_tokens = kMaxContext;
  config.initial_capacity = 16;
  config.options.cuda_graph_for_static_methods = false;
  EXPECT_EQ(cb::CudaExecutor::create(config).error(), Error::InvalidArgument);

  // Reserving the padding row leaves no token per session at 256 sessions.
  config.max_sessions = kMaxCells;
  config.max_session_tokens = 0;
  EXPECT_EQ(cb::CudaExecutor::create(config).error(), Error::InvalidArgument);
  EXPECT_EQ(
      cb::CudaExecutor::create(module(), 4, 0, kBFloat16, 16, {}).error(),
      Error::InvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    Toy,
    CudaExecutorTest,
    ::testing::Values("dense", "sparse", "device_sampling"));

INSTANTIATE_TEST_SUITE_P(
    Toy,
    CudaExecutorDeviceSamplingTest,
    ::testing::Values("device_sampling"));

namespace {

// Must match export_toy_decoder.py's mg_widths artifact.
constexpr int kWidthsMaxStep = 96;
constexpr int kWidthsMaxCells = 1024;

int next_pow2(int n) {
  int p = 1;
  while (p < n) {
    p *= 2;
  }
  return p;
}

// Muse Glimmer's method set: forward_{1,2,...,64} and forward_others from 64.
class CudaExecutorWidthsTest : public CudaExecutorTest {};

} // namespace

// Steps of every width 1..65 (and 96, 97), each a random mix of sequences and
// chunk lengths: a static method's step pads up to the next power of two and
// drops the padding's rows, 65..96 runs forward_others unpadded, and 97 slices.
// Every output a step asks for is the eager greedy successor, so padding
// neither reaches a real row nor disturbs a later step through the cache.
TEST_P(CudaExecutorWidthsTest, EveryStepWidthUpTo65MatchesEagerGreedy) {
  const int sessions = static_cast<int>(cases_.size());
  ASSERT_LT(sessions * kMaxContext, kWidthsMaxCells);
  for (const bool graphs : {true, false}) {
    SCOPED_TRACE(graphs ? "cuda graphs" : "eager");
    cb::CudaExecutorOptions options;
    options.cuda_graph_for_static_methods = graphs;
    auto created = cb::CudaExecutor::create(
        module(),
        sessions,
        kMaxContext,
        kBFloat16,
        /*initial_capacity=*/16,
        options);
    ASSERT_EQ(created.error(), Error::Ok);
    auto& exec = *created.get();
    ASSERT_TRUE(exec.initialize());
    ASSERT_EQ(
        exec.preferred_batch_tokens(), static_cast<size_t>(kWidthsMaxStep));

    // Teacher forcing: each sequence is its prompt and eager continuation, so
    // a step may end any chunk at a position whose successor is known.
    std::vector<std::shared_ptr<const std::vector<batching::Token>>> seqs;
    for (const Case& c : cases_) {
      auto full = c.prompt;
      full.insert(full.end(), c.expected.begin(), c.expected.end());
      ASSERT_LE(full.size(), static_cast<size_t>(kMaxContext));
      seqs.push_back(std::make_shared<const std::vector<batching::Token>>(
          std::move(full)));
    }
    std::vector<batching::SessionId> sids(sessions, -1);
    std::vector<size_t> fed(sessions, 0);
    auto reset = [&] {
      for (int i = 0; i < sessions; ++i) {
        if (sids[i] >= 0) {
          exec.close_session(sids[i]);
        }
        auto sid = exec.open_session();
        ASSERT_TRUE(sid.has_value());
        sids[i] = *sid;
        exec.set_sampling(sids[i], {}, 0);
        fed[i] = 0;
      }
    };
    // Each sequence's last token has no known successor, so it is never fed.
    auto remaining = [&](int i) { return seqs[i]->size() - 1 - fed[i]; };
    reset();

    std::vector<int> widths;
    for (int round = 0; round < 6; ++round) {
      for (int n = 1; n <= 65; ++n) {
        widths.push_back(n);
      }
      widths.push_back(kWidthsMaxStep);
      widths.push_back(kWidthsMaxStep + 1);
    }
    std::mt19937 rng(1234);
    std::shuffle(widths.begin(), widths.end(), rng);

    size_t checked = 0;
    for (const int n : widths) {
      SCOPED_TRACE("step of " + std::to_string(n));
      size_t total = 0;
      for (int i = 0; i < sessions; ++i) {
        total += remaining(i);
      }
      if (total < static_cast<size_t>(n)) {
        reset();
      }
      // A random subset and split of the sequences that sums to n, each
      // sequence at most once.
      std::vector<int> order(sessions);
      std::iota(order.begin(), order.end(), 0);
      std::shuffle(order.begin(), order.end(), rng);
      size_t after = 0;
      for (const int i : order) {
        after += remaining(i);
      }
      batching::BatchInput batch;
      std::vector<int> owner;
      size_t left = static_cast<size_t>(n);
      for (const int i : order) {
        after -= remaining(i);
        if (left == 0 || remaining(i) == 0) {
          continue;
        }
        const size_t most = std::min(remaining(i), left);
        // Enough that the sequences after this one can cover the rest.
        const size_t least = left > after ? left - after : 0;
        if (most < least || (least == 0 && rng() % 3 == 0)) {
          continue;
        }
        const size_t take = std::uniform_int_distribution<size_t>(
            std::max<size_t>(least, 1), most)(rng);
        const size_t end = fed[i] + take;
        // Ask for the chunk's last row whenever its successor is known, but
        // not always, so some steps want none.
        const bool known = end >= cases_[i].prompt.size();
        batch.inputs.push_back(batching::Input{
            sids[i], known && rng() % 4 != 0, fed[i], take, seqs[i], 0});
        owner.push_back(i);
        left -= take;
      }
      ASSERT_EQ(left, 0u);
      ASSERT_EQ(batch.size(), static_cast<size_t>(n));

      const auto before = exec.method_calls();
      batching::BatchOutput out;
      ASSERT_TRUE(exec.execute(batch, out));
      const auto calls = exec.method_calls();

      // The method(s) the width runs through.
      std::map<std::string, std::uint64_t> ran;
      for (size_t m = 0; m < calls.size(); ++m) {
        if (calls[m].calls != before[m].calls) {
          ran[calls[m].name] = calls[m].calls - before[m].calls;
        }
      }
      std::map<std::string, std::uint64_t> want;
      if (n <= 64) {
        want["forward_" + std::to_string(next_pow2(n))] = 1;
      } else if (n <= kWidthsMaxStep) {
        want["forward_others"] = 1;
      } else {
        want["forward_others"] = 1;
        want["forward_" + std::to_string(next_pow2(n - kWidthsMaxStep))] = 1;
      }
      EXPECT_EQ(ran, want);

      ASSERT_EQ(out.outputs.size(), batch.inputs.size());
      for (size_t j = 0; j < batch.inputs.size(); ++j) {
        const int i = owner[j];
        const auto& input = batch.inputs[j];
        const size_t end = input.offset + input.size;
        if (input.produce_output) {
          ASSERT_TRUE(out.outputs[j].has_value()) << "sequence " << i;
          EXPECT_EQ(out.outputs[j]->sid, sids[i]);
          ASSERT_EQ(out.outputs[j]->tokens.size(), 1u);
          EXPECT_EQ(out.outputs[j]->tokens[0], (*seqs[i])[end])
              << "sequence " << i << " after " << end << " tokens";
          ++checked;
        } else {
          EXPECT_FALSE(out.outputs[j].has_value()) << "sequence " << i;
        }
        fed[i] = end;
      }
    }
    EXPECT_GT(checked, widths.size());
    for (const batching::SessionId sid : sids) {
      exec.close_session(sid);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Toy,
    CudaExecutorWidthsTest,
    ::testing::Values("mg_widths"));
