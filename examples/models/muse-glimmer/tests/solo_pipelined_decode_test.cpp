/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cuda_runtime.h>
#include <executorch/examples/models/muse-glimmer/runtime/engine/muse_glimmer_engine.h>
#include <executorch/extension/llm/runner/constants.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace llm = executorch::extension::llm;
namespace rt = executorch::runtime;

namespace {

using Tokens = std::vector<uint64_t>;
using Emitted = std::vector<int64_t>;
using Clock = std::chrono::steady_clock;

constexpr int64_t kTerminal = -1;
constexpr int64_t kErrorBase = -1000;
// The engine's default EOS id, outside the tiny model's 256-token vocabulary.
constexpr uint64_t kNoEos = 200001;

const Tokens kPrompt = {11, 23, 5, 87, 140, 3, 64, 201, 9, 33, 250};
const Tokens kOtherPrompt = {200, 17, 99, 4, 160, 72, 38, 121, 6};
// At least the export's minimum prefill chunk, so it runs on the prefill
// method.
const Tokens kContinuation = {45, 130, 7, 222, 61, 18};
// Shorter than the minimum prefill chunk, so the session feeds it one token at
// a time through the decode method, synchronously.
const Tokens kShortContinuation = {77, 12, 190};

llm::SamplingConfig greedy() {
  llm::SamplingConfig sampling;
  sampling.temperature = 0.0f;
  return sampling;
}

int64_t error_marker(rt::Error error) {
  return kErrorBase - static_cast<int64_t>(error);
}

// The first step, at least three in, whose token was not emitted before it.
size_t first_new_token(const Emitted& emitted) {
  for (size_t step = 3; step < emitted.size(); ++step) {
    const auto end = emitted.begin() + step;
    if (std::find(emitted.begin(), end, emitted[step]) == end) {
      return step;
    }
  }
  return emitted.size();
}

// A copy of the artifact's tokenizer that cannot decode `id`.
std::string write_tokenizer_without(const std::string& dir, uint64_t id) {
  std::ifstream in(dir + "/tokenizer.json");
  auto tokenizer = nlohmann::json::parse(in);
  auto& vocab = tokenizer["model"]["vocab"];
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    if (it.value().get<uint64_t>() == id) {
      vocab.erase(it);
      break;
    }
  }
  const std::string path = ::testing::TempDir() + "tokenizer_without_" +
      std::to_string(id) + ".json";
  std::ofstream(path) << tokenizer.dump();
  return path;
}

// Drives one engine and records every token it emits, followed by kTerminal
// where a decode step ends the stream, or an error marker where it fails.
class Driver {
 public:
  Driver(
      const std::string& dir,
      const std::string& tokenizer_path,
      bool pipelined,
      uint64_t eos_id,
      bool cuda_graph = true) {
    llm::MuseGlimmerConfig config;
    config.model_path = dir + "/model.pte";
    config.data_path = dir + "/aoti_cuda_blob.ptd";
    config.tokenizer_path = tokenizer_path;
    config.eos_id = static_cast<int64_t>(eos_id);
    config.enable_cuda_graph = cuda_graph;
    config.pipelined_decode = pipelined;
    auto engine = llm::MuseGlimmerEngine::create(config);
    EXPECT_EQ(engine.error(), rt::Error::Ok);
    if (engine.ok()) {
      engine_ = std::move(engine.get());
      open();
    }
  }

  bool ready() const {
    return session_ != nullptr;
  }
  bool pipelined() const {
    return engine_->pipelined_decode();
  }
  int64_t position() const {
    return session_->position();
  }
  int64_t max_context() const {
    return engine_->metadata().at(llm::kMaxContextLen);
  }

  // Replaces the session; the engine hosts one at a time.
  void open() {
    session_.reset();
    auto session = engine_->create_session();
    ASSERT_EQ(session.error(), rt::Error::Ok);
    session_ = std::move(session.get());
  }
  void prefill(const Tokens& tokens) {
    const auto sampling = greedy();
    ASSERT_EQ(session_->prefill_tokens(tokens, &sampling), rt::Error::Ok);
  }
  void decode(int steps) {
    for (int i = 0; i < steps; ++i) {
      auto step = session_->decode_one(greedy());
      ASSERT_EQ(step.error(), rt::Error::Ok);
      if (step->is_eos || !step->is_terminal) {
        emitted.push_back(static_cast<int64_t>(step->token_id));
      }
      if (step->is_terminal) {
        emitted.push_back(kTerminal);
        return;
      }
    }
  }
  // Like decode, but records a failed step and stops there.
  void decode_recording_errors(int steps) {
    for (int i = 0; i < steps; ++i) {
      auto step = session_->decode_one(greedy());
      if (!step.ok()) {
        emitted.push_back(error_marker(step.error()));
        return;
      }
      if (step->is_eos || !step->is_terminal) {
        emitted.push_back(static_cast<int64_t>(step->token_id));
      }
      if (step->is_terminal) {
        emitted.push_back(kTerminal);
        return;
      }
    }
  }
  void reset() {
    ASSERT_EQ(session_->reset(), rt::Error::Ok);
  }
  void stop() {
    session_->stop();
  }

  Emitted emitted;

 private:
  std::unique_ptr<llm::MuseGlimmerEngine> engine_;
  std::unique_ptr<llm::LLMSession> session_;
};

// Blocks the stream it is enqueued on for a while.
void hold_stream(void*) {
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

// The CUDA backend's process-wide async_replay_for_method option.
std::string async_replay_methods() {
  rt::BackendOptions<1> options;
  EXPECT_EQ(options.set_option("async_replay_for_method", ""), rt::Error::Ok);
  EXPECT_EQ(rt::get_option("CudaBackend", options.view()), rt::Error::Ok);
  const char* methods = nullptr;
  EXPECT_EQ(
      options.get_option("async_replay_for_method", methods), rt::Error::Ok);
  return methods != nullptr ? methods : "";
}

// A long-lived thread that runs one task at a time, while the caller waits.
class Worker {
 public:
  Worker() : thread_([this] { loop(); }) {}
  ~Worker() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      done_ = true;
    }
    ready_.notify_all();
    thread_.join();
  }
  void run(std::function<void()> task) {
    std::unique_lock<std::mutex> lock(mutex_);
    task_ = std::move(task);
    ready_.notify_all();
    ready_.wait(lock, [this] { return !task_; });
  }

 private:
  void loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      ready_.wait(lock, [this] { return done_ || task_; });
      if (!task_) {
        return;
      }
      task_();
      task_ = nullptr;
      ready_.notify_all();
    }
  }

  std::mutex mutex_;
  std::condition_variable ready_;
  std::function<void()> task_;
  bool done_ = false;
  std::thread thread_;
};

// A host function that blocks the stream it is enqueued on until open(), so
// work enqueued after it on that stream cannot start before then. A watchdog
// opens it after 10 s, so a call that waits for it forever fails the test
// instead of hanging it.
class StreamGate {
 public:
  ~StreamGate() {
    open();
    (void)cudaStreamSynchronize(cudaStreamPerThread);
  }
  // Blocks the calling thread's per-thread stream.
  void hold_current_stream() {
    ASSERT_EQ(
        cudaLaunchHostFunc(cudaStreamPerThread, &StreamGate::wait, this),
        cudaSuccess);
  }
  void open() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (open_) {
        return;
      }
      open_ = true;
      opened_at_ = Clock::now();
    }
    opened_.notify_all();
  }
  Clock::time_point opened_at() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return opened_at_;
  }
  bool timed_out() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return timed_out_;
  }

 private:
  static void wait(void* self) {
    auto* gate = static_cast<StreamGate*>(self);
    std::unique_lock<std::mutex> lock(gate->mutex_);
    if (!gate->opened_.wait_for(
            lock, std::chrono::seconds(10), [gate] { return gate->open_; })) {
      gate->timed_out_ = true;
      gate->open_ = true;
      gate->opened_at_ = Clock::now();
    }
  }

  mutable std::mutex mutex_;
  std::condition_variable opened_;
  bool open_ = false;
  bool timed_out_ = false;
  Clock::time_point opened_at_;
};

// A flag one thread sets and another waits for.
class Signal {
 public:
  void set() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      set_ = true;
    }
    changed_.notify_all();
  }
  void wait_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait_for(lock, timeout, [this] { return set_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool set_ = false;
};

// Calls that share the KV cache and the staged decode token with a pending
// decode step.
enum class KvCall {
  kPrefill, // forward_from_embeddings
  kShortPrefill, // fed one token at a time through decode_from_embedding
  kReset,
  kNewSession,
};

const char* kv_call_name(KvCall call) {
  switch (call) {
    case KvCall::kPrefill:
      return "prefill";
    case KvCall::kShortPrefill:
      return "short prefill";
    case KvCall::kReset:
      return "reset";
    case KvCall::kNewSession:
      return "new session";
  }
  return "";
}

// Decodes six tokens, then makes `call` (on `worker`, if given) while the
// decode step that the last decode_one enqueued is pending. On the pipelined
// engine, a gate holds that step back on the GPU until `call` has started and
// returned, or has had a second to return: decode_one must return without
// waiting for the step, and `call` must return only after it. Ends with a
// prefill, if `call` left none, and 12 decode steps.
void decode_then_call_with_pending_step(
    Driver& d,
    KvCall call,
    Worker* worker) {
  SCOPED_TRACE(
      std::string(kv_call_name(call)) +
      (worker != nullptr ? " from another thread"
                         : " from the decoding thread"));
  d.reset();
  d.prefill(kPrompt);
  d.decode(5);
  std::optional<StreamGate> gate;
  Signal call_started;
  Signal call_returned;
  std::thread opener;
  if (d.pipelined()) {
    gate.emplace();
    gate->hold_current_stream();
    // A decode_one stuck on the held step never starts `call`, so the gate
    // opens after the first timeout and the step is seen to finish first.
    opener = std::thread([&] {
      call_started.wait_for(std::chrono::seconds(5));
      call_returned.wait_for(std::chrono::seconds(1));
      gate->open();
    });
  }
  d.decode(1);
  const Clock::time_point decoded = Clock::now();
  Clock::time_point started;
  Clock::time_point returned;
  auto make_call = [&] {
    started = Clock::now();
    call_started.set();
    switch (call) {
      case KvCall::kPrefill:
        d.prefill(kContinuation);
        break;
      case KvCall::kShortPrefill:
        d.prefill(kShortContinuation);
        break;
      case KvCall::kReset:
        d.reset();
        break;
      case KvCall::kNewSession:
        d.open();
        break;
    }
    returned = Clock::now();
    call_returned.set();
  };
  if (worker != nullptr) {
    worker->run(make_call);
  } else {
    make_call();
  }
  if (gate) {
    opener.join();
    EXPECT_FALSE(gate->timed_out());
    EXPECT_LT(decoded, gate->opened_at());
    EXPECT_LT(started, gate->opened_at());
    EXPECT_GT(returned, gate->opened_at());
  }
  if (call == KvCall::kReset || call == KvCall::kNewSession) {
    d.prefill(kOtherPrompt);
  }
  d.decode(12);
}

using Scenario = std::function<void(Driver&)>;

class SoloPipelinedDecode : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* dir = std::getenv("ET_MUSE_SOLO_TINY_DIR");
    if (!dir || !*dir) {
      GTEST_SKIP() << "Set ET_MUSE_SOLO_TINY_DIR to the output of "
                      "tests/export_cuda_solo_tiny.py";
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      GTEST_SKIP() << "CUDA device required";
    }
    rt::runtime_init();
    dir_ = dir;
    tokenizer_path_ = dir_ + "/tokenizer.json";
  }

  Emitted run(bool pipelined, const Scenario& scenario, uint64_t eos = kNoEos) {
    Driver driver(dir_, tokenizer_path_, pipelined, eos);
    EXPECT_TRUE(driver.ready());
    if (!driver.ready()) {
      return {};
    }
    EXPECT_EQ(driver.pipelined(), pipelined);
    scenario(driver);
    positions_.push_back(driver.position());
    return driver.emitted;
  }

  // Runs the scenario decoding token by token, then pipelined: both must emit
  // the same tokens and end at the same position.
  Emitted expect_same_as_token_by_token(
      const Scenario& scenario,
      uint64_t eos = kNoEos) {
    positions_.clear();
    const Emitted expected = run(false, scenario, eos);
    EXPECT_EQ(run(true, scenario, eos), expected);
    EXPECT_EQ(positions_.size(), 2u);
    if (positions_.size() == 2) {
      EXPECT_EQ(positions_[1], positions_[0]);
    }
    return expected;
  }

  std::string dir_;
  std::string tokenizer_path_;
  std::vector<int64_t> positions_;
};

TEST_F(SoloPipelinedDecode, DecodeMatchesTokenByToken) {
  const Emitted tokens = expect_same_as_token_by_token([](Driver& d) {
    d.prefill(kPrompt);
    d.decode(24);
  });
  EXPECT_EQ(tokens.size(), 24u);
}

// Consecutive decode steps run on two long-lived threads in turn, so each step
// is enqueued on a different per-thread stream than the step before it. On the
// first thread's stream, a host callback holds each step back while the second
// thread enqueues the next one.
TEST_F(SoloPipelinedDecode, DecodeFromAlternatingThreadsMatchesTokenByToken) {
  const Emitted tokens = expect_same_as_token_by_token([](Driver& d) {
    std::array<Worker, 2> workers;
    d.prefill(kPrompt);
    for (int i = 0; i < 24; ++i) {
      const bool hold = i % 2 == 0;
      workers[i % 2].run([&d, hold] {
        if (hold) {
          ASSERT_EQ(
              cudaLaunchHostFunc(cudaStreamPerThread, hold_stream, nullptr),
              cudaSuccess);
        }
        d.decode(1);
      });
    }
    d.reset();
  });
  EXPECT_EQ(tokens.size(), 24u);
}

// The CUDA backend's options are process-wide, so an engine without CUDA graphs
// must not inherit the async replay that a pipelined engine set before it.
TEST_F(SoloPipelinedDecode, EngineWithoutGraphsDoesNotInheritAsyncReplay) {
  {
    Driver pipelined(dir_, tokenizer_path_, /*pipelined=*/true, kNoEos);
    ASSERT_TRUE(pipelined.ready());
    EXPECT_TRUE(pipelined.pipelined());
    EXPECT_EQ(async_replay_methods(), "decode_from_embedding");
  }
  Driver plain(
      dir_, tokenizer_path_, /*pipelined=*/true, kNoEos, /*cuda_graph=*/false);
  ASSERT_TRUE(plain.ready());
  EXPECT_FALSE(plain.pipelined());
  EXPECT_EQ(async_replay_methods(), "");
}

TEST_F(SoloPipelinedDecode, PrefillAfterResetWithStepInFlightMatchesFresh) {
  const Emitted fresh = run(true, [](Driver& d) {
    d.prefill(kOtherPrompt);
    d.decode(12);
  });
  const Emitted tokens = expect_same_as_token_by_token([](Driver& d) {
    d.prefill(kPrompt);
    d.decode(6);
    d.reset();
    d.prefill(kOtherPrompt);
    d.decode(12);
  });
  ASSERT_EQ(tokens.size(), 18u);
  EXPECT_EQ(Emitted(tokens.begin() + 6, tokens.end()), fresh);
}

TEST_F(SoloPipelinedDecode, ContinuationPrefillsAfterStepInFlight) {
  const Emitted tokens = expect_same_as_token_by_token([](Driver& d) {
    d.prefill(kPrompt);
    d.decode(6);
    d.prefill(kContinuation);
    d.decode(12);
    d.prefill(kShortContinuation);
    d.decode(12);
  });
  EXPECT_EQ(tokens.size(), 30u);
}

// Each call that shares the KV cache and the staged decode token with a pending
// decode step, made from the decoding thread and from another thread while the
// step is held back on the GPU: a prefill through forward_from_embeddings, a
// short prefill fed through decode_from_embedding, reset, and replacing the
// session. Each must wait for the step and leave the session as the
// token-by-token path leaves it.
TEST_F(SoloPipelinedDecode, KvCacheCallsWithPendingStepMatchTokenByToken) {
  const Emitted tokens = expect_same_as_token_by_token([](Driver& d) {
    Worker other_thread;
    for (Worker* worker : {static_cast<Worker*>(nullptr), &other_thread}) {
      for (KvCall call :
           {KvCall::kPrefill,
            KvCall::kShortPrefill,
            KvCall::kReset,
            KvCall::kNewSession}) {
        decode_then_call_with_pending_step(d, call, worker);
      }
    }
  });
  EXPECT_EQ(tokens.size(), 8u * (6 + 12));
}

TEST_F(SoloPipelinedDecode, PrefillAfterStopWithStepInFlight) {
  const Emitted tokens = expect_same_as_token_by_token([](Driver& d) {
    d.prefill(kPrompt);
    d.decode(6);
    d.stop();
    d.decode(1);
    d.prefill(kContinuation);
    d.decode(12);
  });
  ASSERT_EQ(tokens.size(), 19u);
  EXPECT_EQ(tokens[6], kTerminal);
}

TEST_F(SoloPipelinedDecode, PrefillAfterEosDiscardsTheEnqueuedStep) {
  const Emitted free_run = run(false, [](Driver& d) {
    d.prefill(kPrompt);
    d.decode(24);
  });
  const size_t eos_step = first_new_token(free_run);
  ASSERT_LT(eos_step, free_run.size()) << "the tiny model only repeats tokens";
  const auto eos = static_cast<uint64_t>(free_run[eos_step]);
  const Emitted tokens = expect_same_as_token_by_token(
      [](Driver& d) {
        d.prefill(kPrompt);
        d.decode(24);
        d.prefill(kContinuation);
        d.decode(12);
      },
      eos);
  ASSERT_GT(tokens.size(), eos_step + 1);
  EXPECT_EQ(tokens[eos_step], free_run[eos_step]);
  EXPECT_EQ(tokens[eos_step + 1], kTerminal);
}

TEST_F(SoloPipelinedDecode, NewSessionAfterDroppingOneWithStepInFlight) {
  const Emitted fresh = run(true, [](Driver& d) {
    d.prefill(kOtherPrompt);
    d.decode(12);
  });
  const Emitted tokens = expect_same_as_token_by_token([](Driver& d) {
    d.prefill(kPrompt);
    d.decode(6);
    d.open();
    d.prefill(kOtherPrompt);
    d.decode(12);
  });
  ASSERT_EQ(tokens.size(), 18u);
  EXPECT_EQ(Emitted(tokens.begin() + 6, tokens.end()), fresh);
}

// A step that fails after the pipelined path has enqueued the next one must
// leave the session where the token-by-token path leaves it: the same error on
// a retry, and a clean reset.
TEST_F(SoloPipelinedDecode, FullContextFailsAsTokenByToken) {
  const Emitted tokens = expect_same_as_token_by_token([](Driver& d) {
    constexpr int kRoom = 6;
    Tokens prompt(static_cast<size_t>(d.max_context() - kRoom));
    for (size_t i = 0; i < prompt.size(); ++i) {
      prompt[i] = 2 + (17 * i + 5) % 251;
    }
    d.prefill(prompt);
    d.decode_recording_errors(kRoom + 2);
    d.decode_recording_errors(1);
    d.reset();
    d.prefill(kOtherPrompt);
    d.decode(12);
  });
  ASSERT_EQ(tokens.size(), 20u);
  EXPECT_EQ(tokens[6], error_marker(rt::Error::InvalidArgument));
  EXPECT_EQ(tokens[7], error_marker(rt::Error::InvalidArgument));
}

TEST_F(SoloPipelinedDecode, DetokenizeFailureFailsAsTokenByToken) {
  const Emitted free_run = run(false, [](Driver& d) {
    d.prefill(kPrompt);
    d.decode(24);
  });
  const size_t step = first_new_token(free_run);
  ASSERT_LT(step, free_run.size()) << "the tiny model only repeats tokens";
  tokenizer_path_ =
      write_tokenizer_without(dir_, static_cast<uint64_t>(free_run[step]));
  const Emitted tokens = expect_same_as_token_by_token([](Driver& d) {
    d.prefill(kPrompt);
    d.decode_recording_errors(24);
    d.decode_recording_errors(1);
    d.reset();
    d.prefill(kOtherPrompt);
    d.decode_recording_errors(12);
  });
  ASSERT_GT(tokens.size(), step + 1);
  EXPECT_EQ(
      Emitted(tokens.begin(), tokens.begin() + step),
      Emitted(free_run.begin(), free_run.begin() + step));
  EXPECT_EQ(tokens[step], error_marker(rt::Error::InvalidArgument));
  EXPECT_EQ(tokens[step + 1], error_marker(rt::Error::InvalidArgument));
}

} // namespace
