/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CI-only workload and measurements for the MG batching serving runtime:
// requests enter ServingRuntime, its Runner and DecodeFirstScheduler pack the
// steps, and the backend executor runs them.
//
// --mode=sweep: for each step width T, a workload that keeps every step at T
// tokens. Up to T = 64 that is T requests decoding together. Wider, it is
// --decode_cap decodes plus a backlog of prompts that fill the rest of each
// step (one per step by default, or --prefill_chunk tokens each), as a server
// under load would. The executor is
// wrapped so each step's width and time are recorded; a row reports the
// median over --iters steady steps after --warmup, both as executor time and
// as the interval between consecutive steps, which adds the runtime's own
// per-step work (scheduling, callbacks, detokenization).
//
// --mode=requests: --requests prompts of --prompt_tokens tokens submitted
// --stagger_ms apart, each generating --max_new_tokens; reports aggregate
// tok/s and each request's TTFT and TPOT.

#include "muse_glimmer_batching_bench_utils.h"
#include "muse_glimmer_batching_workload.h"

#include <executorch/examples/models/muse-glimmer/runtime/backend_factory.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/serving/serving_runtime.h>
#include <gflags/gflags.h>
#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

DECLARE_string(pte);
DECLARE_string(tokenizer);
DECLARE_string(data_path);
DECLARE_string(backend);
DECLARE_int32(max_sessions);
DECLARE_int32(max_session_tokens);
DECLARE_uint64(bos_id);

DEFINE_string(mode, "sweep", "sweep or requests");
DEFINE_string(
    tokens,
    "1,2,4,8,16,32,64,128,256,512,1024,2048,4096",
    "Step widths T to sweep");
DEFINE_int32(decode_cap, 32, "Decodes in a step wider than 64 tokens");
DEFINE_int32(
    prefill_chunk,
    0,
    "Prompt tokens of each backlog request; 0 fills a step's prefill with one");
DEFINE_int32(prefix, 128, "Prompt tokens of each decoding request");
DEFINE_int32(warmup, 6, "Steady steps discarded per T");
DEFINE_int32(iters, 20, "Steady steps measured per T");
DEFINE_int32(timeout_s, 600, "Give up on a T after this long");
DEFINE_int32(requests, 8, "requests mode: concurrent requests");
DEFINE_int32(prompt_tokens, 512, "requests mode: prompt tokens per request");
DEFINE_int32(
    max_new_tokens,
    128,
    "requests mode: tokens generated per request");
DEFINE_int32(stagger_ms, 0, "requests mode: delay between submissions");
DEFINE_int32(max_decode, 32, "requests mode: decodes admitted per step");
DEFINE_string(report_json, "", "Write the results here as JSON");

namespace llm = ::executorch::extension::llm;
namespace batching = llm::batching;
namespace serving = llm::serving;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point origin, Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(t - origin).count();
}

struct StepRecord {
  std::size_t tokens;
  std::size_t decodes;
  double start_ms;
  double end_ms;
};

// Forwards every call to the backend executor and records each step's width
// and time. Called only on the engine thread; the lock is for the reader.
class TimedExecutor final : public batching::Executor {
 public:
  explicit TimedExecutor(batching::Executor& inner)
      : inner_(inner), origin_(Clock::now()) {}

  bool initialize() override {
    return inner_.initialize();
  }
  std::size_t preferred_batch_tokens() const override {
    return inner_.preferred_batch_tokens();
  }
  std::optional<batching::SessionId> open_session() override {
    return inner_.open_session();
  }
  void close_session(batching::SessionId session) override {
    inner_.close_session(session);
  }
  std::optional<batching::SessionId> clone(
      batching::SessionId source,
      batching::Position upto) override {
    return inner_.clone(source, upto);
  }
  void set_sampling(
      batching::SessionId session,
      const batching::SamplingParams& params,
      std::optional<std::uint64_t> seed) override {
    inner_.set_sampling(session, params, seed);
  }
  bool accepts(const batching::PreparedInput& input) const override {
    return inner_.accepts(input);
  }
  bool execute(const batching::BatchInput& batch, batching::BatchOutput& out)
      override {
    const auto start = Clock::now();
    const bool ok = inner_.execute(batch, out);
    const auto end = Clock::now();
    std::size_t decodes = 0;
    for (const auto& input : batch.inputs) {
      decodes += input.size == 1 && input.produce_output &&
          std::holds_alternative<batching::TokenInputPtr>(input.payload);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    steps_.push_back(
        {batch.size(),
         decodes,
         ms_since(origin_, start),
         ms_since(origin_, end)});
    if (ok) {
      evidence_.observe(batch);
      if (batch.size() == watch_tokens_ && decodes == watch_decodes_) {
        ++watched_;
      }
    }
    return ok;
  }

  void watch(std::size_t tokens, std::size_t decodes) {
    std::lock_guard<std::mutex> lock(mutex_);
    watch_tokens_ = tokens;
    watch_decodes_ = decodes;
    watched_ = 0;
    steps_.clear();
  }
  std::size_t watched() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return watched_;
  }
  muse_glimmer_ci::BatchEvidence evidence() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return evidence_;
  }
  std::vector<StepRecord> take() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::exchange(steps_, {});
  }

 private:
  batching::Executor& inner_;
  const Clock::time_point origin_;
  mutable std::mutex mutex_;
  std::vector<StepRecord> steps_;
  muse_glimmer_ci::BatchEvidence evidence_;
  std::size_t watch_tokens_ = 0;
  std::size_t watch_decodes_ = 0;
  std::size_t watched_ = 0;
};

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

muse_glimmer_ci::RequestOptions request_options() {
  return {
      FLAGS_requests,
      FLAGS_prompt_tokens,
      FLAGS_max_new_tokens,
      FLAGS_stagger_ms,
      FLAGS_max_decode,
      FLAGS_timeout_s};
}

struct Bench {
  batching::Executor& executor;
  const tokenizers::Tokenizer& tokenizer;
  std::shared_ptr<const llm::MuseGlimmerPreparationSpec> spec;

  serving::ServingRuntimeConfig runtime_config(std::size_t sessions) const {
    muse_glimmer_ci::check_session_capacity(sessions, FLAGS_max_sessions);
    serving::ServingRuntimeConfig config;
    config.max_sessions = sessions;
    config.max_requests = sessions;
    config.max_pending_operations = sessions;
    config.max_context_length = spec->max_context_length;
    // No stop tokens: every request runs its full budget.
    config.default_stop_tokens = {};
    return config;
  }

  serving::ModelPreparer preparer() const {
    return [spec = spec](
               const serving::PromptPreparationContext& context,
               serving::PromptInput input) {
      return llm::prepare_muse_glimmer_input(context, std::move(input), spec);
    };
  }

  nlohmann::json sweep_width(int requested) const {
    const auto shape = muse_glimmer_ci::sweep_shape(
        requested, FLAGS_decode_cap, FLAGS_prefill_chunk, FLAGS_max_sessions);
    const int decodes = shape.decodes;
    const int chunk = shape.chunk;
    const int prefill = shape.prefill;
    const int t = shape.tokens();
    const int backlog = shape.backlog;
    // Decode-only widths leave room to prefill each prompt in one chunk.
    auto scheduler = prefill > 0
        ? batching::DecodeFirstScheduler::create(t, decodes, chunk)
        : batching::DecodeFirstScheduler::create(
              t + FLAGS_prefix, decodes, FLAGS_prefix);
    TimedExecutor timed(executor);
    muse_glimmer_ci::Inflight inflight;
    std::atomic<std::size_t> backlog_running{0};
    std::vector<serving::RequestHandle> handles;
    // Destroy the runtime before any state retained by its callbacks.
    auto rt = muse_glimmer_ci::make_runtime(
        timed,
        std::move(scheduler),
        tokenizer,
        runtime_config(decodes + backlog),
        preparer());
    const std::size_t steps_needed = FLAGS_warmup + FLAGS_iters + 1;
    // Outlasts the prefills that precede the steady steps.
    const auto decode_tokens =
        static_cast<std::int32_t>(2 * steps_needed + decodes + 200);
    if (static_cast<int64_t>(FLAGS_prefix) + decode_tokens >
        spec->max_context_length) {
      throw std::invalid_argument(
          "--prefix plus the decode budget exceeds context");
    }
    timed.watch(t, decodes);

    for (int i = 0; i < decodes; ++i) {
      if (auto handle = muse_glimmer_ci::submit_request(
              *rt, inflight, FLAGS_prefix, i, decode_tokens)) {
        handles.push_back(std::move(*handle));
      }
    }
    std::size_t salt = decodes;
    const auto deadline = Clock::now() + std::chrono::seconds(FLAGS_timeout_s);
    bool timed_out = false;
    while (timed.watched() < steps_needed) {
      if (Clock::now() > deadline) {
        timed_out = true;
        break;
      }
      while (backlog_running.load() < static_cast<std::size_t>(backlog)) {
        backlog_running.fetch_add(1);
        auto handle = muse_glimmer_ci::submit_request(
            *rt,
            inflight,
            chunk,
            salt++,
            1,
            [&backlog_running](const serving::TerminalEvent&) {
              backlog_running.fetch_sub(1);
            });
        if (!handle) {
          backlog_running.fetch_sub(1);
          break;
        }
      }
      std::unique_lock<std::mutex> lock(inflight.mutex);
      inflight.changed.wait_for(lock, std::chrono::milliseconds(2));
      if (inflight.failed > 0) {
        break;
      }
    }
    for (const auto& handle : handles) {
      handle.cancel();
    }
    rt->shutdown();
    const auto steps = timed.take();

    std::vector<double> exec_ms;
    std::vector<double> wall_ms;
    std::size_t seen = 0;
    for (std::size_t i = 1; i < steps.size(); ++i) {
      if (steps[i].tokens != static_cast<std::size_t>(t) ||
          steps[i].decodes != static_cast<std::size_t>(decodes)) {
        continue;
      }
      if (seen++ < static_cast<std::size_t>(FLAGS_warmup) ||
          exec_ms.size() == static_cast<std::size_t>(FLAGS_iters)) {
        continue;
      }
      exec_ms.push_back(steps[i].end_ms - steps[i].start_ms);
      wall_ms.push_back(steps[i].end_ms - steps[i - 1].end_ms);
    }
    const double exec = median(exec_ms);
    const double wall = median(wall_ms);
    std::lock_guard<std::mutex> lock(inflight.mutex);
    muse_glimmer_ci::check_measurement(
        timed_out,
        exec_ms.size(),
        FLAGS_iters,
        inflight.failed,
        !inflight.error.empty());
    return {
        {"T", t},
        {"requested_T", requested},
        {"decode", decodes},
        {"prefill", prefill},
        {"measured_steps", exec_ms.size()},
        {"median_exec_ms", exec},
        {"median_step_interval_ms", wall},
        {"exec_tokens_per_s", exec > 0 ? t / exec * 1000.0 : 0.0},
        {"tokens_per_s", wall > 0 ? t / wall * 1000.0 : 0.0},
        {"timed_out", timed_out},
        {"failed_requests", inflight.failed},
        {"error", inflight.error}};
  }

  nlohmann::json sweep() const {
    std::vector<int> widths;
    std::stringstream in(FLAGS_tokens);
    for (std::string item; std::getline(in, item, ',');) {
      widths.push_back(std::stoi(item));
    }
    nlohmann::json rows = nlohmann::json::array();
    std::printf(
        "%6s %6s %7s %10s %10s %11s %11s\n",
        "T",
        "decode",
        "prefill",
        "exec_ms",
        "step_ms",
        "exec_tok/s",
        "tok/s");
    double base = 0.0;
    for (const int t : widths) {
      auto row = sweep_width(t);
      const double rate = row["tokens_per_s"].get<double>();
      if (base == 0.0) {
        base = rate;
      }
      row["speedup_vs_first"] = base > 0 ? rate / base : 0.0;
      std::printf(
          "%6d %6d %7d %10.2f %10.2f %11.1f %11.1f  (%.1fx)%s\n",
          t,
          row["decode"].get<int>(),
          row["prefill"].get<int>(),
          row["median_exec_ms"].get<double>(),
          row["median_step_interval_ms"].get<double>(),
          row["exec_tokens_per_s"].get<double>(),
          rate,
          row["speedup_vs_first"].get<double>(),
          row["timed_out"].get<bool>() ? "  TIMED OUT" : "");
      std::fflush(stdout);
      rows.push_back(std::move(row));
    }
    return {{"mode", "sweep"}, {"rows", rows}};
  }

  nlohmann::json requests() const {
    const auto options = request_options();
    muse_glimmer_ci::validate_requests(
        options, FLAGS_max_sessions, spec->max_context_length);
    const int n = options.requests;
    struct Result {
      double submitted_ms = 0;
      double terminal_ms = 0;
      muse_glimmer_ci::RequestTiming timing;
    };
    // run_requests fences delivery and shutdown before these captures die.
    std::vector<Result> results(n);
    TimedExecutor timed(executor);
    Clock::time_point origin;
    muse_glimmer_ci::RequestCallbacks callbacks;
    callbacks.submitted = [&results, &origin](std::size_t i) {
      results[i].timing.submitted = Clock::now();
      if (i == 0) {
        origin = results[i].timing.submitted;
      }
      results[i].submitted_ms = ms_since(origin, results[i].timing.submitted);
    };
    callbacks.text = [&results](std::size_t i) {
      results[i].timing.first_text(Clock::now());
    };
    callbacks.terminal = [&results, &origin](
                             std::size_t i, const serving::TerminalEvent&) {
      results[i].terminal_ms = ms_since(origin, Clock::now());
    };
    const auto workload = muse_glimmer_ci::run_requests(
        options, timed, tokenizer, runtime_config(n), preparer(), callbacks);
    // Measure submission through delivered completion, excluding shutdown.
    double wall = 0;
    for (const auto& result : results) {
      wall = std::max(wall, result.terminal_ms);
    }

    nlohmann::json rows = nlohmann::json::array();
    std::size_t generated = 0;
    for (int i = 0; i < n; ++i) {
      const auto& stats = workload.terminals[i]->stats;
      generated += stats.completion_tokens;
      const double tpot = stats.completion_tokens > 1
          ? stats.decode_ms / (stats.completion_tokens - 1)
          : 0.0;
      rows.push_back(
          {{"submitted_ms", results[i].submitted_ms},
           {"prompt_tokens", stats.prompt_tokens},
           {"completion_tokens", stats.completion_tokens},
           {"ttft_ms",
            results[i].timing.ttft_ms
                ? nlohmann::json(*results[i].timing.ttft_ms)
                : nlohmann::json(nullptr)},
           {"prefill_ms", stats.prefill_ms},
           {"tpot_ms", tpot},
           {"total_ms", stats.total_ms},
           {"failed", false}});
    }
    const auto evidence = timed.evidence();
    const double rate = wall > 0 ? generated / wall * 1000.0 : 0.0;
    std::printf(
        "%d requests x %d prompt tokens: %zu generated in %.1f ms = %.1f tok/s\n",
        n,
        FLAGS_prompt_tokens,
        generated,
        wall,
        rate);
    return {
        {"mode", "requests"},
        {"wall_ms", wall},
        {"generated_tokens", generated},
        {"tokens_per_s", rate},
        {"failed_requests", 0},
        {"max_batch_sessions", evidence.max_sessions},
        {"multi_session_steps", evidence.multi_session_steps},
        {"mixed_prefill_decode_steps", evidence.mixed_steps},
        {"requests", rows}};
  }
};

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    if ((FLAGS_mode != "requests" && FLAGS_mode != "sweep") ||
        FLAGS_max_sessions <= 0 || FLAGS_prefix <= 0 || FLAGS_warmup < 0 ||
        FLAGS_iters <= 0 ||
        static_cast<int64_t>(FLAGS_warmup) + FLAGS_iters >
            (std::numeric_limits<int32_t>::max() - FLAGS_max_sessions - 202LL) /
                2) {
      throw std::invalid_argument("invalid benchmark options");
    }
    if (FLAGS_mode == "requests") {
      muse_glimmer_ci::validate_requests(request_options(), FLAGS_max_sessions);
    } else {
      if (FLAGS_timeout_s <= 0) {
        throw std::invalid_argument("invalid sweep timeout");
      }
      std::stringstream widths(FLAGS_tokens);
      bool any = false;
      for (std::string item; std::getline(widths, item, ',');) {
        std::size_t parsed;
        const int width = std::stoi(item, &parsed);
        if (parsed != item.size() ||
            static_cast<int64_t>(width) + FLAGS_prefix >
                std::numeric_limits<int>::max()) {
          throw std::invalid_argument("invalid --tokens entry");
        }
        muse_glimmer_ci::sweep_shape(
            width, FLAGS_decode_cap, FLAGS_prefill_chunk, FLAGS_max_sessions);
        any = true;
      }
      if (!any) {
        throw std::invalid_argument("--tokens must not be empty");
      }
    }
    tokenizers::HFTokenizer tokenizer;
    if (tokenizer.load(FLAGS_tokenizer) != tokenizers::Error::Ok) {
      throw std::runtime_error("could not load the tokenizer");
    }
    llm::MuseGlimmerBackendConfig config;
    config.backend = FLAGS_backend;
    config.model_path = FLAGS_pte;
    config.data_path = FLAGS_data_path;
    config.max_sessions = FLAGS_max_sessions;
    config.max_session_tokens = FLAGS_max_session_tokens;
    config.bos_id = FLAGS_bos_id;
    auto backend = llm::create_muse_glimmer_backend(config);
    if (!backend.ok()) {
      throw std::runtime_error(
          "could not create the executor: error " +
          std::to_string(static_cast<int>(backend.error())));
    }
    Bench bench{*backend->executor, tokenizer, backend->preparation};
    auto report = FLAGS_mode == "requests" ? bench.requests() : bench.sweep();
    report["backend"] = FLAGS_backend;
    report["model"] = FLAGS_pte;
    report["max_sessions"] = FLAGS_max_sessions;
    report["max_session_tokens"] = backend->preparation->max_context_length;
    if (!FLAGS_report_json.empty()) {
      std::ofstream output(FLAGS_report_json);
      output << report.dump(1) << '\n';
      output.close();
      if (!output) {
        throw std::runtime_error("could not write --report_json");
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "batching bench failed: " << error.what() << '\n';
    return 1;
  }
}
