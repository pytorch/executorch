/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// What a batched run measured, in two tiers.
//
// GenerationMetrics covers one generation and is published on its handle, so a
// caller reads it beside finish_reason(). EngineMetrics covers the engine and
// is owned by the runner.
//
// There is deliberately no session tier: Session::position() already reports a
// session's context, and a session's generations are recovered by grouping on
// GenerationMetrics::sid. Keeping generations separate is the point -- a second
// generation on a warm session has a different profile from a cold one, and an
// average over the two hides exactly that.
//
// Free of ExecuTorch runtime types, like the rest of these headers: no ET_LOG,
// no exceptions, and nothing here allocates outside the format_report calls.

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include <executorch/extension/llm/batching/types.h>
#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

// Monotonic: these are durations, and a wall-clock adjustment mid-run would
// otherwise produce negative ones.
using MetricsClock = std::chrono::steady_clock;
using MetricsTime = MetricsClock::time_point;

ET_EXPERIMENTAL inline std::int64_t us_between(
    MetricsTime from,
    MetricsTime to) {
  return std::chrono::duration_cast<std::chrono::microseconds>(to - from)
      .count();
}

// A default-constructed time_point means the event never happened, which is
// distinct from it happening at time zero: the clock's epoch is the process's,
// so no real event lands there.
ET_EXPERIMENTAL inline bool stamped(MetricsTime t) {
  return t.time_since_epoch().count() != 0;
}

// One generation's timeline and counts. The finish reason is not duplicated
// here; GenerationHandle::finish_reason() already carries it, and two copies
// would be free to disagree.
struct ET_EXPERIMENTAL GenerationMetrics {
  SessionId sid = 0;

  // generate_async() was called. Taken on the caller's thread, before the
  // request is queued.
  MetricsTime t_submit{};
  // The first batch this generation appeared in. Everything between here and
  // t_submit is time the scheduler did not pick it.
  MetricsTime t_first_step{};
  MetricsTime t_first_token{};
  MetricsTime t_end{};

  std::int64_t n_prompt_tokens = 0;
  std::int64_t n_generated_tokens = 0;
  std::int32_t n_prefill_steps = 0;
  std::int32_t n_decode_steps = 0;

  // Caller-visible inter-token latency, excluding the gap to the first token,
  // which is TTFT. The first token in a later callback gets the elapsed gap;
  // further tokens in that callback get zero-time samples. Summary rather than
  // samples: exact mean at four scalars, where a per-token vector would cost
  // 8 KB on a long generation.
  std::int64_t itl_count = 0;
  std::int64_t itl_sum_us = 0;
  std::int64_t itl_min_us = std::numeric_limits<std::int64_t>::max();
  std::int64_t itl_max_us = 0;

  // Time to first token: what a caller waits before anything appears.
  std::int64_t ttft_us() const {
    return stamped(t_submit) && stamped(t_first_token)
        ? us_between(t_submit, t_first_token)
        : 0;
  }

  // The queueing share of ttft_us(). Large means the scheduler was busy, not
  // that prefill was slow.
  std::int64_t queue_wait_us() const {
    return stamped(t_submit) && stamped(t_first_step)
        ? us_between(t_submit, t_first_step)
        : 0;
  }

  // The compute share of ttft_us(). Includes other sessions' work in the steps
  // this one's prefill was spread across, so read it with n_prefill_steps.
  std::int64_t prefill_span_us() const {
    return stamped(t_first_step) && stamped(t_first_token)
        ? us_between(t_first_step, t_first_token)
        : 0;
  }

  // Generation proper, after the first token.
  std::int64_t decode_span_us() const {
    return stamped(t_first_token) && stamped(t_end)
        ? us_between(t_first_token, t_end)
        : 0;
  }

  std::int64_t e2e_us() const {
    return stamped(t_submit) && stamped(t_end) ? us_between(t_submit, t_end)
                                               : 0;
  }

  double itl_mean_us() const {
    return itl_count > 0 ? static_cast<double>(itl_sum_us) / itl_count : 0.0;
  }

  // Zero rather than the sentinel when no gap was ever sampled, as happens to
  // any generation that produced a single token. Mirrors min_ttft_us().
  std::int64_t min_itl_us() const {
    return itl_count > 0 ? itl_min_us : 0;
  }

  // What this one caller saw, which is not the engine's aggregate rate.
  double decode_tokens_per_sec() const {
    if (itl_count == 0) {
      return 0.0;
    }
    return itl_sum_us > 0 ? 1e6 * static_cast<double>(itl_count) / itl_sum_us
                          : std::numeric_limits<double>::infinity();
  }

  // Prompt tokens over the wall time this generation's prefill took. Like the
  // decode rate above it is what this caller experienced, so it includes any
  // work sharing those steps -- a prompt that waits behind another session's
  // chunks reports a lower rate, which is what that caller actually got.
  double prefill_tokens_per_sec() const {
    const std::int64_t span = prefill_span_us();
    return span > 0 ? 1e6 * static_cast<double>(n_prompt_tokens) / span : 0.0;
  }
};

// The engine's own view, accumulated on the engine thread and read once it has
// stopped.
struct ET_EXPERIMENTAL EngineMetrics {
  std::uint64_t steps = 0;
  std::uint64_t steps_failed = 0;

  // Summed over steps, so dividing by `steps` gives the mean. Sessions, not
  // tokens: a prefill chunk is one session and many tokens.
  std::uint64_t decode_sessions_total = 0;
  std::uint64_t prefill_sessions_total = 0;
  // Generations that could have decoded in this step: alive and past their
  // first token. Prefilling generations are excluded because they are not
  // waiting on a decode slot, they are doing their own work. Against
  // decode_sessions_total this says how much of the eligible work the scheduler
  // ran, without naming any scheduler's limits.
  std::uint64_t ready_total = 0;

  std::int64_t step_latency_sum_us = 0;
  std::int64_t step_latency_max_us = 0;

  // Every token the engine processed. Task::is_decode classifies each one
  // exactly, so these are complete.
  //
  // There is deliberately no rate over every step that held them. A step
  // mixing both kinds runs them in one forward pass over one weight read, so
  // no share of its latency belongs to either, and dividing by the time of
  // those steps would make packing prefill alongside decode -- which raises
  // total throughput -- look like a decode regression. The rate that is safe
  // to publish is decode_only_tokens_per_sec(), taken over steps that held no
  // prefill at all, where the attribution is not in question.
  std::uint64_t decode_tokens_total = 0;
  std::uint64_t prefill_tokens_total = 0;
  // The part of decode_tokens_total that ran on decode-only steps, so it can
  // be divided by their time. The comparable figure to a fixed-batch decode
  // benchmark, which measures exactly this shape of step.
  std::uint64_t decode_only_tokens = 0;

  // Steps holding at least one task of each kind. A step can hold both, so
  // these overlap by mixed_steps().
  std::uint64_t steps_with_decode = 0;
  std::uint64_t steps_with_prefill = 0;

  // step_latency_sum_us split by what the step held. A partition, unlike the
  // two counts above: every step is exactly one of these three, so they sum to
  // step_latency_sum_us. Comparing the first two prices the scheduler's choice
  // to pack prefill alongside decode, which no single blended mean can show.
  std::int64_t decode_only_latency_sum_us = 0;
  std::int64_t mixed_latency_sum_us = 0;
  std::int64_t prefill_only_latency_sum_us = 0;

  // Committed session length summed across the batch, once per step: what
  // attention had to carry. Generic -- executor.h defines a session's length,
  // saying nothing about how the state is stored -- and the largest reason one
  // decode step costs more than another.
  std::int64_t context_sum = 0;
  std::int64_t context_max = 0; // longest single session seen in any step

  // Step latency charged once to every decode session in the step. A latency
  // may be counted for several sessions because each of them really did wait
  // it -- unlike time as a cost, waiting is not divided up. Includes mixed
  // steps, where a decode stuck behind a prefill chunk waited the whole thing.
  std::int64_t decode_session_time_sum_us = 0;

  // Opens the executor refused for being at capacity, and the most generations
  // seen installed at once. The peak is sampled once per executed step, so a
  // generation that began and ended between two steps is not counted.
  std::uint64_t sessions_refused = 0;
  std::uint64_t peak_concurrent_generations = 0;

  std::uint64_t generations_started = 0;
  std::uint64_t generations_completed = 0;
  std::uint64_t finished_stop_token = 0;
  std::uint64_t finished_token_limit = 0;
  std::uint64_t finished_cancelled = 0;
  std::uint64_t finished_failed = 0;

  // Over generations that reached a first token, which is not every
  // completion: one cancelled or failed during prefill has no TTFT to report.
  std::uint64_t ttft_count = 0;
  std::int64_t ttft_sum_us = 0;
  std::int64_t ttft_min_us = std::numeric_limits<std::int64_t>::max();
  std::int64_t ttft_max_us = 0;

  std::int64_t total_prompt_tokens = 0;
  std::int64_t total_generated_tokens = 0;

  MetricsTime t_first_step{};
  MetricsTime t_last_step{};

  // Executor::initialize(), timed here because nothing else can: it runs on the
  // engine thread after the constructor has already returned, so a caller has
  // no two points to measure between. Outside wall_us(), which starts at the
  // first step -- folding it in would hide one-time setup inside the run.
  std::int64_t init_us = 0;

  double wall_us() const {
    return stamped(t_first_step) && stamped(t_last_step)
        ? static_cast<double>(us_between(t_first_step, t_last_step))
        : 0.0;
  }

  // Mean decode sessions per step, and the mean that were eligible. Raw
  // counts: normalising against a scheduler's decode limit would tie these to
  // one scheduler, and against a workload smaller than that limit it would
  // report idle capacity that no prompt existed to fill.
  //
  // Both divide by every step, so the pair is comparable. For the width of an
  // actual decode forward, which is what a batched matmul sees, use
  // mean_decode_step_sessions().
  double mean_admitted_decode_sessions() const {
    return steps > 0 ? static_cast<double>(decode_sessions_total) / steps : 0.0;
  }

  double mean_ready_decode_sessions() const {
    return steps > 0 ? static_cast<double>(ready_total) / steps : 0.0;
  }

  // Decode sessions per step that actually held a decode. Each contributes one
  // token, so this is also the decode token width of the forward.
  double mean_decode_step_sessions() const {
    return steps_with_decode > 0
        ? static_cast<double>(decode_sessions_total) / steps_with_decode
        : 0.0;
  }

  // Every session in the step, both kinds. How full the forward pass was,
  // which is the batching question; the decode figures above are the
  // scheduling one.
  double mean_step_sessions() const {
    return steps > 0
        ? static_cast<double>(decode_sessions_total + prefill_sessions_total) /
            steps
        : 0.0;
  }

  // Below 1 the scheduler is holding eligible work back rather than running
  // out of it. Slightly under 1 is normal: a generation is briefly eligible
  // but unqueued between its output being handled and its continuation being
  // submitted.
  double admitted_ratio() const {
    return ready_total > 0
        ? static_cast<double>(decode_sessions_total) / ready_total
        : 0.0;
  }

  // What the engine processed. Distinct from the generation tier's totals,
  // which count what callers were given: a generation's first token comes out
  // of a prefill step, and prompt tokens are processed but never emitted.
  std::uint64_t model_input_tokens() const {
    return decode_tokens_total + prefill_tokens_total;
  }

  std::int64_t total_tokens() const {
    return total_prompt_tokens + total_generated_tokens;
  }

  double mean_step_tokens() const {
    return steps > 0 ? static_cast<double>(model_input_tokens()) / steps : 0.0;
  }

  // Wall time the engine spent outside execute(): waiting for work, draining
  // commands, running callbacks.
  double idle_fraction() const {
    const double wall = wall_us();
    return wall > 0.0 ? 1.0 - static_cast<double>(step_latency_sum_us) / wall
                      : 0.0;
  }

  // What callers were given, per second. The comparable figure when swapping
  // executors: a speculative one feeds one token per decode step and returns
  // several, so processed tokens would stay flat while this rises.
  double generated_tokens_per_sec() const {
    const double wall = wall_us();
    return wall > 0.0 ? 1e6 * static_cast<double>(total_generated_tokens) / wall
                      : 0.0;
  }

  // What the engine put through the model, per second. Against the rate above
  // it shows how much output each processed token bought.
  double model_input_tokens_per_sec() const {
    const double wall = wall_us();
    return wall > 0.0 ? 1e6 * static_cast<double>(model_input_tokens()) / wall
                      : 0.0;
  }

  // Steps that held both kinds at once. Implied by the overlap rather than
  // counted: every step holds decode, prefill, or both.
  std::uint64_t mixed_steps() const {
    const std::uint64_t overlap = steps_with_decode + steps_with_prefill;
    return overlap > steps ? overlap - steps : 0;
  }

  double mean_ttft_us() const {
    return ttft_count > 0 ? static_cast<double>(ttft_sum_us) / ttft_count : 0.0;
  }

  // The sentinel is never shown: with no samples there is no minimum.
  // Mean wall time a decode session waited per step it took part in, mixed
  // steps included. The engine-side counterpart to per-generation inter-token
  // latency, and a cross-check on it. Weighted by sessions, so it is not
  // directly comparable to the unweighted bucket means above -- read it as
  // what sessions experienced, not as what a step cost.
  double mean_decode_us_per_token() const {
    return decode_sessions_total > 0
        ? static_cast<double>(decode_session_time_sum_us) /
            decode_sessions_total
        : 0.0;
  }

  // Steps holding only one kind. Derived, because every step is decode-only,
  // prefill-only, or mixed.
  std::uint64_t decode_only_steps() const {
    const std::uint64_t mixed = mixed_steps();
    return steps_with_decode > mixed ? steps_with_decode - mixed : 0;
  }

  std::uint64_t prefill_only_steps() const {
    const std::uint64_t mixed = mixed_steps();
    return steps_with_prefill > mixed ? steps_with_prefill - mixed : 0;
  }

  double mean_decode_only_step_us() const {
    const std::uint64_t n = decode_only_steps();
    return n > 0 ? static_cast<double>(decode_only_latency_sum_us) / n : 0.0;
  }

  double mean_mixed_step_us() const {
    const std::uint64_t n = mixed_steps();
    return n > 0 ? static_cast<double>(mixed_latency_sum_us) / n : 0.0;
  }

  double mean_prefill_only_step_us() const {
    const std::uint64_t n = prefill_only_steps();
    return n > 0 ? static_cast<double>(prefill_only_latency_sum_us) / n : 0.0;
  }

  // Decode throughput measured only where no prefill shared the forward, so
  // every microsecond in the denominator was spent on these tokens. The one
  // rate here that compares across engines, since a fixed-batch decode
  // benchmark runs exactly this shape of step.
  double decode_only_tokens_per_sec() const {
    return decode_only_latency_sum_us > 0 ? 1e6 *
            static_cast<double>(decode_only_tokens) / decode_only_latency_sum_us
                                          : 0.0;
  }

  // Committed tokens the batch carried, averaged over steps.
  double mean_context_per_step() const {
    return steps > 0 ? static_cast<double>(context_sum) / steps : 0.0;
  }

  std::int64_t min_ttft_us() const {
    return ttft_count > 0 ? ttft_min_us : 0;
  }
};

namespace detail {

inline std::string fixed(double v, int places) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(places) << v;
  return os.str();
}

inline std::string ms(std::int64_t microseconds) {
  return fixed(static_cast<double>(microseconds) / 1000.0, 2);
}

} // namespace detail

// Returned rather than printed: these headers stay free of ExecuTorch's
// logging, so the caller decides where it goes.
ET_EXPERIMENTAL inline std::string format_report(const GenerationMetrics& m) {
  std::ostringstream os;
  os << "session " << m.sid << ": " << m.n_prompt_tokens << " prompt + "
     << m.n_generated_tokens << " generated tokens\n"
     << "  ttft        " << detail::ms(m.ttft_us()) << " ms = queue "
     << detail::ms(m.queue_wait_us()) << " + prefill "
     << detail::ms(m.prefill_span_us()) << " ("
     << detail::fixed(m.prefill_tokens_per_sec(), 1) << " tok/s, "
     << m.n_prefill_steps
     << (m.n_prefill_steps == 1 ? " step)\n" : " steps)\n");
  if (m.itl_count > 0) {
    // The decode span and its rate are omitted: both follow from the mean
    // below, which is decode time over decode tokens by construction.
    os << "  decode      " << detail::ms(m.itl_sum_us / m.itl_count)
       << " ms/token, max " << detail::ms(m.itl_max_us) << " -> "
       << detail::fixed(m.decode_tokens_per_sec(), 1) << " tok/s over "
       << m.n_decode_steps << " steps\n";
  }
  os << "  total       " << detail::ms(m.e2e_us()) << " ms\n";
  return os.str();
}

ET_EXPERIMENTAL inline std::string format_report(const EngineMetrics& m) {
  std::ostringstream os;
  const double wall = m.wall_us();
  const auto share = [wall](std::int64_t part) {
    return wall > 0.0 ? detail::fixed(100.0 * part / wall, 0)
                      : std::string("0");
  };
  // An empty bucket has no mean. Printing 0.00 ms would read as a measurement
  // rather than an absence, and the count of zero is the informative part.
  const auto bucket = [](double mean_us, std::uint64_t n) {
    return n > 0 ? detail::ms(static_cast<std::int64_t>(mean_us)) + " ms"
                 : std::string("--");
  };

  os << "engine\n"
     << "  run          " << detail::fixed(wall / 1e6, 2) << " s wall, "
     << m.steps << " steps";
  if (m.steps_failed > 0) {
    os << " (" << m.steps_failed << " failed)";
  }
  os << ", " << detail::fixed(100.0 * m.idle_fraction(), 1) << "% idle\n";
  if (m.init_us > 0) {
    os << "  init         " << detail::ms(m.init_us)
       << " ms executor setup, before the wall above\n";
  }
  os << "  generations  " << m.generations_completed << " of "
     << m.generations_started << ": " << m.finished_stop_token << " stop, "
     << m.finished_token_limit << " limit, " << m.finished_cancelled
     << " cancelled, " << m.finished_failed << " failed\n"
     << "  sessions     " << m.peak_concurrent_generations
     << " peak concurrent, " << m.sessions_refused << " refused at capacity\n"
     << "\n"
     // A partition, so the three shares add up. The gap between the first two
     // is what packing prefill alongside decode cost the decodes.
     << "  step time    decode-only  "
     << bucket(m.mean_decode_only_step_us(), m.decode_only_steps()) << " x"
     << m.decode_only_steps() << " (" << share(m.decode_only_latency_sum_us)
     << "%)\n"
     << "               mixed        "
     << bucket(m.mean_mixed_step_us(), m.mixed_steps()) << " x"
     << m.mixed_steps() << " (" << share(m.mixed_latency_sum_us) << "%)\n"
     << "               prefill-only "
     << bucket(m.mean_prefill_only_step_us(), m.prefill_only_steps()) << " x"
     << m.prefill_only_steps() << " (" << share(m.prefill_only_latency_sum_us)
     << "%)\n"
     << "\n"
     << "  decode       " << detail::fixed(m.decode_only_tokens_per_sec(), 1)
     << " tok/s on decode-only steps at "
     << detail::fixed(m.mean_decode_step_sessions(), 2) << " sessions/step\n"
     << "               "
     << detail::fixed(m.mean_decode_us_per_token() / 1000.0, 2)
     << " ms/token per session across all " << m.steps_with_decode
     << " decode steps\n"
     << "  context      " << detail::fixed(m.mean_context_per_step(), 0)
     << " tokens resident/step, max " << m.context_max << " per session\n"
     << "\n"
     << "  scheduler    admitted "
     << detail::fixed(m.mean_admitted_decode_sessions(), 2) << " of "
     << detail::fixed(m.mean_ready_decode_sessions(), 2)
     << " ready decode sessions ("
     << detail::fixed(100.0 * m.admitted_ratio(), 1) << "%)\n"
     << "  ttft         mean " << detail::ms(m.mean_ttft_us()) << " ms, min "
     << detail::ms(m.min_ttft_us()) << " ms, max " << detail::ms(m.ttft_max_us)
     << " ms over " << m.ttft_count
     << " generations\n"
     // Wall-clock rates: these move with the prompt-to-generation mix, so they
     // describe the workload as much as the engine.
     << "  delivered    " << m.total_generated_tokens << " generated -> "
     << detail::fixed(m.generated_tokens_per_sec(), 1) << " tok/s wall\n"
     << "  processed    " << m.model_input_tokens() << " tokens -> "
     << detail::fixed(m.model_input_tokens_per_sec(), 1) << " tok/s ("
     << m.decode_tokens_total << " decode + " << m.prefill_tokens_total
     << " prompt)\n";
  return os.str();
}

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
