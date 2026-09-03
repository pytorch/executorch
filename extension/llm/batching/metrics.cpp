/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// The report formatters, kept out of metrics.h so the stream headers they need
// stay out of every translation unit that merely reads a counter.

#include <executorch/extension/llm/batching/metrics.h>

#include <iomanip>
#include <sstream>
#include <string>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

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
std::string format_report(const GenerationMetrics& m) {
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

std::string format_report(const EngineMetrics& m) {
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
