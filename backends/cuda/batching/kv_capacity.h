/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The KV admission arithmetic CudaExecutor checks at startup, and the
// layout-level diagnostics it logs. Pure and GPU-free, so it is testable
// without a program. Wording is neutral: how a program is exported or a server
// configured is the caller's to explain, from the KvCapacityReport.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace executorch::backends::cuda::batching {

// What the resident sessions an executor provisions are for. A serving runtime
// hosts its live sessions, its retained prefix entries and its in-flight prefix
// captures each on its own executor session, and any of them may fill its
// whole token budget. All zero when the caller does not say; otherwise the
// executor requires them to total its resident sessions.
struct KvReservations {
  int live_sessions = 0;
  int prefix_entries = 0;
  int capture_slots = 0;

  std::int64_t total() const {
    return static_cast<std::int64_t>(live_sessions) + prefix_entries +
        capture_slots;
  }
};

// The cell layout's admission: every resident session may hold its token
// budget at once, `reserved_cells` more are held for other uses, and one more
// cell is the padding scratch row.
struct CellBudget {
  // sessions * session_tokens + reserved_cells + 1.
  std::int64_t required_cells;
  // The pool's cells.
  std::int64_t available_cells;
  // The largest per-session budget that fits:
  // (cells - 1 - reserved_cells) / sessions.
  std::int64_t max_session_tokens;

  bool fits() const {
    return required_cells <= available_cells;
  }
};

inline CellBudget cell_budget(
    int sessions,
    int session_tokens,
    std::int64_t max_cells,
    std::int64_t reserved_cells = 0) {
  CellBudget budget{};
  budget.required_cells =
      static_cast<std::int64_t>(sessions) * session_tokens + reserved_cells + 1;
  budget.available_cells = max_cells;
  const std::int64_t free_cells = max_cells - 1 - reserved_cells;
  budget.max_session_tokens =
      sessions > 0 && free_cells > 0 ? free_cells / sessions : 0;
  return budget;
}

// "16 resident sessions (4 live + 8 prefix entries + 4 capture slots)", or
// only the count when no breakdown totals `sessions`.
inline std::string describe_sessions(
    int sessions,
    const KvReservations& reservations) {
  std::string text = std::to_string(sessions) + " resident sessions";
  if (sessions <= 0 || reservations.total() != sessions) {
    return text;
  }
  return text + " (" + std::to_string(reservations.live_sessions) + " live + " +
      std::to_string(reservations.prefix_entries) + " prefix entries + " +
      std::to_string(reservations.capture_slots) + " capture slots)";
}

// Bytes with one decimal: MiB while that rounds below 1024.0 MiB, else GiB
// ("512.0 MiB", "1.0 GiB", "104.0 GiB").
inline std::string format_memory(std::int64_t bytes) {
  constexpr double kMiB = 1024.0 * 1024.0;
  const double mib = static_cast<double>(bytes) / kMiB;
  char text[32];
  if (std::round(mib * 10.0) < 10240.0) {
    std::snprintf(text, sizeof(text), "%.1f MiB", mib);
  } else {
    std::snprintf(text, sizeof(text), "%.1f GiB", mib / 1024.0);
  }
  return text;
}

// What the diagnostics report about a program's cache, beyond its size.
struct CellPoolFacts {
  // Every layer's K and V for one token; zero when the dtype is not a valid
  // ScalarType, which leaves the memory estimate out.
  std::int64_t bytes_per_cell = 0;
  // Whether any layer attends a sliding window (a ring layer), which a
  // sequence-layout program keeps in a ring buffer.
  bool has_windowed_layers = false;
};

inline CellPoolFacts cell_pool_facts(
    const ::executorch::extension::llm::cache::CacheGeometry& geometry,
    int kv_dtype) {
  using ::executorch::extension::llm::cache::LayerPolicy;
  const auto dtype = static_cast<::executorch::aten::ScalarType>(kv_dtype);
  const std::int64_t element = ::executorch::runtime::isValid(dtype)
      ? static_cast<std::int64_t>(::executorch::runtime::elementSize(dtype))
      : 0;
  CellPoolFacts facts;
  for (const auto& layer : geometry.layers) {
    facts.bytes_per_cell += 2 * element * layer.n_kv_heads * layer.head_dim;
    facts.has_windowed_layers = facts.has_windowed_layers ||
        (layer.policy.kind == LayerPolicy::Kind::Ring &&
         layer.policy.window > 0);
  }
  return facts;
}

// What CudaExecutor reports about its KV admission
// (CudaExecutorConfig::capacity_report).
enum class KvAdmission {
  // Automatic context mode resolved a per-session token budget.
  kAutomatic,
  // Automatic context mode: not one token per session fits the cells.
  kNoBudget,
  // An explicit token limit: the cell pool cannot hold every session's budget.
  kCells,
  // More resident sessions than the layout tracks. Reported in preference to
  // the cases above, which no token limit or cell count could fix.
  kSessionLimit,
};

// Automatic context mode's per-session budget on the cell layout:
// min(context, (cells - 1) / sessions).
inline std::int64_t automatic_session_tokens(
    int sessions,
    std::int64_t context_length,
    std::int64_t cells) {
  return sessions > 0 ? std::min(context_length, (cells - 1) / sessions) : 0;
}

struct KvAdmissionCheck {
  KvAdmission outcome;
  // The requested budget, or automatic context mode's resolved one.
  std::int64_t session_tokens;
};

// What the cell layout reports for `sessions` resident sessions asking for
// `requested_tokens` each (zero for automatic context mode), over `cells`
// cells and at most `session_limit` resident sessions: kSessionLimit past the
// limit; in automatic context mode kAutomatic with the resolved budget, or
// kNoBudget; else kCells. CudaExecutor calls it in automatic context mode, and
// for a requested limit only once cell_budget() has refused it.
inline KvAdmissionCheck check_admission(
    int sessions,
    int requested_tokens,
    std::int64_t context_length,
    std::int64_t cells,
    int session_limit) {
  const std::int64_t tokens = requested_tokens != 0
      ? requested_tokens
      : automatic_session_tokens(sessions, context_length, cells);
  if (sessions > session_limit) {
    return {KvAdmission::kSessionLimit, tokens};
  }
  if (requested_tokens == 0) {
    return {
        tokens > 0 ? KvAdmission::kAutomatic : KvAdmission::kNoBudget, tokens};
  }
  return {KvAdmission::kCells, tokens};
}

struct KvCapacityReport {
  KvAdmission outcome = KvAdmission::kAutomatic;
  bool sequence_layout = false;
  int sessions = 0;
  KvReservations reservations;
  // Whether no per-session limit was requested (automatic context mode).
  bool automatic = false;
  // The requested per-session budget, or automatic context mode's resolved
  // one (zero when none fits).
  int session_tokens = 0;
  std::int64_t context_length = 0;
  // The cell layout's pool; zero for the sequence layout.
  std::int64_t cells = 0;
  // The most resident sessions the layout tracks: the cache's own count once
  // it is built, else the program's sequence columns or the cell table's
  // limit.
  int session_limit = 0;
  CellPoolFacts facts;
};

// Cells holding every session's budget: sessions * session_tokens + 1.
inline std::int64_t required_cells(const KvCapacityReport& report) {
  return cell_budget(report.sessions, report.session_tokens, report.cells)
      .required_cells;
}

// The largest per-session budget the cells hold for the report's sessions,
// capped by the context.
inline std::int64_t fitting_session_tokens(const KvCapacityReport& report) {
  return std::min(
      report.context_length,
      cell_budget(report.sessions, report.session_tokens, report.cells)
          .max_session_tokens);
}

// The most sessions of the report's budget the cells hold, capped by the
// layout's session limit.
inline std::int64_t fitting_sessions(const KvCapacityReport& report) {
  if (report.session_tokens <= 0) {
    return 0;
  }
  std::int64_t sessions =
      std::max<std::int64_t>(0, report.cells - 1) / report.session_tokens;
  if (report.session_limit > 0) {
    sessions = std::min<std::int64_t>(sessions, report.session_limit);
  }
  return sessions;
}

// The report as log lines, each well under the 256-byte ET_LOG limit. They
// describe the layout's limits and what would fit, without naming any export
// or serving option.
inline std::vector<std::string> describe_capacity(
    const KvCapacityReport& report) {
  const std::string who =
      describe_sessions(report.sessions, report.reservations);
  const std::string sessions = std::to_string(report.sessions);
  const std::string tokens = std::to_string(report.session_tokens);
  const std::string cells = std::to_string(report.cells);
  const std::string limit = std::to_string(report.session_limit);
  std::vector<std::string> lines;
  switch (report.outcome) {
    case KvAdmission::kAutomatic:
      if (report.sequence_layout) {
        lines.push_back(
            "CudaExecutor: automatic context: " + who + " each get " + tokens +
            " tokens of a private sequence column");
      } else {
        lines.push_back(
            "CudaExecutor: automatic context: " + who + " share " + cells +
            " cells; each may hold " + tokens + " tokens (model context " +
            std::to_string(report.context_length) + ")");
      }
      break;
    case KvAdmission::kNoBudget:
      lines.push_back(
          "CudaExecutor: no cell budget remains: " + who + " need at least " +
          std::to_string(report.sessions + 1LL) +
          " cells with the padding cell, the program has " + cells);
      lines.push_back(
          "CudaExecutor: at most " +
          std::to_string(std::max<std::int64_t>(0, report.cells - 1)) +
          " one-token sessions fit");
      break;
    case KvAdmission::kCells: {
      lines.push_back(
          "CudaExecutor: " + sessions + " sessions of " + tokens +
          " tokens exceed the program's " + cells +
          " cells, one of which is reserved");
      const std::string required = std::to_string(required_cells(report));
      lines.push_back(
          "CudaExecutor: " + who + " share one cell pool; " + sessions + " x " +
          tokens + " + 1 padding = " + required + " cells needed, " + cells +
          " available");
      const std::int64_t fit_tokens = fitting_session_tokens(report);
      const std::int64_t fit_sessions = fitting_sessions(report);
      if (fit_tokens > 0) {
        std::string fit = "CudaExecutor: at most " +
            std::to_string(fit_tokens) + " tokens per session fit " + sessions +
            " sessions";
        if (fit_sessions > 0) {
          fit += ", or " + std::to_string(fit_sessions) + " sessions of " +
              tokens + " tokens";
        }
        lines.push_back(fit);
      } else if (fit_sessions > 0) {
        lines.push_back(
            "CudaExecutor: not even one token per session fits " + sessions +
            " sessions; at most " + std::to_string(fit_sessions) +
            " sessions of " + tokens + " tokens fit");
      } else {
        lines.push_back(
            "CudaExecutor: one session of " + tokens + " tokens alone needs " +
            std::to_string(report.session_tokens + 1LL) + " cells");
      }
      std::string larger = "CudaExecutor: a program with at least " + required +
          " cells would hold them";
      if (report.facts.bytes_per_cell > 0) {
        larger += " (" +
            format_memory(
                      required_cells(report) * report.facts.bytes_per_cell) +
            " of K/V at " + std::to_string(report.facts.bytes_per_cell) +
            " bytes per cell)";
      }
      lines.push_back(larger);
      if (report.facts.has_windowed_layers) {
        lines.push_back(
            "CudaExecutor: the model has sliding-window layers, which a "
            "sequence-layout program keeps in ring buffers; that may need far "
            "less memory than a larger cell pool");
      }
      break;
    }
    case KvAdmission::kSessionLimit:
      if (report.sequence_layout) {
        lines.push_back(
            "CudaExecutor: " + who + " exceed the program's " + limit +
            " sequence columns, one per resident session");
        lines.push_back(
            "CudaExecutor: reserve at most " + limit +
            " sessions, or serve a program with at least " + sessions +
            " sequence columns");
      } else {
        lines.push_back(
            "CudaExecutor: " + who + " exceed the cell layout's limit of " +
            limit + " resident sequences, whatever its cells");
        // With a known budget the cells may hold fewer sessions than the limit.
        const std::string fit = !report.automatic && report.session_tokens > 0
            ? std::to_string(fitting_sessions(report)) + " sessions of " +
                tokens + " tokens fit its cells and limit"
            : "at most " + limit + " sessions fit its limit";
        lines.push_back(
            "CudaExecutor: " + fit +
            "; a sequence-layout program would need at least " + sessions +
            " sequence columns");
      }
      break;
  }
  return lines;
}

} // namespace executorch::backends::cuda::batching
