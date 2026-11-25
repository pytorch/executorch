/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The KV admission arithmetic CudaExecutor checks at startup, and the
// diagnostics it logs when a configuration does not fit. Pure, so it is
// testable without a GPU or a program.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace executorch::backends::cuda::batching {

// What the resident sessions an executor provisions are for. A serving runtime
// hosts its live sessions, its retained prefix entries and its in-flight prefix
// captures each on its own executor session, and any of them may fill its
// whole token budget. All zero when the caller does not say.
struct KvReservations {
  int live_sessions = 0;
  int prefix_entries = 0;
  int capture_slots = 0;

  bool known() const {
    return live_sessions > 0 || prefix_entries > 0 || capture_slots > 0;
  }
  std::int64_t total() const {
    return static_cast<std::int64_t>(live_sessions) + prefix_entries +
        capture_slots;
  }
};

// The cell layout's admission: every resident session may hold its token
// budget at once, `reserved_cells` more are held for other uses (e.g. a
// DFlash draft's temporary owner blocks), and one more cell is the padding
// scratch row.
struct CellBudget {
  // sessions * session_tokens + reserved_cells + 1.
  std::int64_t required_cells;
  // The program's cells.
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

// "16 resident sessions (4 live + 8 prefix entries + 4 capture slots)". A
// breakdown that does not cover every session names the rest "other"; one that
// claims more sessions than there are is dropped.
inline std::string describe_sessions(
    int sessions,
    const KvReservations& reservations) {
  std::string text = std::to_string(sessions) + " resident sessions";
  const std::int64_t listed = reservations.total();
  if (!reservations.known() || listed > sessions) {
    return text;
  }
  text += " (" + std::to_string(reservations.live_sessions) + " live + " +
      std::to_string(reservations.prefix_entries) + " prefix entries + " +
      std::to_string(reservations.capture_slots) + " capture slots";
  if (listed < sessions) {
    text += " + " + std::to_string(sessions - listed) + " other";
  }
  return text + ")";
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

// What a cell-layout diagnostic knows about the program, beyond its cells.
struct CellPoolFacts {
  // Every layer's K and V for one token; zero leaves the memory estimate out.
  std::int64_t bytes_per_cell = 0;
  // Whether any layer attends a sliding window, which the sequence layout
  // keeps in a ring buffer.
  bool has_windowed_layers = false;
};

// Why `sessions` sessions of `session_tokens` tokens do not fit `max_cells`
// cells, and what would. One entry per log line, each well under the 256-byte
// ET_LOG limit.
inline std::vector<std::string> cell_shortfall_lines(
    int sessions,
    int session_tokens,
    std::int64_t max_cells,
    const KvReservations& reservations,
    const CellPoolFacts& facts = {}) {
  const CellBudget budget = cell_budget(sessions, session_tokens, max_cells);
  const std::string required = std::to_string(budget.required_cells);
  std::vector<std::string> lines;
  lines.push_back(
      "CudaExecutor: " + std::to_string(sessions) + " sessions of " +
      std::to_string(session_tokens) + " tokens exceed the program's " +
      std::to_string(max_cells) + " cells, one of which is reserved");
  lines.push_back(
      "CudaExecutor: " + describe_sessions(sessions, reservations) +
      " share one cell pool; " + std::to_string(sessions) + " x " +
      std::to_string(session_tokens) + " + 1 padding = " + required +
      " cells needed, " + std::to_string(max_cells) + " available");
  if (budget.max_session_tokens > 0) {
    lines.push_back(
        "CudaExecutor: at most " + std::to_string(budget.max_session_tokens) +
        " tokens per session fit; lower the session token limit or reserve "
        "fewer live sessions, prefix entries or capture slots");
  } else {
    lines.push_back(
        "CudaExecutor: not even one token per session fits; reserve at most " +
        std::to_string(max_cells > 1 ? max_cells - 1 : 0) + " sessions");
  }
  std::string reexport =
      "CudaExecutor: or re-export with --max-cells >= " + required;
  if (facts.bytes_per_cell > 0) {
    reexport += " (" +
        format_memory(budget.required_cells * facts.bytes_per_cell) +
        " of K/V at " + std::to_string(facts.bytes_per_cell) +
        " bytes per cell)";
  }
  lines.push_back(reexport);
  if (facts.has_windowed_layers) {
    lines.push_back(
        "CudaExecutor: this model has sliding-window layers; --kv-layout "
        "sequence keeps them in ring buffers and may need far less memory");
  }
  return lines;
}

// Why `sessions` sessions exceed the layout's `max_seqs` resident sequences,
// and what would fit. The sequence layout gives each a column the export sized
// with --max-kv-seqs; the cell layout tracks at most a fixed number of
// sequences (one owner bit each). One entry per log line.
inline std::vector<std::string> sequence_shortfall_lines(
    int sessions,
    int max_seqs,
    const KvReservations& reservations,
    bool sequence_layout) {
  const std::string count = std::to_string(sessions);
  if (sequence_layout) {
    return {
        "CudaExecutor: " + count +
            " resident sessions requested, but the layout holds " +
            std::to_string(max_seqs),
        "CudaExecutor: " + describe_sessions(sessions, reservations) +
            " each need a sequence column",
        "CudaExecutor: re-export with --max-kv-seqs >= " + count +
            ", or reserve fewer live sessions, prefix entries or capture "
            "slots",
    };
  }
  return {
      "CudaExecutor: " + count +
          " resident sessions requested, but the cell layout tracks at most " +
          std::to_string(max_seqs),
      "CudaExecutor: " + describe_sessions(sessions, reservations) +
          " each take one of its sequence slots",
      "CudaExecutor: reserve fewer live sessions, prefix entries or capture "
      "slots, or re-export with --kv-layout sequence --max-kv-seqs >= " +
          count,
  };
}

// Why a DFlash draft pool of `max_cells` cells cannot hold `sessions`
// histories of `session_tokens` tokens plus `temporary_cells` cells for the
// owners' draft blocks and one padding cell. One entry per log line.
inline std::vector<std::string> draft_shortfall_lines(
    int sessions,
    int session_tokens,
    std::int64_t max_cells,
    std::int64_t temporary_cells,
    const KvReservations& reservations) {
  const CellBudget budget =
      cell_budget(sessions, session_tokens, max_cells, temporary_cells);
  std::vector<std::string> lines{
      "DFlash: " + describe_sessions(sessions, reservations) + " x " +
          std::to_string(session_tokens) + " + " +
          std::to_string(temporary_cells) +
          " temporary draft-block cells + 1 padding = " +
          std::to_string(budget.required_cells) + " draft cells needed, " +
          std::to_string(max_cells) + " available",
  };
  if (budget.max_session_tokens > 0) {
    lines.push_back(
        "DFlash: at most " + std::to_string(budget.max_session_tokens) +
        " tokens per session fit the draft pool; lower the session token "
        "limit, reserve fewer sessions, or re-export with --draft-max-cells "
        ">= " +
        std::to_string(budget.required_cells));
  } else {
    lines.push_back(
        "DFlash: the draft pool cannot hold its temporary blocks; re-export "
        "with --draft-max-cells >= " +
        std::to_string(budget.required_cells));
  }
  return lines;
}

} // namespace executorch::backends::cuda::batching
