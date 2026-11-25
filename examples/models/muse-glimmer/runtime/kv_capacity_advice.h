/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

// Muse Glimmer's advice for a CUDA KV admission, in terms of its export
// (export_solo / export_dflash) and serving (batching worker) options, and the
// DFlash draft pool's admission rule. Pure apart from logging, so it is
// testable without a GPU.

#include <executorch/backends/cuda/batching/kv_capacity.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace executorch::extension::llm::kv_advice {

namespace cb = ::executorch::backends::cuda::batching;

inline constexpr char kReservationFlags[] =
    "--max_sessions, --prefix_cache_entries or --prefix_capture_slots";
inline constexpr char kTokenLimitFlag[] =
    "--max_session_tokens (serve_batching: --max-context)";

// What to change when the CUDA executor reports `report`. A DFlash executor
// passes `session_token_cap`, the draft pool's per-session limit, so the
// advised context fits both pools.
inline std::vector<std::string> serving_advice(
    const cb::KvCapacityReport& report,
    std::optional<std::int64_t> session_token_cap = std::nullopt) {
  const auto capped = [&](std::int64_t tokens) {
    return session_token_cap ? std::min(tokens, *session_token_cap) : tokens;
  };
  const std::string sessions = std::to_string(report.sessions);
  const std::string limit = std::to_string(report.session_limit);
  std::vector<std::string> lines;
  switch (report.outcome) {
    case cb::KvAdmission::kAutomatic: {
      const std::int64_t tokens = capped(report.session_tokens);
      if (report.sequence_layout || tokens >= report.context_length) {
        break;
      }
      lines.push_back(
          "Muse Glimmer: each session may use " + std::to_string(tokens) +
          " of the model's " + std::to_string(report.context_length) +
          " context tokens; fewer " + kReservationFlags + " leave more");
      cb::KvCapacityReport full = report;
      full.session_tokens = static_cast<int>(report.context_length);
      std::string cells =
          "Muse Glimmer: the full context needs an export with "
          "--max-cells >= " +
          std::to_string(cb::required_cells(full));
      if (report.facts.bytes_per_cell > 0) {
        cells += " (" +
            cb::format_memory(
                     cb::required_cells(full) * report.facts.bytes_per_cell) +
            " of K/V)";
      }
      lines.push_back(cells);
      if (report.facts.has_windowed_layers) {
        lines.push_back(
            "Muse Glimmer: or --kv-layout sequence --max-kv-seqs >= " +
            sessions + ", which keeps sliding-window layers in ring buffers");
      }
      break;
    }
    case cb::KvAdmission::kNoBudget:
      lines.push_back(
          std::string("Muse Glimmer: lower ") + kReservationFlags +
          " to total at most " +
          std::to_string(std::max<std::int64_t>(0, report.cells - 1)) +
          ", or re-export with a larger --max-cells");
      break;
    case cb::KvAdmission::kCells: {
      const std::int64_t tokens = capped(cb::fitting_session_tokens(report));
      const std::int64_t fitting = cb::fitting_sessions(report);
      if (tokens > 0) {
        lines.push_back(
            std::string("Muse Glimmer: set ") + kTokenLimitFlag +
            " to at most " + std::to_string(tokens) + ", or lower " +
            kReservationFlags);
      } else if (fitting > 0) {
        lines.push_back(
            std::string("Muse Glimmer: lower ") + kReservationFlags +
            " to total at most " + std::to_string(fitting));
      } else {
        lines.push_back(
            std::string("Muse Glimmer: set ") + kTokenLimitFlag + " below " +
            std::to_string(report.cells));
      }
      std::string cells = "Muse Glimmer: or re-export with --max-cells >= " +
          std::to_string(cb::required_cells(report));
      if (report.facts.bytes_per_cell > 0) {
        cells += " (" +
            cb::format_memory(
                     cb::required_cells(report) * report.facts.bytes_per_cell) +
            " of K/V)";
      }
      lines.push_back(cells);
      if (report.facts.has_windowed_layers) {
        lines.push_back(
            "Muse Glimmer: for long contexts, re-export with --kv-layout "
            "sequence --max-kv-seqs >= " +
            sessions);
      }
      break;
    }
    case cb::KvAdmission::kSessionLimit:
      if (report.sequence_layout) {
        lines.push_back(
            "Muse Glimmer: re-export with --max-kv-seqs >= " + sessions +
            ", or lower " + kReservationFlags + " to total at most " + limit);
      } else {
        lines.push_back(
            std::string("Muse Glimmer: lower ") + kReservationFlags +
            " to total at most " + limit +
            ", or re-export with --kv-layout sequence --max-kv-seqs >= " +
            sessions);
      }
      break;
  }
  return lines;
}

// Logs serving_advice(): at Info for an automatic context, else as an error.
inline void log_serving_advice(
    const cb::KvCapacityReport& report,
    std::optional<std::int64_t> session_token_cap = std::nullopt) {
  for (const std::string& line : serving_advice(report, session_token_cap)) {
    if (report.outcome == cb::KvAdmission::kAutomatic) {
      ET_LOG(Info, "%s", line.c_str());
    } else {
      ET_LOG(Error, "%s", line.c_str());
    }
  }
}

// A DFlash draft cache as its program publishes it.
struct DraftPool {
  // A private history per sequence; else one shared pool of cells.
  bool sequence_layout = false;
  // The shared pool's cells, or each sequence's rows.
  int cells = 0;
  int max_owners = 0;
  int block = 0;
};

// Cells a cell-layout draft holds for every owner's block while drafting.
inline int temporary_cells(const DraftPool& pool) {
  return pool.sequence_layout ? 0 : pool.max_owners * pool.block;
}

// The per-session context the draft pool allows: each sequence's rows less
// its block, or the shared cells less the padding cell and every owner's block,
// divided among the sessions.
inline int draft_session_budget(int sessions, const DraftPool& pool) {
  return pool.sequence_layout
      ? pool.cells - pool.block
      : (pool.cells - 1 - temporary_cells(pool)) / sessions;
}

struct DraftAdmission {
  // The context both the target and the draft hold.
  int max_context;
  // Whether it exceeds the draft block and, when a limit was requested,
  // equals it.
  bool fits;
};

// The DFlash executor's admission of `sessions` sessions whose target budget is
// `target_tokens`; `requested_tokens` is the configured limit, zero for
// automatic.
inline DraftAdmission draft_admission(
    int sessions,
    int requested_tokens,
    int target_tokens,
    const DraftPool& pool) {
  const int max_context =
      std::min(target_tokens, draft_session_budget(sessions, pool));
  return {
      max_context,
      max_context > pool.block &&
          (requested_tokens == 0 || max_context == requested_tokens)};
}

// Why draft_admission() refused, and what would fit. One entry per log line.
inline std::vector<std::string> draft_shortfall_lines(
    int sessions,
    int requested_tokens,
    int target_tokens,
    const DraftPool& pool,
    const cb::KvReservations& reservations) {
  const int budget = draft_session_budget(sessions, pool);
  const std::string block = std::to_string(pool.block);
  const std::string who = cb::describe_sessions(sessions, reservations);
  const int temporary = temporary_cells(pool);
  std::vector<std::string> lines;
  if (target_tokens <= pool.block && target_tokens <= budget) {
    lines.push_back(
        "DFlash: a " + std::to_string(target_tokens) +
        "-token session budget must exceed the " + block +
        "-token draft block");
    lines.push_back(
        requested_tokens != 0 ? std::string("Muse Glimmer: raise ") +
                kTokenLimitFlag + " above " + block
                              : std::string("Muse Glimmer: lower ") +
                kReservationFlags + ", or re-export with a larger --max-cells");
    return lines;
  }
  if (budget <= pool.block) {
    if (pool.sequence_layout) {
      lines.push_back(
          "DFlash: each draft sequence's " + std::to_string(pool.cells) +
          " rows leave " + std::to_string(budget) + " tokens beside the " +
          block + "-token block, which they must exceed");
      lines.push_back(
          "Muse Glimmer: re-export the draft with a larger sequence capacity");
      return lines;
    }
    const std::int64_t needed =
        static_cast<std::int64_t>(sessions) * (pool.block + 1) + temporary + 1;
    lines.push_back(
        "DFlash: " + std::to_string(pool.cells) + " draft cells, less " +
        std::to_string(temporary) +
        " temporary block cells and 1 padding, leave " +
        std::to_string(std::max(0, budget)) + " tokens for each of " + who +
        ", which must exceed the " + block + "-token block");
    lines.push_back(
        "Muse Glimmer: reserve at most " +
        std::to_string(
            std::max(0, (pool.cells - 1 - temporary) / (pool.block + 1))) +
        " sessions, or re-export with --draft-max-cells >= " +
        std::to_string(needed));
    return lines;
  }
  // A requested limit the draft cannot hold for every session.
  if (pool.sequence_layout) {
    lines.push_back(
        "DFlash: " + std::to_string(requested_tokens) +
        " tokens per session exceed each draft sequence's " +
        std::to_string(pool.cells) + " rows less the " + block +
        "-token block");
  } else {
    lines.push_back(
        "DFlash: " + who + " x " + std::to_string(requested_tokens) + " + " +
        std::to_string(temporary) +
        " temporary draft-block cells + 1 padding = " +
        std::to_string(
            cb::cell_budget(sessions, requested_tokens, pool.cells, temporary)
                .required_cells) +
        " draft cells needed, " + std::to_string(pool.cells) + " available");
  }
  std::string advice = std::string("Muse Glimmer: set ") + kTokenLimitFlag +
      " to at most " + std::to_string(budget);
  if (!pool.sequence_layout) {
    advice += ", or re-export with --draft-max-cells >= " +
        std::to_string(cb::cell_budget(
                           sessions, requested_tokens, pool.cells, temporary)
                           .required_cells);
  }
  lines.push_back(advice);
  return lines;
}

} // namespace executorch::extension::llm::kv_advice
