/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

// Muse Glimmer's notes on a CUDA KV admission: which pool refused (the
// target's or the DFlash draft's), what it holds, and where the sizing rules
// are documented, as facts with a pointer to BATCHING_SERVING.md. Also the
// DFlash draft pool's admission rule, which the DFlash executor calls. Pure
// apart from logging, so it is testable without a GPU.

#include <executorch/backends/cuda/batching/kv_capacity.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace executorch::extension::llm::kv_notes {

namespace cb = ::executorch::backends::cuda::batching;

// A DFlash draft cache as its program publishes it.
struct DraftPool {
  // A private history per sequence; else one shared pool of cells.
  bool sequence_layout = false;
  // The shared pool's cells, or each sequence's rows.
  std::int64_t cells = 0;
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
inline std::int64_t draft_session_budget(int sessions, const DraftPool& pool) {
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
  const int max_context = static_cast<int>(std::min<std::int64_t>(
      target_tokens, draft_session_budget(sessions, pool)));
  return {
      max_context,
      max_context > pool.block &&
          (requested_tokens == 0 || max_context == requested_tokens)};
}

inline constexpr char kResidentSessions[] =
    "Muse Glimmer: resident sessions = --max_sessions/--max-sessions + "
    "--prefix_cache_entries/--prefix-cache-entries + capture slots "
    "(--prefix_capture_slots/--prefix-capture-slots; 0 sizes them "
    "automatically)";
inline constexpr char kSizingPointer[] =
    "Muse Glimmer: see \"KV capacity and long contexts\" in "
    "BATCHING_SERVING.md for the sizing formulas";
inline constexpr char kSequenceLayoutHint[] =
    "Muse Glimmer: for long contexts on sliding-window models, a --kv-layout "
    "sequence export (export_solo or export_dflash) keeps those layers in "
    "ring buffers and is recommended over a larger cell pool";

// Notes on CudaExecutor's report for the target pool. `dflash` notes that a
// cell shortfall may also concern the draft pool.
inline std::vector<std::string> target_notes(
    const cb::KvCapacityReport& report,
    bool dflash = false) {
  std::vector<std::string> lines;
  // The sliding-window hint where a larger cell pool is the alternative; the
  // backend's own cell-shortfall and session-limit lines already cover the
  // sequence layout.
  bool layout_hint =
      !report.sequence_layout && report.facts.has_windowed_layers;
  switch (report.outcome) {
    case cb::KvAdmission::kAutomatic:
      // Nothing to note when the pool holds the full context, but for at most
      // the padding cell.
      if (report.sequence_layout ||
          report.session_tokens >= report.context_length ||
          report.cells >= report.sessions * report.context_length) {
        return lines;
      }
      lines.push_back(
          "Muse Glimmer: the target pool gives each session " +
          std::to_string(report.session_tokens) + " of the model's " +
          std::to_string(report.context_length) + " context tokens");
      break;
    case cb::KvAdmission::kNoBudget:
    case cb::KvAdmission::kCells:
      lines.push_back(
          "Muse Glimmer: the target pool refused the sessions" +
          std::string(
              dflash ? "; a DFlash export also sizes a draft pool "
                       "(--draft-max-cells, default --max-cells)"
                     : ""));
      layout_hint = layout_hint && report.outcome == cb::KvAdmission::kNoBudget;
      break;
    case cb::KvAdmission::kSessionLimit:
      lines.push_back("Muse Glimmer: the target pool refused the sessions");
      layout_hint = false;
      break;
  }
  lines.push_back(kResidentSessions);
  lines.push_back(kSizingPointer);
  if (layout_hint) {
    lines.push_back(kSequenceLayoutHint);
  }
  return lines;
}

// Notes on the DFlash draft pool for `sessions` sessions whose target budget is
// `target_tokens` (`requested_tokens`, zero for automatic): empty unless the
// draft is the pool that limits or refuses them, or the target's budget is at
// most the draft block.
inline std::vector<std::string> draft_notes(
    int sessions,
    int requested_tokens,
    int target_tokens,
    const DraftPool& pool,
    const cb::KvReservations& reservations = {}) {
  std::vector<std::string> lines;
  const std::int64_t budget = draft_session_budget(sessions, pool);
  const std::string block = std::to_string(pool.block);
  if (target_tokens <= pool.block && budget >= target_tokens) {
    lines.push_back(
        "DFlash: the target pool's " + std::to_string(target_tokens) +
        "-token session budget does not exceed the " + block +
        "-token draft block");
    lines.push_back(kSizingPointer);
    return lines;
  }
  if (budget >= target_tokens) {
    // The target is the limit and DFlash accepts it.
    return lines;
  }
  const bool refused =
      !draft_admission(sessions, requested_tokens, target_tokens, pool).fits;
  std::string line = pool.sequence_layout
      ? "DFlash: each draft sequence's " + std::to_string(pool.cells) +
          " rows hold " + std::to_string(budget) + " tokens beside its " +
          block + "-token block"
      : "DFlash: the draft pool's " + std::to_string(pool.cells) +
          " cells hold " + std::to_string(std::max<std::int64_t>(0, budget)) +
          " tokens for each of " + std::to_string(sessions) +
          " sessions after " + std::to_string(pool.max_owners) + " x " + block +
          " temporary block cells and 1 padding cell";
  line += refused ? ", so it refused the sessions" : ", which limits them";
  lines.push_back(line);
  if (requested_tokens != 0 && refused && pool.sequence_layout) {
    // Each sequence has its own rows: no session count changes this.
    lines.push_back(
        "DFlash: on the sequence layout a session holds at most --max-seq-len "
        "less the block, " +
        std::to_string(pool.cells) + " - " + block + " = " +
        std::to_string(budget) + " draft tokens, so a requested " +
        std::to_string(requested_tokens) + "-token --max-context is refused");
  } else if (requested_tokens != 0) {
    if (refused) {
      // The target admitted the requested limit without a report: say where
      // the sessions sharing the draft pool come from.
      lines.push_back(
          "DFlash: " + cb::describe_sessions(sessions, reservations) +
          " share the draft pool");
      lines.push_back(kResidentSessions);
    }
    lines.push_back(
        "DFlash: " + std::to_string(requested_tokens) +
        " tokens per session were requested; " + std::to_string(sessions) +
        " x " + std::to_string(requested_tokens) + " + " +
        std::to_string(temporary_cells(pool)) + " + 1 = " +
        std::to_string(
            cb::cell_budget(
                sessions, requested_tokens, pool.cells, temporary_cells(pool))
                .required_cells) +
        " draft cells hold them");
  } else {
    lines.push_back(
        "DFlash: the target pool holds " + std::to_string(target_tokens) +
        " tokens per session");
  }
  if (budget <= pool.block) {
    lines.push_back(
        "DFlash: a session's context must exceed the " + block +
        "-token draft block");
  }
  lines.push_back(kSizingPointer);
  return lines;
}

// Logs notes: at Info when the sessions were admitted, else as errors.
inline void log_notes(const std::vector<std::string>& lines, bool admitted) {
  for (const std::string& line : lines) {
    if (admitted) {
      ET_LOG(Info, "%s", line.c_str());
    } else {
      ET_LOG(Error, "%s", line.c_str());
    }
  }
}

} // namespace executorch::extension::llm::kv_notes
