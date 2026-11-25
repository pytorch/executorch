/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

// Muse Glimmer's advice for a CUDA KV admission, in terms of its export
// (export_solo / export_dflash) and serving (native batching worker and
// serve_batching) options, with the DFlash draft pool's admission rule.
//
// Every fix the advice names is checked before it is offered: applied alone to
// the configuration, it must pass every capacity check (check_startup(), which
// uses CudaExecutor's check_admission() and the DFlash executor's
// draft_admission()). A change no single option fixes is described, not
// prescribed. Pure apart from logging, so it is testable without a GPU.

#include <executorch/backends/cuda/batching/kv_capacity.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace executorch::extension::llm::kv_advice {

namespace cb = ::executorch::backends::cuda::batching;

// The native worker's gflags, then serve_batching's options.
inline constexpr char kSessionFlags[] =
    "--max_sessions/--max-sessions, --prefix_cache_entries/"
    "--prefix-cache-entries or --prefix_capture_slots/--prefix-capture-slots";
inline constexpr char kTokenLimitFlag[] = "--max_session_tokens/--max-context";
inline constexpr char kCaptureNote[] =
    "Muse Glimmer: capture slots default to automatic sizing (0); lower them "
    "with an explicit positive --prefix_capture_slots/--prefix-capture-slots";

// The Muse Glimmer factories' rule for a configured per-session limit: zero
// (automatic) or room for a prompt and a prediction, so never one token.
inline bool valid_requested_tokens(int session_tokens) {
  return session_tokens != 1;
}

// The factories' rule for the context a session gets: a prompt and a
// prediction.
inline bool holds_prompt_and_prediction(std::int64_t session_tokens) {
  return session_tokens > 1;
}

// A DFlash draft cache as its program publishes it.
struct DraftPool {
  // A private history per sequence; else one shared pool of cells.
  bool sequence_layout = false;
  // The shared pool's cells, or each sequence's rows.
  std::int64_t cells = 0;
  int max_owners = 0;
  int block = 0;
  // The draft's K and V for one token in every layer; zero if unknown.
  std::int64_t bytes_per_cell = 0;
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

// What Muse Glimmer's CUDA startup admission depends on.
struct StartupConfig {
  bool sequence_layout = false;
  int sessions = 0;
  // Zero for automatic context mode.
  int requested_tokens = 0;
  std::int64_t context_length = 0;
  // The target's cells (cell layout).
  std::int64_t cells = 0;
  // The most resident sessions the target layout tracks; zero for no limit.
  int session_limit = 0;
  std::optional<DraftPool> draft;
};

inline StartupConfig startup_config(
    const cb::KvCapacityReport& report,
    const std::optional<DraftPool>& draft = std::nullopt) {
  StartupConfig config;
  config.sequence_layout = report.sequence_layout;
  config.sessions = report.sessions;
  config.requested_tokens = report.automatic ? 0 : report.session_tokens;
  config.context_length = report.context_length;
  config.cells = report.cells;
  config.session_limit = report.session_limit;
  config.draft = draft;
  return config;
}

struct StartupResult {
  bool admitted;
  // Each session's context: the requested limit, or the resolved budget.
  std::int64_t session_tokens;
};

// The fewest tokens a session may get: a prompt and a prediction, or more
// than the DFlash draft block.
inline std::int64_t least_session_tokens(
    const std::optional<DraftPool>& draft) {
  return draft ? draft->block + 1 : 2;
}

// The capacity checks, with the functions the factories and executors call:
// the factories' valid_requested_tokens(); CudaExecutor's
// valid_session_limits(), within_context(), valid_cell_count() and
// check_admission(); then holds_prompt_and_prediction() for a non-DFlash
// session, or the DFlash executor's draft_admission() (whose draft pool also
// fits an int).
inline StartupResult check_startup(const StartupConfig& config) {
  if (!valid_requested_tokens(config.requested_tokens) ||
      !cb::valid_session_limits(config.sessions, config.requested_tokens) ||
      !cb::within_context(config.requested_tokens, config.context_length) ||
      (!config.sequence_layout && !cb::valid_cell_count(config.cells)) ||
      (config.draft && !cb::valid_cell_count(config.draft->cells))) {
    return {false, 0};
  }
  const cb::KvAdmissionCheck target = cb::check_admission(
      config.sequence_layout,
      config.sessions,
      config.requested_tokens,
      config.context_length,
      config.cells,
      config.session_limit);
  if (target.outcome != cb::KvAdmission::kAdmitted &&
      target.outcome != cb::KvAdmission::kAutomatic) {
    return {false, target.session_tokens};
  }
  if (!config.draft) {
    return {
        holds_prompt_and_prediction(target.session_tokens),
        target.session_tokens};
  }
  const DraftAdmission draft = draft_admission(
      config.sessions,
      config.requested_tokens,
      static_cast<int>(target.session_tokens),
      *config.draft);
  return {draft.fits, draft.max_context};
}

enum class FixKind {
  // Total resident sessions of at most `value`.
  kSessions,
  // A per-session token limit in [`low`, `value`].
  kTokenLimit,
  // A re-export with at least `value` target cells and, under DFlash, at
  // least `draft_cells` draft cells.
  kCells,
  // A sequence-layout re-export with at least `value` sequence columns.
  kSequenceColumns,
  // A cell-layout program re-exported as a sequence layout with at least
  // `value` sequence columns (not offered under DFlash).
  kSequenceLayout,
  // A draft re-exported with at least `value` rows per sequence.
  kDraftSequenceRows,
};

struct Fix {
  FixKind kind;
  std::int64_t value = 0;
  std::int64_t low = 0;
  std::int64_t draft_cells = 0;
};

struct Advice {
  // Whether the configuration passes every capacity check as it is.
  bool admitted = false;
  std::vector<Fix> fixes;
  std::vector<std::string> lines;
};

// The configuration with one fix applied, at its least favorable value.
inline StartupConfig apply_fix(StartupConfig config, const Fix& fix) {
  switch (fix.kind) {
    case FixKind::kSessions:
      config.sessions = static_cast<int>(fix.value);
      break;
    case FixKind::kTokenLimit:
      config.requested_tokens = static_cast<int>(fix.value);
      break;
    case FixKind::kCells:
      config.cells = fix.value;
      if (config.draft) {
        config.draft->cells = fix.draft_cells;
      }
      break;
    case FixKind::kSequenceColumns:
      config.session_limit = static_cast<int>(fix.value);
      break;
    case FixKind::kSequenceLayout:
      config.sequence_layout = true;
      config.session_limit = static_cast<int>(fix.value);
      break;
    case FixKind::kDraftSequenceRows:
      config.draft->cells = fix.value;
      break;
  }
  return config;
}

namespace detail {

inline bool admits(const StartupConfig& config) {
  return check_startup(config).admitted;
}

inline std::string memory(std::int64_t bytes) {
  return " (" + cb::format_memory(bytes) + " of K/V)";
}

// The most resident sessions below the configured count that pass every
// check; fewer sessions only free cells and raise automatic budgets.
inline std::optional<std::int64_t> fewer_sessions(const StartupConfig& config) {
  int top = config.sessions - 1;
  if (config.session_limit > 0) {
    top = std::min(top, config.session_limit);
  }
  for (int sessions = top, tries = 0; sessions >= 1 && tries < 1 << 16;
       --sessions, ++tries) {
    StartupConfig fewer = config;
    fewer.sessions = sessions;
    if (admits(fewer)) {
      return sessions;
    }
  }
  return std::nullopt;
}

// The widest token limits the configured sessions could have: up to the
// context, the cells each session gets and the draft's budget.
inline Fix token_limit(const StartupConfig& config) {
  std::int64_t high = std::min<std::int64_t>(
      config.context_length, std::numeric_limits<int>::max());
  if (!config.sequence_layout) {
    high = std::min(high, (config.cells - 1) / config.sessions);
  }
  if (config.draft) {
    high = std::min(high, draft_session_budget(config.sessions, *config.draft));
  }
  return {FixKind::kTokenLimit, high, least_session_tokens(config.draft)};
}

} // namespace detail

// What advise() weighs for `report`, before checking any fix.
struct Candidates {
  StartupConfig config;
  StartupResult result;
  // An admitted automatic budget below the context: fixes must reach the full
  // context.
  bool full_context = false;
  std::vector<Fix> fixes;
};

// Every fix advise() considers for `report`, which CudaExecutor made for the
// target, under the DFlash `draft` pool if any; unchecked.
inline Candidates candidate_fixes(
    const cb::KvCapacityReport& report,
    const std::optional<DraftPool>& draft = std::nullopt) {
  Candidates candidates;
  const StartupConfig& config = candidates.config =
      startup_config(report, draft);
  candidates.result = check_startup(config);
  const bool automatic = config.requested_tokens == 0;
  const bool over_limit =
      config.session_limit > 0 && config.sessions > config.session_limit;
  candidates.full_context = candidates.result.admitted && automatic &&
      candidates.result.session_tokens < config.context_length;
  if (candidates.result.admitted && !candidates.full_context) {
    return candidates;
  }
  auto& fixes = candidates.fixes;
  if (!candidates.full_context) {
    if (const auto fewer = detail::fewer_sessions(config)) {
      fixes.push_back({FixKind::kSessions, *fewer});
    }
    if (!automatic) {
      fixes.push_back(detail::token_limit(config));
    }
  }
  if (!over_limit && !config.sequence_layout) {
    // Re-export cells for the requested limit, the full context, or the
    // fewest tokens a session may have.
    const std::int64_t goal = !automatic ? config.requested_tokens
        : candidates.full_context        ? config.context_length
                                         : least_session_tokens(draft);
    Fix cells{
        FixKind::kCells,
        std::max(
            config.cells,
            cb::cell_budget(config.sessions, static_cast<int>(goal), 0)
                .required_cells)};
    if (draft) {
      cells.draft_cells = std::max(
          draft->cells,
          cb::cell_budget(
              config.sessions,
              static_cast<int>(goal),
              0,
              temporary_cells(*draft))
              .required_cells);
    }
    if (cells.value != config.cells ||
        (draft && cells.draft_cells != draft->cells)) {
      fixes.push_back(cells);
    }
  }
  if (config.sequence_layout && over_limit) {
    fixes.push_back({FixKind::kSequenceColumns, config.sessions});
  }
  if (!config.sequence_layout && !draft &&
      (over_limit || report.facts.has_windowed_layers)) {
    fixes.push_back({FixKind::kSequenceLayout, config.sessions});
  }
  if (draft && draft->sequence_layout && !over_limit) {
    const std::int64_t goal = automatic
        ? cb::automatic_session_tokens(
              true, config.sessions, config.context_length, 0)
        : config.requested_tokens;
    if (goal + draft->block > draft->cells) {
      fixes.push_back({FixKind::kDraftSequenceRows, goal + draft->block});
    }
  }
  return candidates;
}

// The configurations a fix's advice lets a user choose, at its least favorable
// values: the fewest sessions and the least token limit it allows too, and
// for DFlash cells the default draft pool sized by --max-cells alone.
inline std::vector<StartupConfig> fix_choices(
    const StartupConfig& config,
    const Fix& fix) {
  std::vector<StartupConfig> choices{apply_fix(config, fix)};
  StartupConfig choice = config;
  switch (fix.kind) {
    case FixKind::kSessions:
      choice.sessions = 1;
      choices.push_back(choice);
      break;
    case FixKind::kTokenLimit:
      choice.requested_tokens = static_cast<int>(fix.low);
      choices.push_back(choice);
      break;
    case FixKind::kCells:
      if (config.draft) {
        choice.cells = std::max(fix.value, fix.draft_cells);
        choice.draft->cells = choice.cells;
        choices.push_back(choice);
      }
      break;
    default:
      break;
  }
  return choices;
}

// Whether every choice a fix allows passes every capacity check, reaching the
// full context when that is what the advice is for.
inline bool closes(const Candidates& candidates, const Fix& fix) {
  if (fix.kind == FixKind::kTokenLimit && fix.value < fix.low) {
    return false;
  }
  for (const StartupConfig& choice : fix_choices(candidates.config, fix)) {
    const StartupResult after = check_startup(choice);
    if (!after.admitted ||
        (candidates.full_context &&
         after.session_tokens != candidates.config.context_length)) {
      return false;
    }
  }
  return true;
}

// The advice for `report`, which CudaExecutor made for the target, under the
// DFlash `draft` pool if any: the facts, and each candidate fix that closes().
inline Advice advise(
    const cb::KvCapacityReport& report,
    const std::optional<DraftPool>& draft = std::nullopt) {
  const Candidates candidates = candidate_fixes(report, draft);
  const StartupConfig& config = candidates.config;
  const StartupResult& result = candidates.result;
  const bool full_context = candidates.full_context;
  Advice advice;
  advice.admitted = result.admitted;
  if (result.admitted && !full_context) {
    return advice;
  }
  for (const Fix& fix : candidates.fixes) {
    if (closes(candidates, fix)) {
      advice.fixes.push_back(fix);
    }
  }
  const bool over_limit =
      config.session_limit > 0 && config.sessions > config.session_limit;
  const std::string sessions = std::to_string(config.sessions);

  // The facts.
  const cb::KvAdmissionCheck target = cb::check_admission(
      config.sequence_layout,
      config.sessions,
      config.requested_tokens,
      config.context_length,
      config.cells,
      config.session_limit);
  if (draft && !over_limit && target.session_tokens > 0) {
    const std::int64_t budget = draft_session_budget(config.sessions, *draft);
    if (!result.admitted || budget < target.session_tokens) {
      std::string fact = draft->sequence_layout
          ? "DFlash: each draft sequence's " + std::to_string(draft->cells) +
              " rows leave " + std::to_string(budget) + " tokens beside its " +
              std::to_string(draft->block) + "-token block"
          : "DFlash: the draft pool's " + std::to_string(draft->cells) +
              " cells, less " + std::to_string(temporary_cells(*draft)) +
              " block cells and 1 padding, leave " +
              std::to_string(std::max<std::int64_t>(0, budget)) +
              " tokens for each of " + sessions + " sessions";
      fact += " (the target allows " + std::to_string(target.session_tokens) +
          "); a session needs more than the block";
      advice.lines.push_back(fact);
    }
  }
  if (full_context) {
    advice.lines.push_back(
        "Muse Glimmer: each session may use " +
        std::to_string(result.session_tokens) + " of the model's " +
        std::to_string(config.context_length) +
        " context tokens; fewer resident sessions leave more");
  }

  // The fixes in words.
  const std::string prefix = full_context
      ? "Muse Glimmer: for the full context, "
      : "Muse Glimmer: to start, ";
  for (const Fix& fix : advice.fixes) {
    switch (fix.kind) {
      case FixKind::kSessions:
        advice.lines.push_back(
            prefix + "lower " + kSessionFlags + " to total at most " +
            std::to_string(fix.value) + " resident sessions");
        if (report.reservations.capture_slots > 0) {
          advice.lines.push_back(kCaptureNote);
        }
        break;
      case FixKind::kTokenLimit:
        advice.lines.push_back(
            prefix + "set " + kTokenLimitFlag +
            (fix.low == fix.value ? " to " + std::to_string(fix.value)
                                  : " between " + std::to_string(fix.low) +
                     " and " + std::to_string(fix.value)));
        break;
      case FixKind::kCells:
        if (!draft) {
          std::string line = prefix +
              "re-export with --max-cells >= " + std::to_string(fix.value);
          if (report.facts.bytes_per_cell > 0) {
            line += detail::memory(fix.value * report.facts.bytes_per_cell);
          }
          advice.lines.push_back(line);
        } else {
          std::string line = prefix + "re-export with ";
          if (fix.value != config.cells) {
            line += "--max-cells >= " + std::to_string(fix.value) + " and ";
          }
          line += "--draft-max-cells >= " + std::to_string(fix.draft_cells);
          if (report.facts.bytes_per_cell > 0 && draft->bytes_per_cell > 0) {
            line += detail::memory(
                fix.value * report.facts.bytes_per_cell +
                fix.draft_cells * draft->bytes_per_cell);
          }
          advice.lines.push_back(line);
          advice.lines.push_back(
              "Muse Glimmer: the draft pool defaults to --max-cells, so "
              "without --draft-max-cells, --max-cells >= " +
              std::to_string(std::max(fix.value, fix.draft_cells)) +
              " sizes both");
        }
        break;
      case FixKind::kSequenceColumns:
        advice.lines.push_back(
            prefix +
            "re-export with --max-kv-seqs >= " + std::to_string(fix.value));
        break;
      case FixKind::kSequenceLayout:
        advice.lines.push_back(
            prefix + "re-export with --kv-layout sequence --max-kv-seqs >= " +
            std::to_string(fix.value) +
            (report.facts.has_windowed_layers
                 ? ", which keeps sliding-window layers in ring buffers"
                 : ""));
        break;
      case FixKind::kDraftSequenceRows:
        advice.lines.push_back(
            prefix + "re-export the draft with sequence columns of at least " +
            std::to_string(fix.value) + " rows");
        break;
    }
  }
  if (!result.admitted && advice.fixes.empty()) {
    advice.lines.push_back(
        "Muse Glimmer: no single session, token-limit or cell option admits "
        "this; see \"KV capacity and long contexts\" in BATCHING_SERVING.md");
  }
  return advice;
}

// Logs the advice: at Info for an admitted automatic budget, else as errors.
inline void log_advice(const Advice& advice) {
  for (const std::string& line : advice.lines) {
    if (advice.admitted) {
      ET_LOG(Info, "%s", line.c_str());
    } else {
      ET_LOG(Error, "%s", line.c_str());
    }
  }
}

} // namespace executorch::extension::llm::kv_advice
