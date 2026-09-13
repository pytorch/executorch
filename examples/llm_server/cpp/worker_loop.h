/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Shared model-worker generation loop + JSONL protocol for every model worker
// (for example, model-specific workers like qwen3_5_moe_worker): a worker
// constructs its engine + tokenizer and calls
// run_worker_stdio_loop(); the protocol, session routing, and decode loop live
// here once.
//
// The worker owns one LLMEngine (weights loaded once) and serves multiple
// isolated LLMSessions keyed by session_id, up to the engine's serving
// capacity; anonymous requests (no session_id) share one scratch session that
// is reset every request. Execution is synchronous: one in-flight request at a
// time.
//
// Warm resume: a named session keeps its decoded context across requests. The
// new prompt's token ids are matched against the session's resident token ids;
// on an exact prefix only the suffix is prefilled (continuing at pos>0). The
// match is exact-token (never retokenized text) and falls back to a full
// reset+prefill whenever exact reuse can't be proven, so it is always correct.
// See plan_prefill().
//
// Protocol (one JSON object per line; matches worker_client.py). stdout carries
// ONLY protocol JSON; logs go to stderr (ET_LOG):
//   worker -> stdout, once:    {"ready": true, "max_sessions": int,
//                               "max_named_sessions": int,
//                               "supports_cancel": bool}
//   client -> stdin:
//     generate: {"max_new_tokens": int, "temperature": float, "stop":
//     [str,...], "cancel_request_id"?: positive uint64,
//                "session_id"?: str, and exactly one prompt form:
//                  "prompt": str
//                  "prompt_segments": [{"text": str} | {"ids": [int,...]}]}
//     open/close/reset: {"op": "open"|"close"|"reset", "session_id": str}
//   worker -> stdout:
//     generate: {"token": str} *  (streamed), then
//               {"done": true, "prompt_tokens": int, "completion_tokens": int,
//                "finish_reason": "stop"|"length",
//                "reused_prompt_tokens": int, "prefilled_prompt_tokens": int,
//                "session_reset_reason": str
//                (new|exact_prefix|mismatch|dirty|equal),
//                "prefill_ms": float, "decode_ms": float, "total_ms": float,
//                "prefill_tok_s": float, "decode_tok_s": float,
//                "cancelled"?: true, // omitted unless cancelled out of band
//                "generated_token_ids"?: [int,...],  // omitted if not
//                resumable
//                ...optional model-specific terminal stats}
//     open/close/reset: {"opened"|"closed"|"reset": true, "session_id": str}
//     error:    {"error": str, "code"?: str}  // capacity_exhausted |
//                                              // unsupported_session

#include <nlohmann/json.hpp>

#include <executorch/examples/llm_server/cpp/worker_prefill_plan.h>
#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/llm/runner/util.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#define EXECUTORCH_LLM_WORKER_POSIX_CONTROL 1
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <climits>
#else
#define EXECUTORCH_LLM_WORKER_POSIX_CONTROL 0
#endif

namespace executorch {
namespace extension {
namespace llm {

// Emit one protocol object as a JSON line on stdout. error_handler::replace
// keeps a stray invalid UTF-8 byte (byte-level BPE) from aborting
// serialization.
inline void worker_emit(const nlohmann::json& obj) {
  std::cout << obj.dump(
                   -1, ' ', false, nlohmann::json::error_handler_t::replace)
            << "\n";
  std::cout.flush();
}

// A named session plus the warm-resume bookkeeping the worker maintains for it.
// Invariant (while not mid-mutation): resident_token_ids.size() ==
// session->position() -- the resident ids are exactly the tokens currently in
// the session's KV/recurrent state, in order.
struct WorkerSessionState {
  std::unique_ptr<LLMSession> session;
  std::vector<uint64_t> resident_token_ids;
  // Set when the resident state can no longer be trusted as an exact token
  // prefix (e.g. a stop-string trimmed the emitted text mid-token, or a
  // prefill/decode failed after mutating state). Forces a reset next request.
  bool dirty = false;
};

// Cancellation state belongs to one request. The controller thread only moves
// it from active to cancelled; the request thread seals it at terminal
// completion before constructing the terminal response.
class WorkerCancellationState {
 public:
  bool request_cancel(LLMSession& session) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (sealed_ || cancel_requested_) {
      return false;
    }
    cancel_requested_ = true;
    deliver_stop_locked(session);
    return true;
  }

  // reset()/prefill_tokens() may clear a stop flag. Defer delivery until both
  // have completed, then deliver it exactly once at the decode boundary.
  void enter_decode(LLMSession& session) {
    std::lock_guard<std::mutex> lock(mutex_);
    decode_started_ = true;
    deliver_stop_locked(session);
  }

  bool cancelled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancel_requested_;
  }

  // Returns true exactly when cancellation won the race with terminal
  // completion. Further cancellation attempts cannot change the result.
  bool seal() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (sealed_) {
      return false;
    }
    sealed_ = true;
    return cancel_requested_;
  }

 private:
  void deliver_stop_locked(LLMSession& session) {
    if (cancel_requested_ && decode_started_ && !stop_delivered_ && !sealed_) {
      stop_delivered_ = true;
      session.stop();
    }
  }

  mutable std::mutex mutex_;
  bool cancel_requested_ = false;
  bool decode_started_ = false;
  bool stop_delivered_ = false;
  bool sealed_ = false;
};

// Reads fixed-width little-endian cancellation IDs from the inherited control
// descriptor. Parser state is bounded to one partial frame and one pending ID.
class WorkerCancellationController {
 public:
  static constexpr const char* kControlFdEnv =
      "EXECUTORCH_LLM_WORKER_CONTROL_FD";

  class ActiveRequest {
   public:
    ActiveRequest() = default;
    ActiveRequest(
        WorkerCancellationController* controller,
        WorkerSessionState* worker_state,
        WorkerCancellationState* cancellation_state)
        : controller_(controller),
          worker_state_(worker_state),
          cancellation_state_(cancellation_state) {}
    ActiveRequest(const ActiveRequest&) = delete;
    ActiveRequest& operator=(const ActiveRequest&) = delete;
    ActiveRequest(ActiveRequest&& other) noexcept
        : controller_(std::exchange(other.controller_, nullptr)),
          worker_state_(std::exchange(other.worker_state_, nullptr)),
          cancellation_state_(
              std::exchange(other.cancellation_state_, nullptr)) {}
    ActiveRequest& operator=(ActiveRequest&& other) noexcept {
      if (this != &other) {
        reset();
        controller_ = std::exchange(other.controller_, nullptr);
        worker_state_ = std::exchange(other.worker_state_, nullptr);
        cancellation_state_ = std::exchange(other.cancellation_state_, nullptr);
      }
      return *this;
    }
    ~ActiveRequest() {
      reset();
    }

    void reset() {
      if (controller_ != nullptr) {
        controller_->detach(worker_state_, cancellation_state_);
        controller_ = nullptr;
        worker_state_ = nullptr;
        cancellation_state_ = nullptr;
      }
    }

   private:
    WorkerCancellationController* controller_ = nullptr;
    WorkerSessionState* worker_state_ = nullptr;
    WorkerCancellationState* cancellation_state_ = nullptr;
  };

  WorkerCancellationController()
      : WorkerCancellationController(control_fd_from_environment()) {}

  explicit WorkerCancellationController(int control_fd) {
#if EXECUTORCH_LLM_WORKER_POSIX_CONTROL
    if (control_fd < 0) {
      return;
    }
    struct stat descriptor_stat {};
    const int status_flags = ::fcntl(control_fd, F_GETFL, 0);
    const int descriptor_flags = ::fcntl(control_fd, F_GETFD, 0);
    if (control_fd <= STDERR_FILENO ||
        ::fstat(control_fd, &descriptor_stat) < 0 ||
        !S_ISFIFO(descriptor_stat.st_mode) || status_flags < 0 ||
        descriptor_flags < 0 || (status_flags & O_ACCMODE) == O_WRONLY ||
        ::fcntl(control_fd, F_SETFL, status_flags | O_NONBLOCK) < 0 ||
        ::fcntl(control_fd, F_SETFD, descriptor_flags | FD_CLOEXEC) < 0) {
      ::close(control_fd);
      return;
    }
    control_fd_ = control_fd;
    // Consume frames queued before worker startup deterministically. The reader
    // is nonblocking, so this cannot delay readiness.
    if (!read_available()) {
      ::close(control_fd_);
      control_fd_ = -1;
      return;
    }
    try {
      thread_ = std::thread([this]() { run(); });
    } catch (...) {
      ::close(control_fd_);
      control_fd_ = -1;
    }
#else
    (void)control_fd;
#endif
  }

  WorkerCancellationController(const WorkerCancellationController&) = delete;
  WorkerCancellationController& operator=(const WorkerCancellationController&) =
      delete;

  ~WorkerCancellationController() {
    shutdown();
  }

  bool supported() const {
    return control_fd_ >= 0;
  }

  uint64_t processed_frame_count_for_testing() const {
    return processed_frame_count_.load(std::memory_order_acquire);
  }

  // Reject a stale/duplicate request ID before its session is selected. Control
  // frames with such IDs are still ignored silently by handle_frame().
  void validate_request_id(uint64_t request_id) {
    if (!supported() || request_id == 0) {
      throw std::runtime_error("invalid cancel_request_id");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (request_id <= last_completed_request_id_) {
      throw std::runtime_error("cancel_request_id is stale or duplicate");
    }
  }

  ActiveRequest activate(
      uint64_t request_id,
      WorkerSessionState& worker_state,
      WorkerCancellationState& cancellation_state) {
    if (!supported() || request_id == 0) {
      return {};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_ != nullptr) {
      throw std::runtime_error("worker cancellation request already active");
    }
    if (request_id <= last_completed_request_id_) {
      throw std::runtime_error("cancel_request_id is stale or duplicate");
    }

    active_request_id_ = request_id;
    active_session_ = worker_state.session.get();
    active_cancellation_state_ = &cancellation_state;
    if (pending_request_id_.has_value()) {
      const bool matches = *pending_request_id_ == request_id;
      pending_request_id_.reset();
      if (matches) {
        cancel_active_locked();
      }
    }
    return ActiveRequest(this, &worker_state, &cancellation_state);
  }

  void shutdown() {
    stopping_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
      thread_.join();
    }
#if EXECUTORCH_LLM_WORKER_POSIX_CONTROL
    if (control_fd_ >= 0) {
      ::close(control_fd_);
    }
#endif
    control_fd_ = -1;
  }

 private:
  static int control_fd_from_environment() {
#if EXECUTORCH_LLM_WORKER_POSIX_CONTROL
    const char* raw = std::getenv(kControlFdEnv);
    if (raw == nullptr || *raw == '\0') {
      return -1;
    }
    for (const char* cursor = raw; *cursor != '\0'; ++cursor) {
      if (*cursor < '0' || *cursor > '9') {
        ::unsetenv(kControlFdEnv);
        return -1;
      }
    }
    errno = 0;
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    const bool valid = errno == 0 && end != raw && *end == '\0' && value >= 0 &&
        value <= INT_MAX;
    ::unsetenv(kControlFdEnv);
    return valid ? static_cast<int>(value) : -1;
#else
    return -1;
#endif
  }

  void detach(
      WorkerSessionState* worker_state,
      WorkerCancellationState* cancellation_state) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_cancellation_state_ != cancellation_state) {
      return;
    }
    // Seal on exceptional exits too. Only the request thread mutates dirty.
    if (cancellation_state->seal()) {
      worker_state->dirty = true;
    }
    last_completed_request_id_ =
        std::max(last_completed_request_id_, active_request_id_);
    if (pending_request_id_.has_value() &&
        *pending_request_id_ <= last_completed_request_id_) {
      pending_request_id_.reset();
    }
    active_request_id_ = 0;
    active_session_ = nullptr;
    active_cancellation_state_ = nullptr;
  }

  void cancel_active_locked() {
    if (active_session_ != nullptr && active_cancellation_state_ != nullptr) {
      // The request state defers delivery during prefill and guarantees exactly
      // one stop() call after decode begins.
      active_cancellation_state_->request_cancel(*active_session_);
    }
  }

  void handle_frame(uint64_t request_id) {
    if (request_id == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_session_ != nullptr) {
      if (request_id == active_request_id_) {
        cancel_active_locked();
      }
      return; // conflicting active IDs are never queued for a later request
    }
    if (request_id <= last_completed_request_id_) {
      return;
    }
    if (!pending_request_id_.has_value()) {
      pending_request_id_ = request_id;
    }
    // A duplicate or conflicting preactivation frame is ignored. The first
    // complete, non-stale ID owns the single bounded pending slot.
  }

#if EXECUTORCH_LLM_WORKER_POSIX_CONTROL
  void consume_bytes(const uint8_t* bytes, size_t count) {
    for (size_t index = 0; index < count; ++index) {
      if (partial_frame_size_ < partial_frame_.size()) {
        partial_frame_[partial_frame_size_++] = bytes[index];
        continue;
      }
      uint64_t request_id = static_cast<uint64_t>(bytes[index]) << 56;
      for (size_t byte = 0; byte < partial_frame_.size(); ++byte) {
        request_id |= static_cast<uint64_t>(partial_frame_[byte]) << (byte * 8);
      }
      partial_frame_size_ = 0;
      handle_frame(request_id);
      processed_frame_count_.fetch_add(1, std::memory_order_release);
    }
  }

  bool read_available() {
    std::array<uint8_t, 64> bytes{};
    while (true) {
      const ssize_t count = ::read(control_fd_, bytes.data(), bytes.size());
      if (count > 0) {
        consume_bytes(bytes.data(), static_cast<size_t>(count));
        return true;
      }
      if (count == 0) {
        return false;
      }
      if (errno == EINTR) {
        continue;
      }
      return errno == EAGAIN || errno == EWOULDBLOCK;
    }
  }

  void run() {
    while (!stopping_.load(std::memory_order_acquire)) {
      pollfd descriptor{control_fd_, POLLIN, 0};
      const int poll_result = ::poll(&descriptor, 1, 50);
      if (poll_result < 0) {
        if (errno == EINTR) {
          continue;
        }
        break;
      }
      if (poll_result == 0) {
        continue;
      }
      if ((descriptor.revents & POLLIN) != 0) {
        if (!read_available()) {
          break;
        }
        continue;
      }
      if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        break;
      }
    }
  }
#endif

  int control_fd_ = -1;
  std::atomic<bool> stopping_{false};
  std::atomic<uint64_t> processed_frame_count_{0};
  std::thread thread_;
  std::mutex mutex_;
  uint64_t active_request_id_ = 0;
  uint64_t last_completed_request_id_ = 0;
  std::optional<uint64_t> pending_request_id_;
  std::array<uint8_t, sizeof(uint64_t) - 1> partial_frame_{};
  size_t partial_frame_size_ = 0;
  LLMSession* active_session_ = nullptr;
  WorkerCancellationState* active_cancellation_state_ = nullptr;
};

// Strictly validate the optional protocol field. Legacy requests remain valid;
// a cancellation ID is accepted only when the worker advertised the capability.
inline std::optional<uint64_t> worker_cancel_request_id(
    const nlohmann::json& request,
    bool supports_cancel) {
  if (!request.contains("cancel_request_id")) {
    return std::nullopt;
  }
  if (!supports_cancel) {
    throw std::runtime_error("cancel_request_id is not supported");
  }
  const auto& value = request.at("cancel_request_id");
  uint64_t request_id = 0;
  if (value.is_number_unsigned()) {
    request_id = value.get<uint64_t>();
  } else if (value.is_number_integer()) {
    const int64_t signed_id = value.get<int64_t>();
    if (signed_id > 0) {
      request_id = static_cast<uint64_t>(signed_id);
    }
  } else {
    throw std::runtime_error("cancel_request_id must be a positive uint64");
  }
  if (request_id == 0) {
    throw std::runtime_error("cancel_request_id must be a positive uint64");
  }
  return request_id;
}

// One generation request against a session. Encodes the prompt, chooses a
// prefill plan (warm suffix reuse for named sessions, or a full reset+prefill),
// then streams complete-UTF-8 text pieces from decode_one(). A terminal step
// (EOS or cooperative stop) ends generation and is not emitted or counted.
// Maintains st.resident_token_ids / st.dirty. Throws std::runtime_error on
// failure; the caller reports it as {"error": ...}.
inline void worker_handle_request(
    WorkerSessionState& st,
    bool warm,
    ::tokenizers::Tokenizer& tokenizer,
    const std::unordered_map<std::string, int64_t>& metadata,
    const nlohmann::json& req,
    const std::vector<uint64_t>& prompt_prefix_ids = {},
    const nlohmann::json& additional_terminal_stats = nlohmann::json::object(),
    WorkerCancellationState* cancellation = nullptr) {
  if (!additional_terminal_stats.is_object()) {
    throw std::runtime_error("additional terminal stats must be a JSON object");
  }
  static const std::vector<std::string> kReservedTerminalKeys = {
      "done",
      "prompt_tokens",
      "completion_tokens",
      "finish_reason",
      "reused_prompt_tokens",
      "prefilled_prompt_tokens",
      "session_reset_reason",
      "cancelled",
      "generated_token_ids",
      "prefill_ms",
      "decode_ms",
      "total_ms",
      "prefill_tok_s",
      "decode_tok_s"};
  for (const auto& [key, value] : additional_terminal_stats.items()) {
    (void)value;
    if (std::find(
            kReservedTerminalKeys.begin(), kReservedTerminalKeys.end(), key) !=
        kReservedTerminalKeys.end()) {
      throw std::runtime_error(
          "additional terminal stat collides with reserved key: " + key);
    }
  }

  const auto request_start = std::chrono::steady_clock::now();
  LLMSession& session = *st.session;
  int64_t max_new = req.value("max_new_tokens", static_cast<int64_t>(-1));
  const float temperature = req.value("temperature", 0.0f);
  const double top_p_value = req.value("top_p", 1.0);
  const int64_t top_k_value = req.value("top_k", static_cast<int64_t>(0));
  const int64_t seed_value = req.value("seed", static_cast<int64_t>(0));
  if (!std::isfinite(top_p_value) || top_p_value <= 0.0 || top_p_value > 1.0) {
    throw std::runtime_error("top_p must be finite and in (0, 1]");
  }
  if (top_k_value < 0 || top_k_value > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error("top_k must fit in a nonnegative int32");
  }
  // seed == 0 means "unset" (SamplingConfig::seed == 0 -> worker picks a random
  // seed), so 0 is intentionally accepted here even though the HTTP layer
  // treats 0 as the omitted sentinel and rejects an explicit seed=0. A JSON
  // seed >= 2^63 won't fit int64_t and is rejected at parse time above; the
  // HTTP layer caps the same range at 2^63 - 1 with a structured error.
  if (seed_value < 0) {
    throw std::runtime_error("seed must be nonnegative");
  }
  // Stop strings (the request's `stop` sequences): terminate at the token
  // boundary where one appears so we don't generate to EOS/max_new past it. The
  // control plane also enforces these as a backstop.
  const std::vector<std::string> stops =
      req.value("stop", std::vector<std::string>{});

  // The prompt is either a single rendered string ("prompt") or an ordered list
  // of segments ("prompt_segments"), each a {"text": ...} chunk to tokenize or
  // a
  // {"ids": [...]} run of literal token ids. Segments let the control plane
  // splice the exact generated token ids of prior assistant turns back in,
  // instead of re-tokenizing the chat template's lossy re-rendering of them (so
  // warm resume can hit on tool-use turns). Text is encoded with no special
  // tokens (already rendered), matching the runner's own encode path.
  const bool has_prompt = req.contains("prompt");
  const bool has_segments = req.contains("prompt_segments");
  if (has_prompt == has_segments) {
    throw std::runtime_error(
        "exactly one of prompt / prompt_segments is required");
  }
  std::vector<uint64_t> ids = prompt_prefix_ids;
  auto encode_text = [&](const std::string& text) {
    auto enc = tokenizer.encode(text, /*bos=*/0, /*eos=*/0);
    if (!enc.ok()) {
      throw std::runtime_error("prompt encode failed");
    }
    ids.insert(ids.end(), enc->begin(), enc->end());
  };
  if (has_segments) {
    for (const auto& seg : req.at("prompt_segments")) {
      if (seg.contains("ids")) {
        const auto& raw_ids = seg.at("ids");
        std::transform(
            raw_ids.begin(),
            raw_ids.end(),
            std::back_inserter(ids),
            [](const auto& id) { return id.template get<uint64_t>(); });
      } else if (seg.contains("text")) {
        encode_text(seg.at("text").get<std::string>());
      } else {
        throw std::runtime_error("prompt_segment needs `text` or `ids`");
      }
    }
  } else {
    encode_text(req.at("prompt").get<std::string>());
  }
  if (ids.empty()) {
    throw std::runtime_error("empty prompt");
  }
  const int64_t num_prompt = static_cast<int64_t>(ids.size());

  // Bound generation to the context window: default to filling the remaining
  // room, and clamp an explicit max_new_tokens too, so decode never steps past
  // the window (which would error mid-generation after partial output). The
  // bound is on the FULL prompt length (= pos after prefill), regardless of how
  // much is reused.
  const auto ctx_it = metadata.find(kMaxContextLen);
  if (ctx_it != metadata.end()) {
    const int64_t room = ctx_it->second - num_prompt;
    if (room <= 0) {
      throw std::runtime_error(
          "prompt fills the context window; no room to generate");
    }
    if (max_new <= 0 || max_new > room) {
      max_new = room;
    }
  } else if (max_new <= 0) {
    max_new = 2048;
  }

  // Decide full vs warm-suffix prefill. Anonymous (scratch) and warm-disabled
  // sessions always full-prefill from a clean state.
  PrefillPlan plan = warm ? plan_prefill(st.resident_token_ids, ids, st.dirty)
                          : PrefillPlan{PrefillPlan::kFull, 0, "new"};
  int64_t reused = 0;
  std::vector<uint64_t> to_prefill;
  if (plan.action == PrefillPlan::kSuffix) {
    reused = static_cast<int64_t>(plan.suffix_start);
    to_prefill.assign(ids.begin() + plan.suffix_start, ids.end());
  } else {
    if (session.reset() != ::executorch::runtime::Error::Ok) {
      st.dirty = true;
      throw std::runtime_error("session reset failed");
    }
    st.resident_token_ids.clear();
    st.dirty = false;
    to_prefill = ids;
  }
  const int64_t prefilled = static_cast<int64_t>(to_prefill.size());

  SamplingConfig sampling;
  sampling.temperature = temperature;
  sampling.top_p = static_cast<float>(top_p_value);
  sampling.top_k = static_cast<int32_t>(top_k_value);
  sampling.seed = static_cast<uint64_t>(seed_value);
  const auto prefill_start = std::chrono::steady_clock::now();
  if (session.prefill_tokens(to_prefill, &sampling) !=
      ::executorch::runtime::Error::Ok) {
    st.dirty = true; // state may be partially mutated; force a reset next time
    throw std::runtime_error("prefill failed");
  }
  // The resident state now equals the full prompt (resident prefix + prefilled
  // suffix, or the whole prompt). Keep the invariant
  // resident.size()==position().
  st.resident_token_ids = ids;
  // reset()/prefill_tokens() may clear a stop requested before or during
  // prefill. Entering decode delivers a latched request exactly once; later
  // cancellation is delivered directly by the controller thread.
  if (cancellation != nullptr) {
    cancellation->enter_decode(session);
  }
  const auto decode_start = std::chrono::steady_clock::now();

  std::string buf; // bytes not yet forming a complete UTF-8 prefix
  std::string pending; // complete-UTF-8 text held back for stop-string matching
  int64_t num_generated = 0;
  std::string finish = "length"; // EOS or stop string -> "stop"
  bool stop_string = false; // a request stop string was matched
  bool cancelled = false;
  bool cancellation_sealed = false;
  auto seal_cancellation = [&]() {
    if (!cancellation_sealed) {
      cancelled = cancellation != nullptr && cancellation->seal();
      cancellation_sealed = true;
      if (cancelled) {
        // Seal can precede fallible token bookkeeping/output on a terminal
        // iteration. Mark dirty immediately so exceptions cannot expose a
        // partially mutated session as warm-resumable.
        st.dirty = true;
      }
    }
  };
  for (int64_t step = 0; step < max_new; ++step) {
    const bool reached_length = step + 1 == max_new;
    auto step_result = session.decode_one(sampling);
    if (step_result.error() != ::executorch::runtime::Error::Ok) {
      st.dirty = true;
      throw std::runtime_error("decode failed");
    }
    const auto& d = step_result.get();
    if (d.is_terminal) {
      seal_cancellation();
      finish = "stop";
      // Terminal step (EOS / cooperative stop): the terminal token is neither
      // emitted as text nor counted in num_generated -> completion_tokens. This
      // is intentional -- completion_tokens reflects the visible completion the
      // client received, not internal forward steps; an EOS the user never sees
      // is not part of that count.
      break;
    }
    if (reached_length) {
      seal_cancellation();
    }
    // The token was forwarded into the cache (pos advanced); track it so the
    // resident-ids/position invariant holds. EOS/terminal tokens are not
    // forwarded, so they are not appended (above).
    st.resident_token_ids.push_back(d.token_id);
    ++num_generated;
    buf += d.text_piece;
    const size_t cut = utf8_complete_prefix_len(buf);
    if (cut > 0) {
      pending += buf.substr(0, cut);
      buf.erase(0, cut);
    }
    bool stop_hit = false;
    const size_t safe = stop_safe_prefix_len(pending, stops, stop_hit);
    if (stop_hit) {
      // The request is terminal as soon as the stop sequence is recognized.
      // Seal before worker_emit(), which may block while flushing stdout.
      seal_cancellation();
    }
    if (safe > 0) {
      worker_emit({{"token", pending.substr(0, safe)}});
      pending.erase(0, safe);
    }
    if (stop_hit) {
      finish = "stop"; // reached a stop string: drop it and everything after
      stop_string = true;
      // Trimming at the stop means the next turn's prompt won't be an exact
      // token extension of resident, so force a reset (no false prefix match).
      //
      // CONTRACT: every *string* stop is non-resumable this way (trim + dirty +
      // omit generated_token_ids) -- right for user/request and content-cleanup
      // stops, which change visible text. A clean turn terminator stays
      // warm-resumable only if the engine surfaces it as a terminal/EOS token
      // id (handled above via d.is_terminal; e.g. Qwen adds <|im_end|> to
      // eos_ids).
      st.dirty = true;
      break;
    }
  }
  // Seal at the terminal decision, before formatting or emitting the terminal
  // event. A late frame can no longer relabel or dirty this request.
  seal_cancellation();
  if (cancelled) {
    finish = "stop";
    st.dirty = true;
  }
  if (!stop_string) {
    // EOS, length, or cancellation: flush held-back text + any trailing bytes
    // (replaced if invalid). A stop-string hit drops the remainder instead.
    pending += buf;
    if (!pending.empty()) {
      worker_emit({{"token", pending}});
    }
  }
  // finish_reason: "stop" if the model emitted EOS, hit a stop string, or was
  // cancelled; otherwise "length" after max_new (possibly context-clamped).
  // reused/prefilled sum to prompt_tokens; session_reset_reason explains the
  // prefill plan (for measuring warm-resume hit rate).
  nlohmann::json done = {
      {"done", true},
      {"prompt_tokens", num_prompt},
      {"completion_tokens", num_generated},
      {"finish_reason", finish},
      {"reused_prompt_tokens", reused},
      {"prefilled_prompt_tokens", prefilled},
      {"session_reset_reason", plan.reason}};
  if (cancelled) {
    done["cancelled"] = true;
  }
  // generated_token_ids = the (non-terminal) tokens made resident this turn,
  // for the control plane to splice back as an `ids` segment. Only emit them
  // when they faithfully decode to the emitted text: a stop-string trim kept
  // the post-stop tokens resident but dropped them from the output, so splicing
  // them would inject text the client never saw. Omitting them makes the
  // control plane record this turn as not resumable (falls back to a text
  // re-render).
  if (!stop_string && !cancelled) {
    done["generated_token_ids"] = std::vector<uint64_t>(
        st.resident_token_ids.end() - num_generated,
        st.resident_token_ids.end());
  }
  const auto request_end = std::chrono::steady_clock::now();
  const double prefill_ms =
      std::chrono::duration<double, std::milli>(decode_start - prefill_start)
          .count();
  const double decode_ms =
      std::chrono::duration<double, std::milli>(request_end - decode_start)
          .count();
  const double total_ms =
      std::chrono::duration<double, std::milli>(request_end - request_start)
          .count();
  done["prefill_ms"] = prefill_ms;
  done["decode_ms"] = decode_ms;
  done["total_ms"] = total_ms;
  done["prefill_tok_s"] = prefill_ms > 0.0
      ? (static_cast<double>(prefilled) * 1000.0 / prefill_ms)
      : 0.0;
  done["decode_tok_s"] = decode_ms > 0.0
      ? (static_cast<double>(num_generated) * 1000.0 / decode_ms)
      : 0.0;
  done.update(additional_terminal_stats);
  worker_emit(done);
}

// Owns the engine's sessions for one worker: named sessions keyed by id plus a
// single scratch session for anonymous requests. Single-threaded (driven by the
// stdio loop), so no internal locking.
class WorkerSessions {
 public:
  explicit WorkerSessions(LLMEngine& engine)
      : engine_(engine),
        // Reserve one capacity slot for the scratch (anonymous) session when
        // the backend can host more than one; a single-session backend hosts
        // only the scratch and reports 0 named sessions.
        max_named_(std::max(
            0,
            engine.serving_capacity()
                    .max_physical_sessions_without_weight_duplication -
                1)) {}

  int32_t max_named() const {
    return max_named_;
  }

  // Resolve (and admit, creating on first use) a named session. Returns nullptr
  // and sets code on failure: "unsupported_session" when the backend hosts no
  // named sessions, "capacity_exhausted" when all named slots are taken.
  WorkerSessionState* open_named(const std::string& id, std::string& code) {
    auto it = named_.find(id);
    if (it != named_.end()) {
      return &it->second; // idempotent open / reuse across requests
    }
    if (max_named_ == 0) {
      code = "unsupported_session";
      return nullptr;
    }
    if (static_cast<int32_t>(named_.size()) >= max_named_) {
      code = "capacity_exhausted";
      return nullptr;
    }
    auto result = engine_.create_session();
    if (result.error() != ::executorch::runtime::Error::Ok) {
      code = "capacity_exhausted"; // engine-side capacity backstop
      return nullptr;
    }
    WorkerSessionState& st = named_[id];
    st.session = std::move(result.get());
    return &st;
  }

  // Destroy a named session (freeing its per-session state); idempotent.
  void close_named(const std::string& id) {
    named_.erase(id);
  }

  // Clear a named session's context (reset KV/recurrent + resident ids) while
  // keeping its capacity slot allocated. No-op if the session doesn't exist.
  // Returns Ok (including the absent no-op); on a failed reset returns the
  // session's error and leaves resident state intact, so the control plane
  // keeps its transcript in lockstep instead of clearing it after a failed
  // reset.
  ::executorch::runtime::Error reset_named(const std::string& id) {
    auto it = named_.find(id);
    if (it == named_.end()) {
      return ::executorch::runtime::Error::Ok;
    }
    auto err = it->second.session->reset();
    if (err != ::executorch::runtime::Error::Ok) {
      return err;
    }
    it->second.resident_token_ids.clear();
    it->second.dirty = false;
    return ::executorch::runtime::Error::Ok;
  }

  // The scratch session for anonymous requests, created on first use. Throws if
  // the engine cannot create it.
  WorkerSessionState* scratch() {
    if (!scratch_.session) {
      auto result = engine_.create_session();
      if (result.error() != ::executorch::runtime::Error::Ok) {
        throw std::runtime_error("failed to create scratch session");
      }
      scratch_.session = std::move(result.get());
    }
    return &scratch_;
  }

 private:
  LLMEngine& engine_;
  int32_t max_named_;
  std::unordered_map<std::string, WorkerSessionState> named_;
  WorkerSessionState scratch_;
};

// Emit {"ready": true, ...}, then read JSONL requests from stdin and dispatch
// each (generate / open / close / reset), reporting exceptions as
// {"error": ...} and continuing to serve. Returns 0 when stdin closes.
// enable_warm_resume gates warm suffix reuse for named sessions (off -> every
// request resets and re-prefills; useful for A/B measurement).
inline int run_worker_stdio_loop(
    LLMEngine& engine,
    ::tokenizers::Tokenizer& tokenizer,
    const std::unordered_map<std::string, int64_t>& metadata,
    bool enable_warm_resume = true,
    const std::vector<uint64_t>& prompt_prefix_ids = {}) {
  WorkerSessions sessions(engine);
  WorkerCancellationController cancellation_controller;
  worker_emit(
      {{"ready", true},
       {"max_sessions",
        engine.serving_capacity()
            .max_physical_sessions_without_weight_duplication},
       {"max_named_sessions", sessions.max_named()},
       {"supports_cancel", cancellation_controller.supported()}});

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    try {
      const nlohmann::json req = nlohmann::json::parse(line);
      const std::string op = req.value("op", std::string{});

      if (op == "open" || op == "close" || op == "reset") {
        const std::string id = req.at("session_id").get<std::string>();
        if (id.empty()) {
          throw std::runtime_error("session_id required for op");
        }
        if (op == "close") {
          sessions.close_named(id);
          worker_emit({{"closed", true}, {"session_id", id}});
        } else if (op == "reset") {
          // idempotent (no-op if absent); only acks success if the reset took
          if (sessions.reset_named(id) != ::executorch::runtime::Error::Ok) {
            worker_emit(
                {{"error", "session reset failed"}, {"session_id", id}});
          } else {
            worker_emit({{"reset", true}, {"session_id", id}});
          }
        } else { // open
          std::string code;
          if (sessions.open_named(id, code) == nullptr) {
            worker_emit(
                {{"error", "cannot open session"},
                 {"code", code},
                 {"session_id", id}});
          } else {
            worker_emit({{"opened", true}, {"session_id", id}});
          }
        }
        continue;
      }

      // Generation. A session_id routes to its named session (admitted on first
      // use, warm-resumable); its absence uses the shared scratch session,
      // which is always reset per request.
      const auto cancel_request_id =
          worker_cancel_request_id(req, cancellation_controller.supported());
      if (cancel_request_id.has_value()) {
        cancellation_controller.validate_request_id(*cancel_request_id);
      }
      const std::string id = req.value("session_id", std::string{});
      WorkerSessionState* st = nullptr;
      bool warm = false;
      if (id.empty()) {
        st = sessions.scratch();
      } else {
        std::string code;
        st = sessions.open_named(id, code);
        if (st == nullptr) {
          worker_emit(
              {{"error", "cannot open session"},
               {"code", code},
               {"session_id", id}});
          continue;
        }
        warm = enable_warm_resume;
      }
      WorkerCancellationState cancellation;
      WorkerCancellationController::ActiveRequest active_request;
      WorkerCancellationState* cancellation_ptr = nullptr;
      if (cancel_request_id.has_value()) {
        active_request = cancellation_controller.activate(
            *cancel_request_id, *st, cancellation);
        cancellation_ptr = &cancellation;
      }
      worker_handle_request(
          *st,
          warm,
          tokenizer,
          metadata,
          req,
          prompt_prefix_ids,
          nlohmann::json::object(),
          cancellation_ptr);
    } catch (const std::exception& e) { // report and keep serving
      worker_emit({{"error", std::string(e.what())}});
    }
  }
  return 0;
}

} // namespace llm
} // namespace extension
} // namespace executorch

#undef EXECUTORCH_LLM_WORKER_POSIX_CONTROL
