/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/runner.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

namespace {

// Everything the runner needs from one answered input, checked together so a
// broken executor is reported the same way whether or not this step ends the
// generation. `room` is how far the session can still advance, so a run that
// would carry it past Position is caught here rather than wrapping later.
bool valid_output(const Output& output, SessionId expected, std::size_t room) {
  return output.sid == expected && !output.tokens.empty() &&
      output.tokens.size() <= room;
}

// Whether the executor answered the batch it was given. A false return and a
// missing answer both condemn the whole batch, unlike a single malformed
// Output, which condemns only its own generation.
bool batch_answered(bool ok, const BatchInput& batch, const BatchOutput& out) {
  return ok && out.outputs.size() == batch.inputs.size();
}

template <class... Visitors>
struct Overloaded : Visitors... {
  using Visitors::operator()...;
};

template <class... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

} // namespace

// Shared between the runner and every handle to one generation. Out of the
// public header so that callers see neither the lock nor what it guards.
enum class CompletionPhase { Pending, DeliveringCallback, Done };

struct GenerationHandleState {
  std::mutex mutex;
  std::condition_variable cv;
  CompletionPhase phase = CompletionPhase::Pending;
  std::optional<FinishReason> reason;
  std::string error_message;
  GenerationMetrics metrics;
  std::atomic<bool> cancelled{false};
};

struct SessionStatus {
  std::atomic<bool> open{true};
  // The session's executor-committed position. It lives here rather than on
  // the engine thread's SessionRecord so that Session::position() can read it
  // without touching sessions_, which only the engine thread may. The engine
  // thread is the sole writer.
  std::atomic<Position> position{0};
};

struct TerminalOutcome {
  FinishReason reason;
  std::vector<Token> tokens;
  std::string error_message;

  static TerminalOutcome stopped(std::vector<Token> tokens = {}) {
    return {FinishReason::StopToken, std::move(tokens), {}};
  }

  static TerminalOutcome limit_reached(std::vector<Token> tokens = {}) {
    return {FinishReason::NewTokenLimit, std::move(tokens), {}};
  }

  static TerminalOutcome cancelled() {
    return {FinishReason::Cancelled, {}, {}};
  }

  static TerminalOutcome failed(std::string error_message) {
    return {FinishReason::Failed, {}, std::move(error_message)};
  }
};

class TerminalCompletion {
 public:
  static std::optional<TerminalCompletion> try_claim(
      const std::shared_ptr<GenerationHandleState>& state) {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->phase != CompletionPhase::Pending) {
      return std::nullopt;
    }
    state->phase = CompletionPhase::DeliveringCallback;
    return TerminalCompletion(state);
  }

  TerminalCompletion(TerminalCompletion&& other) noexcept
      : state_(std::move(other.state_)) {}
  TerminalCompletion& operator=(TerminalCompletion&&) = delete;
  TerminalCompletion(const TerminalCompletion&) = delete;
  TerminalCompletion& operator=(const TerminalCompletion&) = delete;

  ~TerminalCompletion() {
    if (state_) {
      finish(TerminalOutcome::failed({}), GenerationMetrics{});
    }
  }

  // Metrics ride alongside the outcome rather than inside it: they describe
  // the generation, not the reason it ended. Published here, with the reason
  // and under the same lock, so a handle never shows one without the other.
  void finish(TerminalOutcome outcome, const GenerationMetrics& metrics) {
    assert(state_);
    auto state = std::move(state_);
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      assert(state->phase == CompletionPhase::DeliveringCallback);
      state->reason = outcome.reason;
      state->error_message = std::move(outcome.error_message);
      state->metrics = metrics;
      state->phase = CompletionPhase::Done;
    }
    state->cv.notify_all();
  }

 private:
  explicit TerminalCompletion(std::shared_ptr<GenerationHandleState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<GenerationHandleState> state_;
};

struct CallbackResult {
  bool succeeded;
  std::string error_message;
};

CallbackResult invoke_callback(
    const GenerationCallback& on_update,
    GenerationUpdate update) {
  if (!on_update) {
    return {true, {}};
  }
#if ET_HAS_EXCEPTIONS
  try {
    on_update(update);
  } catch (const std::exception& error) {
    return {false, error.what()};
  } catch (...) {
    return {false, "generation callback threw a non-standard exception"};
  }
#else
  on_update(update);
#endif
  return {true, {}};
}

void finalize_terminal(
    const std::shared_ptr<GenerationHandleState>& state,
    const GenerationCallback& on_update,
    TerminalOutcome outcome) {
  auto completion = TerminalCompletion::try_claim(state);
  if (!completion) {
    return;
  }
  auto callback_result = invoke_callback(
      on_update,
      GenerationUpdate{
          std::move(outcome.tokens), outcome.reason, outcome.error_message});
  if (!callback_result.succeeded) {
    outcome = TerminalOutcome::failed(std::move(callback_result.error_message));
  }
  // Rejected before admission, so there is no timeline to report.
  completion->finish(std::move(outcome), GenerationMetrics{});
}

// --- GenerationHandle ---

// --- GenerationHandle ------------------------------------------------------

bool GenerationHandle::valid() const noexcept {
  return state_ != nullptr;
}

void GenerationHandle::cancel() const {
  if (state_) {
    state_->cancelled.store(true);
  }
}

bool GenerationHandle::done() const {
  if (!state_) {
    return false;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->phase == CompletionPhase::Done;
}

void GenerationHandle::wait() const {
  if (!state_) {
    return;
  }
  std::unique_lock<std::mutex> lock(state_->mutex);
  state_->cv.wait(
      lock, [this] { return state_->phase == CompletionPhase::Done; });
}

std::optional<FinishReason> GenerationHandle::finish_reason() const {
  if (!state_) {
    return std::nullopt;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  if (state_->phase != CompletionPhase::Done) {
    return std::nullopt;
  }
  assert(state_->reason.has_value());
  return state_->reason;
}

std::string GenerationHandle::error_message() const {
  if (!state_) {
    return {};
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->error_message;
}

// Published with the reason, which lands only after the terminal callback
// returns. Reading this from inside that callback yields an empty snapshot,
// exactly as finish_reason() yields nullopt there.
GenerationMetrics GenerationHandle::metrics() const {
  if (!state_) {
    return {};
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->metrics;
}

// Everything the runner owns. Held by shared_ptr from both Runner and every
// Session, so a Session outliving its Runner finds a stopped object rather
// than a dangling one.
class RunnerImpl : public std::enable_shared_from_this<RunnerImpl> {
 public:
  RunnerImpl(Executor& executor, std::unique_ptr<Scheduler> scheduler)
      : executor_(executor), scheduler_(std::move(scheduler)) {
    assert(scheduler_ != nullptr && "Runner requires a scheduler");
  }

  ~RunnerImpl() {
    shutdown();
  }

  // Separate from the constructor: the engine thread hands out Sessions, which
  // needs shared_from_this, which is not available until construction ends.
  void start() {
    std::lock_guard<std::mutex> lock(control_mutex_);
    assert(
        lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Running &&
        !engine_.joinable());
    engine_ = std::thread([this] { run_(); });
    engine_thread_id_ = engine_.get_id();
  }

  void shutdown();
  std::future<std::optional<Session>> open_session_async();
  void request_close(SessionId session) noexcept;
  GenerationHandle generate_async(
      SessionId session,
      std::vector<Token> delta,
      GenConfig config,
      GenerationCallback on_update);

  // Engine-thread data, so only stable once that thread is joined.
  EngineMetrics metrics() const {
    return metrics_;
  }

 private:
  enum class Lifecycle { Running, Stopping, Stopped };

  // Runtime state for the one generation currently active on a session.
  struct Generation {
    std::int32_t remaining_tokens = 0;
    std::vector<Token> stop_tokens;
    GenerationCallback on_update;
    // Shared with every handle, so cancelling needs no route back to the
    // runner and works after it is gone.
    std::shared_ptr<GenerationHandleState> state;
    GenerationMetrics m;
    // Engine-side only. An inter-token gap needs the previous delivery, and
    // the published metrics keep only the summary, not the last timestamp.
    MetricsTime last_token_at{};
  };

  // Start-only data. The sampling policy is installed on the executor at
  // admission and the scheduler tasks own the delta after submission, so
  // neither belongs in an active Generation.
  struct GenerationRequest {
    SessionId session = 0;
    std::shared_ptr<const std::vector<Token>> delta;
    SamplingParams sampling;
    std::optional<std::uint64_t> seed;
    Generation generation;
  };

  struct CompleteGeneration {
    FinishReason reason;
    std::optional<Token> pending_token;
  };

  struct ContinueGeneration {
    Token pending_token;
  };

  using NextGenerationAction =
      std::variant<CompleteGeneration, ContinueGeneration>;

  struct InterpretedOutput {
    std::vector<Token> emitted_tokens;
    std::size_t committed_tokens;
    std::int32_t remaining_tokens;
    NextGenerationAction next;
  };

  struct PreparedCompletion {
    TerminalOutcome outcome;
  };

  struct PreparedContinuation {
    Generation* generation;
    GenerationUpdate update;
    Position position;
    std::shared_ptr<const std::vector<Token>> input;
  };

  using PreparedOutput = std::variant<PreparedCompletion, PreparedContinuation>;

  struct SessionRecord {
    SessionRecord() = default;

    // Exclusively owned by sessions_. A copy would duplicate the generation
    // and the carried token, and both copies would answer for one session.
    // Moved into the map on open, so the move members stay.
    SessionRecord(const SessionRecord&) = delete;
    SessionRecord& operator=(const SessionRecord&) = delete;
    SessionRecord(SessionRecord&&) = default;
    SessionRecord& operator=(SessionRecord&&) = default;

    std::shared_ptr<SessionStatus> status;
    bool poisoned = false;
    // The last prediction delivered to the caller but not yet fed back to the
    // executor. It belongs to the logical context but is not included in the
    // executor-committed `position`. A continuation carries it immediately;
    // if generation ends first, the next generation carries it with its delta.
    std::optional<Token> pending;
    std::optional<Generation> active_generation;

    // Engine thread only, so its own reads need no ordering; the release pairs
    // with the acquire in Session::position().
    Position position() const {
      return status->position.load(std::memory_order_relaxed);
    }

    // Cannot wrap: admission bounds a delta inside Position, and
    // handle_output_ checks the room a run needs before taking it.
    //
    // Clears the pending token because the input that just ran carried it.
    // Clearing at submit instead would lose the token if that input were
    // cancelled or rejected before execution.
    void advance(std::size_t tokens) {
      status->position.store(
          position() + static_cast<Position>(tokens),
          std::memory_order_release);
      pending.reset();
    }
  };

  struct OpenCommand {
    std::shared_ptr<std::promise<std::optional<Session>>> ack;
  };

  struct CloseCommand {
    SessionId session;
  };

  struct StartCommand {
    GenerationRequest request;
  };

  struct RetiredSession {
    std::optional<Generation> active_generation;
  };

  using Command = std::variant<OpenCommand, CloseCommand, StartCommand>;

  void run_();
  void process_pending_commands_();
  void process_command_(OpenCommand command);
  void process_command_(CloseCommand command);
  void process_command_(StartCommand command);
  std::optional<RetiredSession> retire_session_(SessionId session);
  void reap_cancelled_();
  bool execute_one_batch_();
  void start_generation_(GenerationRequest request);
  std::optional<TerminalOutcome> validate_generation_start_(
      const GenerationRequest& request,
      const SessionRecord& record) const;
  std::shared_ptr<const std::vector<Token>> build_initial_delta_(
      GenerationRequest& request,
      const SessionRecord& record) const;

  // Split `tokens` into chunks, the last of which produces output.
  //
  // `is_continuation` distinguishes what the executor handed back from the
  // caller's opening delta. Only a continuation of exactly one token is a
  // decode: an opening delta is prefill however short it is, and a longer
  // continuation is re-fed as prefill too.
  std::vector<Task> create_tasks_(
      SessionId session,
      std::shared_ptr<const std::vector<Token>> tokens,
      Position position,
      bool is_continuation);
  InterpretedOutput interpret_output_(
      const Generation& generation,
      const Output& output) const;
  // Prepare callback-visible state from an output-producing task. The caller
  // has already advanced past the slice the executor consumed, so this
  // advances only by however much of the output survives.
  std::optional<PreparedOutput> prepare_output_(
      SessionId session,
      std::optional<Output> output);
  void dispatch_prepared_output_(SessionId session, PreparedOutput prepared);
  void resume_generation_(
      SessionId session,
      Position position,
      std::shared_ptr<const std::vector<Token>> input);
  void handle_output_(SessionId session, std::optional<Output> output);
  void submit_(SessionId session, std::vector<Task> tasks);

  // Central callback seam for admitted generations. Call only after releasing
  // runner and handle locks.
  CallbackResult dispatch_update_(
      const Generation& generation,
      GenerationUpdate update);
  FinishReason deliver_claimed_terminal_(
      const Generation& generation,
      TerminalCompletion completion,
      TerminalOutcome outcome);

  // Invoke the terminal callback and publish state for a detached generation.
  //
  // `on_engine_thread` is false only on the post-shutdown path out of
  // generate_async(), which runs on the caller's thread. The engine may still
  // be draining there, so that path publishes to the handle but must leave the
  // engine's own counters alone.
  void complete_generation_(
      Generation generation,
      TerminalOutcome outcome,
      bool on_engine_thread);
  void complete_request_(
      GenerationRequest request,
      TerminalOutcome outcome,
      bool on_engine_thread);
  // Engine thread only: rolls one finished generation into metrics_.
  void record_completion_(const GenerationMetrics& m, FinishReason reason);
  std::optional<Generation> detach_active_generation_(SessionId session);
  void complete_active_generation_(SessionId session, TerminalOutcome outcome);
  void fail_active_generation_after_callback_(
      SessionId session,
      std::string error_message);
  bool is_running_() const;
  void notify_engine_();

  Executor& executor_;
  // Owned, so that a Session outliving its Runner still finds a live scheduler
  // on the close and cancel paths.
  std::unique_ptr<Scheduler> scheduler_;

  // Lifecycle writes and operations that must linearize with shutdown hold this
  // mutex. Lock-free lifecycle observations use acquire loads.
  std::mutex control_mutex_;
  std::condition_variable engine_cv_;
  std::condition_variable stopped_cv_;
  std::vector<Command> inbox_;
  std::atomic<Lifecycle> lifecycle_{Lifecycle::Running};
  bool join_in_progress_ = false;
  std::thread::id engine_thread_id_;

  // Engine thread only. One record per open session: its shared status, poison
  // flag, carried token, and active generation.
  std::unordered_map<SessionId, SessionRecord> sessions_;
  // Kept after records close to enforce executor IDs are lifetime-unique.
  std::unordered_set<SessionId> issued_session_ids_;
  TaskId next_tid_ = 1;
  EngineMetrics metrics_;

  std::thread engine_;
};

// --- Session ---------------------------------------------------------------

struct SessionState {
  SessionState(
      std::shared_ptr<RunnerImpl> impl,
      SessionId session,
      std::shared_ptr<SessionStatus> status)
      : impl(std::move(impl)), session(session), status(std::move(status)) {}

  ~SessionState() {
    status->open.store(false, std::memory_order_release);
    impl->request_close(session);
  }

  // A copy would share the session id and close it a second time. Held only
  // by unique_ptr, so it is never moved either.
  SessionState(const SessionState&) = delete;
  SessionState& operator=(const SessionState&) = delete;

  std::shared_ptr<RunnerImpl> impl;
  SessionId session;
  std::shared_ptr<SessionStatus> status;
};

Session::Session() = default;
Session::Session(std::unique_ptr<SessionState> state)
    : state_(std::move(state)) {}
Session::~Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

bool Session::valid() const noexcept {
  return state_ && state_->status->open.load(std::memory_order_acquire);
}

Position Session::position() const noexcept {
  if (!state_) {
    return 0;
  }
  return state_->status->position.load(std::memory_order_acquire);
}

GenerationHandle Session::generate_async(
    std::vector<Token> delta,
    GenConfig config,
    GenerationCallback on_update) const {
  if (!state_) {
    auto handle_state = std::make_shared<GenerationHandleState>();
    GenerationHandle handle(handle_state);
    finalize_terminal(
        handle_state,
        on_update,
        TerminalOutcome::failed("session is not initialized"));
    return handle;
  }
  if (!state_->status->open.load(std::memory_order_acquire)) {
    auto handle_state = std::make_shared<GenerationHandleState>();
    GenerationHandle handle(handle_state);
    finalize_terminal(handle_state, on_update, TerminalOutcome::cancelled());
    return handle;
  }
  return state_->impl->generate_async(
      state_->session,
      std::move(delta),
      std::move(config),
      std::move(on_update));
}

// --- lifecycle -------------------------------------------------------------

Runner::Runner(Executor& executor, std::unique_ptr<Scheduler> scheduler)
    : impl_(std::make_shared<RunnerImpl>(executor, std::move(scheduler))) {
  impl_->start();
}

Runner::~Runner() {
  shutdown();
}

void Runner::shutdown() {
  impl_->shutdown();
}

EngineMetrics Runner::metrics() const {
  return impl_->metrics();
}

std::future<std::optional<Session>> Runner::open_session_async() {
  return impl_->open_session_async();
}

// --- RunnerImpl ------------------------------------------------------------

void RunnerImpl::shutdown() {
  bool owns_join = false;
  {
    std::unique_lock<std::mutex> lock(control_mutex_);
    if (lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Running) {
      // Publish the stop after serializing it with admission and reservation.
      lifecycle_.store(Lifecycle::Stopping, std::memory_order_release);
    }

    if (lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Stopped) {
      return;
    }
    if (engine_thread_id_ == std::thread::id{}) {
      // start() never produced a thread, so there is nothing to wake or join.
      // Reaching the terminal state rather than asserting matters because the
      // only way here is a RunnerImpl destroyed after a failed start, and
      // throwing out of that destructor would terminate.
      //
      // Tested on the id rather than engine_.joinable(), which is also false in
      // the window between a successful join and the store below.
      lifecycle_.store(Lifecycle::Stopped, std::memory_order_release);
      return;
    }
    if (std::this_thread::get_id() == engine_thread_id_) {
      lock.unlock();
      notify_engine_();
      return; // the engine thread cannot wait for or join itself
    }
    if (!join_in_progress_) {
      join_in_progress_ = true;
      owns_join = true;
    } else {
      stopped_cv_.wait(lock, [this] {
        return lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Stopped;
      });
      return;
    }
  }

  assert(owns_join && engine_.joinable());
  notify_engine_();
  engine_.join();
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    lifecycle_.store(Lifecycle::Stopped, std::memory_order_release);
    join_in_progress_ = false;
  }
  stopped_cv_.notify_all();
}

bool RunnerImpl::is_running_() const {
  return lifecycle_.load(std::memory_order_acquire) == Lifecycle::Running;
}

void RunnerImpl::notify_engine_() {
  engine_cv_.notify_one();
}

// --- engine thread ---------------------------------------------------------

void RunnerImpl::run_() {
  // Before the loop and before any command is answered, so a caller is never
  // handed a session for an executor that did not come up, and so one-time
  // setup is not charged to whichever generation happened to go first.
  const MetricsTime init_start = MetricsClock::now();
  const bool ready = executor_.initialize();
  metrics_.init_us = us_between(init_start, MetricsClock::now());
  if (!ready) {
    // Stop without running work. The drain below still answers whatever was
    // queued while this was starting, so no caller is left waiting.
    std::lock_guard<std::mutex> lock(control_mutex_);
    lifecycle_.store(Lifecycle::Stopping, std::memory_order_release);
  }

  while (is_running_()) {
    process_pending_commands_();
    reap_cancelled_();
    if (execute_one_batch_()) {
      continue; // more work may be ready
    }
    std::unique_lock<std::mutex> lock(control_mutex_);
    engine_cv_.wait_for(lock, std::chrono::milliseconds(20), [this] {
      return lifecycle_.load(std::memory_order_relaxed) != Lifecycle::Running ||
          !inbox_.empty() || scheduler_->has_work();
    });
  }

  // Commands that raced the stop still need answering, or a caller waiting on
  // an ack would hang.
  process_pending_commands_();

  scheduler_->clear();

  std::vector<SessionId> session_ids;
  session_ids.reserve(sessions_.size());
  for (const auto& entry : sessions_) {
    session_ids.push_back(entry.first);
  }

  std::vector<std::pair<SessionId, std::optional<Generation>>> open_sessions;
  open_sessions.reserve(session_ids.size());
  for (const auto session : session_ids) {
    auto retired = retire_session_(session);
    assert(retired);
    open_sessions.emplace_back(session, std::move(retired->active_generation));
  }
  // Every record is retired before callbacks run, so reentrant requests see
  // all sessions closed.
  assert(sessions_.empty());

  for (auto& entry : open_sessions) {
    if (entry.second) {
      complete_generation_(
          std::move(*entry.second),
          TerminalOutcome::cancelled(),
          /*on_engine_thread=*/true);
    }
    executor_.close_session(entry.first);
  }
}

// Ends generations cancelled since the last pass, before the next get_work()
// can batch their waiting tasks.
void RunnerImpl::reap_cancelled_() {
  std::vector<SessionId> doomed;
  for (const auto& entry : sessions_) {
    const std::optional<Generation>& generation =
        entry.second.active_generation;
    if (generation && generation->state->cancelled.load()) {
      doomed.push_back(entry.first);
    }
  }
  for (SessionId session : doomed) {
    complete_active_generation_(session, TerminalOutcome::cancelled());
  }
}

void RunnerImpl::process_pending_commands_() {
  std::vector<Command> commands;
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    commands.swap(inbox_);
  }
  for (auto& command : commands) {
    std::visit(
        Overloaded{
            [this](OpenCommand& open) { process_command_(std::move(open)); },
            [this](CloseCommand& close) { process_command_(std::move(close)); },
            [this](StartCommand& start) {
              process_command_(std::move(start));
            }},
        command);
  }
}

void RunnerImpl::process_command_(OpenCommand command) {
  const bool running = is_running_();
  auto sid = running ? executor_.open_session() : std::nullopt;
  if (running && !sid) {
    // At capacity. Counted apart from a refusal caused by the runner stopping,
    // which says nothing about how loaded the executor was.
    ++metrics_.sessions_refused;
  }
  bool newly_issued = false;
  bool published = false;
  if (sid) {
    newly_issued = issued_session_ids_.insert(*sid).second;
    assert(
        newly_issued &&
        "Executor::open_session must return lifetime-unique ids");
    if (newly_issued) {
      std::lock_guard<std::mutex> lock(control_mutex_);
      if (lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Running) {
        auto status = std::make_shared<SessionStatus>();
        SessionRecord record;
        record.status = status;
        sessions_.emplace(*sid, std::move(record));
        command.ack->set_value(Session(std::make_unique<SessionState>(
            shared_from_this(), *sid, std::move(status))));
        published = true;
      }
    }
  }
  if (!published) {
    if (newly_issued) {
      executor_.close_session(*sid);
    }
    command.ack->set_value(std::nullopt);
  }
}

void RunnerImpl::process_command_(CloseCommand command) {
  auto retired = retire_session_(command.session);
  if (!retired) {
    return;
  }
  if (retired->active_generation) {
    complete_generation_(
        std::move(*retired->active_generation),
        TerminalOutcome::cancelled(),
        /*on_engine_thread=*/true);
  }
  executor_.close_session(command.session);
}

void RunnerImpl::process_command_(StartCommand command) {
  start_generation_(std::move(command.request));
}

std::optional<RunnerImpl::RetiredSession> RunnerImpl::retire_session_(
    SessionId session_id) {
  auto session = sessions_.find(session_id);
  if (session == sessions_.end()) {
    return std::nullopt;
  }
  session->second.status->open.store(false, std::memory_order_release);
  RetiredSession retired{std::move(session->second.active_generation)};
  // Retire before callback delivery so reentrant requests see the session
  // closed. Cancellation is unconditional because queued tasks may outlive the
  // generation that submitted them.
  sessions_.erase(session);
  (void)scheduler_->cancel(session_id);
  return retired;
}

bool RunnerImpl::execute_one_batch_() {
  std::vector<Task> tasks;
  {
    // Work reservation and the stop transition have a total order.
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (lifecycle_.load(std::memory_order_relaxed) != Lifecycle::Running) {
      return false;
    }
    tasks = scheduler_->get_work();
  }
  if (tasks.empty()) {
    return false;
  }

  // Composition is read here, before to_batch_input moves the Inputs out and
  // drops is_decode with the rest of the scheduling fields.
  //
  // Decode tasks are one sequence each. A session's prefill can arrive as
  // several chunks which are not necessarily adjacent -- DecodeFirstScheduler
  // rotates, taking one chunk per session per pass, so two sessions prefilling
  // together interleave as A, B, A, B. Track every session seen rather than
  // comparing with the previous one, which would count each chunk as a new
  // sequence and charge the step to the generation repeatedly.
  std::uint64_t decode_sessions = 0;
  std::uint64_t prefill_sessions = 0;
  std::uint64_t decode_tokens = 0;
  std::uint64_t prefill_tokens = 0;
  std::vector<SessionId> prefilling; // small: bounded by the batch width
  const MetricsTime step_start = MetricsClock::now();
  for (const Task& task : tasks) {
    bool first_chunk = false;
    if (task.is_decode) {
      ++decode_sessions;
      decode_tokens += task.input.size;
    } else {
      prefill_tokens += task.input.size;
      first_chunk =
          std::find(prefilling.begin(), prefilling.end(), task.input.sid) ==
          prefilling.end();
      if (first_chunk) {
        ++prefill_sessions;
        prefilling.push_back(task.input.sid);
      }
    }
    // Charge the step to the generation once, however many chunks it brought.
    if (!task.is_decode && !first_chunk) {
      continue;
    }
    auto session = sessions_.find(task.input.sid);
    if (session == sessions_.end() || !session->second.active_generation) {
      continue;
    }
    GenerationMetrics& m = session->second.active_generation->m;
    if (!stamped(m.t_first_step)) {
      m.t_first_step = step_start;
    }
    if (task.is_decode) {
      ++m.n_decode_steps;
    } else {
      ++m.n_prefill_steps;
    }
  }
  // Generations eligible for a decode slot, admitted or not. Past their first
  // token, so a generation still prefilling is not counted as held back when
  // it is simply busy elsewhere. Against decode_sessions this is what
  // separates a scheduler holding work back from there being no work.
  //
  // The same pass records how many generations are installed at once, so the
  // batch widths above can be read against the concurrency that was available.
  std::uint64_t live_generations = 0;
  for (const auto& entry : sessions_) {
    const auto& generation = entry.second.active_generation;
    if (!generation) {
      continue;
    }
    ++live_generations;
    if (stamped(generation->m.t_first_token)) {
      ++metrics_.ready_total;
    }
  }
  metrics_.peak_concurrent_generations =
      std::max(metrics_.peak_concurrent_generations, live_generations);

  // Committed context the batch carries into the forward: what attention has
  // to cover. Read from the input positions, which say nothing about how the
  // executor stores the state.
  //
  // Counted once per session, at the end of its furthest chunk. A prompt split
  // across several chunks of one step is one context, not one per chunk, and
  // scheduler.h lets a session appear more than once for exactly that reason.
  std::vector<std::pair<SessionId, std::int64_t>> context_ends;
  for (const Task& task : tasks) {
    const auto end = static_cast<std::int64_t>(task.input.position) +
        static_cast<std::int64_t>(task.input.offset) +
        static_cast<std::int64_t>(task.input.size);
    metrics_.context_max = std::max(metrics_.context_max, end);
    auto entry = std::find_if(
        context_ends.begin(), context_ends.end(), [&](const auto& seen) {
          return seen.first == task.input.sid;
        });
    if (entry == context_ends.end()) {
      context_ends.emplace_back(task.input.sid, end);
    } else {
      entry->second = std::max(entry->second, end);
    }
  }
  for (const auto& entry : context_ends) {
    metrics_.context_sum += entry.second;
  }

  BatchInput batch = to_batch_input(tasks);
  BatchOutput out;
  const bool ok = executor_.execute(batch, out);
  const MetricsTime step_end = MetricsClock::now();

  const std::int64_t latency = us_between(step_start, step_end);
  ++metrics_.steps;
  metrics_.decode_sessions_total += decode_sessions;
  metrics_.prefill_sessions_total += prefill_sessions;
  // Only what the model is known to have taken in. A failed execute leaves
  // what it processed unknown -- that is why the batch is condemned and its
  // sessions poisoned -- so counting the attempt as throughput would credit
  // work that may never have happened. The time is still counted below,
  // because it was really spent, and steps_failed records the attempt.
  if (ok) {
    metrics_.decode_tokens_total += decode_tokens;
    metrics_.prefill_tokens_total += prefill_tokens;
  }
  metrics_.step_latency_sum_us += latency;
  metrics_.step_latency_max_us =
      std::max(metrics_.step_latency_max_us, latency);
  if (!stamped(metrics_.t_first_step)) {
    metrics_.t_first_step = step_start;
  }
  metrics_.t_last_step = step_end;
  if (decode_tokens > 0) {
    ++metrics_.steps_with_decode;
    // Charged once per session: each of them waited this whole step.
    metrics_.decode_session_time_sum_us +=
        latency * static_cast<std::int64_t>(decode_sessions);
  }
  if (prefill_tokens > 0) {
    ++metrics_.steps_with_prefill;
  }
  // Exactly one of the three, so the sums partition step_latency_sum_us. The
  // three step-count conditions below use the same two predicates, so the
  // counts partition `steps` too.
  assert(
      (decode_tokens > 0 || prefill_tokens > 0) &&
      "a non-empty batch carries tokens of at least one kind");
  if (decode_tokens > 0 && prefill_tokens > 0) {
    metrics_.mixed_latency_sum_us += latency;
  } else if (decode_tokens > 0) {
    metrics_.decode_only_latency_sum_us += latency;
    // Attributable, because no prefill shared this forward. Gated on `ok` for
    // the same reason as decode_tokens_total: a failed execute leaves what the
    // model consumed unknown. The latency above is still counted, since it was
    // really spent and the three buckets have to partition the total.
    if (ok) {
      metrics_.decode_only_tokens += decode_tokens;
    }
  } else {
    metrics_.prefill_only_latency_sum_us += latency;
  }
  if (!ok) {
    ++metrics_.steps_failed;
  }

  if (!is_running_()) {
    return true; // discard an in-flight result after the stop boundary
  }
  const bool answered = batch_answered(ok, batch, out);
  // Succeeding without answering every input is a broken executor rather than
  // a failed forward, so trip in debug instead of quietly ending generations.
  assert(
      (!ok || answered) && "Executor::execute must fill one output per input");

  if (!answered) {
    // Whatever the executor did before failing is unknown, so what it holds
    // may no longer match what was asked for. A session can have several
    // chunks in this batch, but its generation is finished only once.
    std::unordered_set<SessionId> failed_sessions;
    for (const Input& in : batch.inputs) {
      auto session = sessions_.find(in.sid);
      if (session != sessions_.end()) {
        session->second.poisoned = true;
        failed_sessions.insert(in.sid);
      }
    }
    const std::string error_message = ok
        ? "executor returned an incomplete batch output"
        : "executor failed to execute the batch";
    for (SessionId session : failed_sessions) {
      complete_active_generation_(
          session, TerminalOutcome::failed(error_message));
    }
    return true;
  }

  std::unordered_set<SessionId> malformed_sessions;
  for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
    const Input& input = batch.inputs[i];
    if (input.produce_output != out.outputs[i].has_value()) {
      malformed_sessions.insert(input.sid);
    }

    auto session = sessions_.find(input.sid);
    if (session != sessions_.end()) {
      // Account the whole settled batch before any callback can publish
      // completion. A session may own several prefill chunks in this batch.
      session->second.advance(input.size);
    }
  }

  for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
    if (!is_running_()) {
      break; // an earlier output callback requested shutdown
    }
    const Input& input = batch.inputs[i];
    auto session = sessions_.find(input.sid);
    if (session == sessions_.end() || !session->second.active_generation) {
      continue;
    }
    if (malformed_sessions.count(input.sid) != 0) {
      // Output presence is part of the executor contract. A mismatch leaves
      // its committed state unknowable, just like a malformed Output.
      session->second.poisoned = true;
      complete_active_generation_(
          input.sid,
          TerminalOutcome::failed("executor returned an invalid output"));
      continue;
    }
    // Session destruction publishes logical closure immediately, before its
    // queued Close command can wait behind this execute. Do not let an
    // in-flight final result beat that close and complete successfully.
    if (!session->second.status->open.load(std::memory_order_acquire)) {
      complete_active_generation_(input.sid, TerminalOutcome::cancelled());
      continue;
    }
    if (session->second.active_generation->state->cancelled.load()) {
      // Nothing this step produced is kept; all consumed slices were accounted
      // above before completion became observable.
      complete_active_generation_(input.sid, TerminalOutcome::cancelled());
      continue;
    }
    if (input.produce_output) {
      // Advancing past the output is left to handle_output_: only it knows how
      // much of it the stop tokens and the token budget let through.
      handle_output_(input.sid, std::move(out.outputs[i]));
    }
  }
  return true;
}

// --- sessions --------------------------------------------------------------

std::future<std::optional<Session>> RunnerImpl::open_session_async() {
  auto ack = std::make_shared<std::promise<std::optional<Session>>>();
  std::future<std::optional<Session>> f = ack->get_future();
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (lifecycle_.load(std::memory_order_relaxed) != Lifecycle::Running) {
      ack->set_value(std::nullopt);
      return f;
    }
    inbox_.emplace_back(OpenCommand{std::move(ack)});
  }
  notify_engine_();
  return f;
}

void RunnerImpl::request_close(SessionId session) noexcept {
  auto queue_close = [this, session] {
    {
      std::lock_guard<std::mutex> lock(control_mutex_);
      if (lifecycle_.load(std::memory_order_relaxed) != Lifecycle::Running) {
        return;
      }
      inbox_.emplace_back(CloseCommand{session});
    }
    notify_engine_();
  };
#if ET_HAS_EXCEPTIONS
  try {
    queue_close();
  } catch (...) {
    // Destruction cannot report admission failure. Shutdown remains the final
    // cleanup boundary for a close request that could not be queued.
  }
#else
  // Queueing can only fail by allocation, which terminates here rather than
  // unwinding, so there is nothing to contain.
  queue_close();
#endif
}

// --- generations -----------------------------------------------------------

GenerationHandle RunnerImpl::generate_async(
    SessionId session,
    std::vector<Token> delta,
    GenConfig config,
    GenerationCallback on_update) {
  auto state = std::make_shared<GenerationHandleState>();
  GenerationRequest request;
  request.session = session;
  request.delta = std::make_shared<const std::vector<Token>>(std::move(delta));
  request.seed = config.seed;
  request.sampling = std::move(config.sampling);
  request.generation.remaining_tokens = config.max_new_tokens;
  request.generation.stop_tokens = std::move(config.stop_tokens);
  request.generation.on_update = std::move(on_update);
  request.generation.state = state;
  // The caller's thread, before the request is queued: the wait a caller sees
  // starts here, not when the engine gets round to it.
  request.generation.m.sid = session;
  request.generation.m.t_submit = MetricsClock::now();

  auto handle = GenerationHandle(state);
  bool admitted = false;
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Running) {
      inbox_.emplace_back(StartCommand{std::move(request)});
      admitted = true;
    }
  }
  if (admitted) {
    notify_engine_();
    return handle;
  }

  // After shutdown nothing drains the inbox, so complete synchronously instead
  // of admitting a start that can never report completion.
  complete_request_(
      std::move(request),
      TerminalOutcome::cancelled(),
      /*on_engine_thread=*/false);
  return handle;
}

std::optional<TerminalOutcome> RunnerImpl::validate_generation_start_(
    const GenerationRequest& request,
    const SessionRecord& record) const {
  if (request.generation.remaining_tokens <= 0) {
    return TerminalOutcome::failed("max_new_tokens must be greater than zero");
  }
  if (!request.delta || request.delta->empty()) {
    return TerminalOutcome::failed("generation delta must not be empty");
  }

  const auto start_position = record.position();
  if (start_position < 0) {
    return TerminalOutcome::failed("session position is invalid");
  }
  const auto room = static_cast<std::size_t>(
      std::numeric_limits<Position>::max() - start_position);
  const auto pending_tokens = record.pending ? 1u : 0u;
  if (pending_tokens > room || request.delta->size() > room - pending_tokens) {
    return TerminalOutcome::failed(
        "generation delta exceeds the session position range");
  }
  if (record.poisoned) {
    return TerminalOutcome::failed(
        "session cannot continue after an executor failure");
  }
  if (record.active_generation) {
    return TerminalOutcome::failed("session already has an active generation");
  }
  return std::nullopt;
}

std::shared_ptr<const std::vector<Token>> RunnerImpl::build_initial_delta_(
    GenerationRequest& request,
    const SessionRecord& record) const {
  auto delta = std::move(request.delta);
  if (!record.pending) {
    return delta;
  }

  std::vector<Token> carried;
  carried.reserve(delta->size() + 1);
  carried.push_back(*record.pending);
  carried.insert(carried.end(), delta->begin(), delta->end());
  return std::make_shared<const std::vector<Token>>(std::move(carried));
}

void RunnerImpl::start_generation_(GenerationRequest request) {
  // Counted on arrival at the engine, not on successful install: every path
  // below ends in a completion, so deferring this would let completions
  // exceed starts.
  ++metrics_.generations_started;
  if (!is_running_() || request.generation.state->cancelled.load()) {
    complete_request_(
        std::move(request),
        TerminalOutcome::cancelled(),
        /*on_engine_thread=*/true);
    return;
  }

  auto session = sessions_.find(request.session);
  if (session == sessions_.end()) {
    complete_request_(
        std::move(request),
        TerminalOutcome::failed("session is not open"),
        /*on_engine_thread=*/true);
    return;
  }
  auto& record = session->second;
  if (auto rejection = validate_generation_start_(request, record)) {
    complete_request_(
        std::move(request), std::move(*rejection), /*on_engine_thread=*/true);
    return;
  }

  const auto start_position = record.position();
  // The caller's own delta, captured before build_initial_delta_ moves it and
  // before any carried token is prepended. The generation tier counts what
  // callers gave; tokens actually fed to the model are EngineMetrics'
  // model_input_tokens().
  request.generation.m.n_prompt_tokens =
      static_cast<std::int64_t>(request.delta->size());
  auto delta = build_initial_delta_(request, record);
  executor_.set_sampling(request.session, request.sampling, request.seed);

  bool installed = false;
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Running) {
      record.active_generation.emplace(std::move(request.generation));
      installed = true;
    }
  }
  if (!installed) {
    // Sampling began before the stop transition, but no task was submitted.
    complete_request_(
        std::move(request),
        TerminalOutcome::cancelled(),
        /*on_engine_thread=*/true);
    return;
  }

  // advance() clears a carried token only after the input holding it runs.
  submit_(
      request.session,
      create_tasks_(
          request.session,
          std::move(delta),
          start_position,
          /*is_continuation=*/false));
}

std::vector<Task> RunnerImpl::create_tasks_(
    SessionId session,
    std::shared_ptr<const std::vector<Token>> tokens,
    Position position,
    bool is_continuation) {
  const auto total = static_cast<std::int32_t>(tokens->size());
  // The scheduler owns this limit, so the runner cannot split a prompt into
  // chunks the scheduler would then refuse. It is non-zero and clamped to the
  // token count, so the loop below always advances.
  const auto limit = scheduler_->max_prefill_chunk_size();
  const auto chunk = limit >= static_cast<std::size_t>(total)
      ? total
      : static_cast<std::int32_t>(limit);
  // A one-token continuation is the decode step. Anything else is prefill,
  // including an opening delta that happens to be a single token.
  const auto decode = is_continuation && total == 1;
  std::vector<Task> tasks;

  for (std::int32_t i = 0; i < total; i += chunk) {
    const auto n = std::min(chunk, total - i);
    const auto last = (i + n) == total;

    Task t;
    t.tid = next_tid_++;
    t.cancelled = false;
    t.is_decode = decode;
    t.input.sid = session;
    t.input.produce_output = last;
    t.input.offset = static_cast<size_t>(i);
    t.input.size = static_cast<size_t>(n);
    t.input.tokens = tokens;
    t.input.position = position;
    tasks.push_back(std::move(t));
  }
  return tasks;
}

void RunnerImpl::submit_(SessionId session, std::vector<Task> tasks) {
  bool running = false;
  bool accepted = false;
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    running = lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Running;
    if (running && !tasks.empty()) {
      accepted = scheduler_->submit(std::move(tasks));
    }
  }
  if (!accepted) {
    auto outcome = running
        ? TerminalOutcome::failed("scheduler rejected generation tasks")
        : TerminalOutcome::cancelled();
    complete_active_generation_(session, std::move(outcome));
  }
}

RunnerImpl::InterpretedOutput RunnerImpl::interpret_output_(
    const Generation& generation,
    const Output& output) const {
  std::vector<Token> emitted_tokens;
  auto remaining_tokens = generation.remaining_tokens;
  std::optional<FinishReason> terminal_reason;

  for (const auto token : output.tokens) {
    emitted_tokens.push_back(token);
    --remaining_tokens;
    if (std::find(
            generation.stop_tokens.begin(),
            generation.stop_tokens.end(),
            token) != generation.stop_tokens.end()) {
      // A stop token is part of the generated stream. It takes precedence when
      // it also exhausts the new-token budget.
      terminal_reason = FinishReason::StopToken;
      break;
    }
    if (remaining_tokens <= 0) {
      terminal_reason = FinishReason::NewTokenLimit;
      break;
    }
  }

  const auto committed_tokens =
      std::min(emitted_tokens.size(), output.tokens.size() - 1);
  if (terminal_reason) {
    const auto pending_token = emitted_tokens.size() == output.tokens.size()
        ? std::optional<Token>(emitted_tokens.back())
        : std::nullopt;
    return InterpretedOutput{
        std::move(emitted_tokens),
        committed_tokens,
        remaining_tokens,
        CompleteGeneration{*terminal_reason, pending_token}};
  }

  assert(!emitted_tokens.empty());
  const auto pending_token = emitted_tokens.back();
  return InterpretedOutput{
      std::move(emitted_tokens),
      committed_tokens,
      remaining_tokens,
      ContinueGeneration{pending_token}};
}

std::optional<RunnerImpl::PreparedOutput> RunnerImpl::prepare_output_(
    SessionId session_id,
    std::optional<Output> output) {
  auto session = sessions_.find(session_id);
  if (session == sessions_.end() || !session->second.active_generation) {
    return std::nullopt;
  }
  auto& record = session->second;
  auto& generation = *record.active_generation;

  // The caller already advanced past the slice the executor consumed, so the
  // session stands at the input's end, and the run below is measured from
  // there.
  const auto room = static_cast<std::size_t>(
      std::numeric_limits<Position>::max() - record.position());
  if (!output || !valid_output(*output, session_id, room)) {
    // The forward ran, so what the executor holds no longer matches anything
    // the runner can name. Same standing as a batch that failed outright.
    record.poisoned = true;
    return PreparedCompletion{
        TerminalOutcome::failed("executor returned an invalid output")};
  }

  auto interpreted = interpret_output_(generation, *output);
  generation.remaining_tokens = interpreted.remaining_tokens;

  // Counted here rather than at delivery: this is the one place that sees the
  // emitted run with the generation in hand, and it covers both the
  // completing and the continuing branch below, which move the tokens away.
  if (!interpreted.emitted_tokens.empty()) {
    const auto emitted =
        static_cast<std::int64_t>(interpreted.emitted_tokens.size());
    const MetricsTime now = MetricsClock::now();
    generation.m.n_generated_tokens += emitted;
    if (!stamped(generation.m.t_first_token)) {
      generation.m.t_first_token = now;
      // The first token's wait is TTFT. Further tokens in the same run have
      // zero caller-visible latency between them.
      const std::int64_t intra_burst = emitted - 1;
      if (intra_burst > 0) {
        generation.m.itl_count += intra_burst;
        generation.m.itl_min_us = 0;
      }
    } else {
      const std::int64_t gap = us_between(generation.last_token_at, now);
      generation.m.itl_count += emitted;
      generation.m.itl_sum_us += gap;
      generation.m.itl_min_us = std::min(
          generation.m.itl_min_us, emitted > 1 ? std::int64_t{0} : gap);
      generation.m.itl_max_us = std::max(generation.m.itl_max_us, gap);
    }
    generation.last_token_at = now;
  }

  // The transcript grows by what the caller keeps, capped by what the executor
  // committed. The last emitted token lands only when it is fed back.
  record.advance(interpreted.committed_tokens);

  return std::visit(
      Overloaded{
          [&](CompleteGeneration complete) -> PreparedOutput {
            record.pending = complete.pending_token;
            auto outcome = complete.reason == FinishReason::StopToken
                ? TerminalOutcome::stopped(
                      std::move(interpreted.emitted_tokens))
                : TerminalOutcome::limit_reached(
                      std::move(interpreted.emitted_tokens));
            return PreparedCompletion{std::move(outcome)};
          },
          [&](ContinueGeneration continuation_action) -> PreparedOutput {
            record.pending = continuation_action.pending_token;

            // Only the last token still has to reach the executor; it committed
            // the rest while producing them. It belongs where the session now
            // stands.
            auto continuation = std::make_shared<const std::vector<Token>>(
                std::vector<Token>{continuation_action.pending_token});
            return PreparedContinuation{
                &generation,
                GenerationUpdate{
                    std::move(interpreted.emitted_tokens), std::nullopt, {}},
                record.position(),
                std::move(continuation)};
          }},
      std::move(interpreted.next));
}

void RunnerImpl::dispatch_prepared_output_(
    SessionId session_id,
    PreparedOutput prepared) {
  std::visit(
      Overloaded{
          [&](PreparedCompletion completion) {
            // Detach and cancel queued work before delivering the terminal
            // callback, preserving the existing reentrant-start behavior.
            complete_active_generation_(
                session_id, std::move(completion.outcome));
          },
          [&](PreparedContinuation continuation) {
            // The pointer names the active generation in sessions_. Commands
            // are not drained during synchronous dispatch, so it remains valid
            // until the callback returns. Resume does not reuse it.
            auto callback_result = dispatch_update_(
                *continuation.generation, std::move(continuation.update));
            if (!callback_result.succeeded) {
              fail_active_generation_after_callback_(
                  session_id, std::move(callback_result.error_message));
              return;
            }
            resume_generation_(
                session_id,
                continuation.position,
                std::move(continuation.input));
          }},
      std::move(prepared));
}

void RunnerImpl::resume_generation_(
    SessionId session_id,
    Position position,
    std::shared_ptr<const std::vector<Token>> input) {
  if (!is_running_()) {
    return; // shutdown callback; cleanup cancels this generation
  }

  // Re-find engine-owned state after arbitrary user code. Task creation stays
  // here so callback failure, shutdown, or retirement does not consume a tid.
  const auto session = sessions_.find(session_id);
  if (session == sessions_.end() || !session->second.active_generation) {
    return;
  }

  submit_(
      session_id,
      create_tasks_(
          session_id,
          std::move(input),
          position,
          /*is_continuation=*/true));
}

void RunnerImpl::handle_output_(
    SessionId session_id,
    std::optional<Output> output) {
  auto prepared = prepare_output_(session_id, std::move(output));
  if (prepared) {
    dispatch_prepared_output_(session_id, std::move(*prepared));
  }
}

CallbackResult RunnerImpl::dispatch_update_(
    const Generation& generation,
    GenerationUpdate update) {
  // TODO: Dispatch user callbacks through a callback pool while preserving
  // per-generation ordering and posting completion back to the engine thread.
  return invoke_callback(generation.on_update, std::move(update));
}

// Returns the reason actually published, which is not the one passed in when
// the terminal callback throws. The engine counts that final reason, so the
// tallies agree with what the handle reports.
FinishReason RunnerImpl::deliver_claimed_terminal_(
    const Generation& generation,
    TerminalCompletion completion,
    TerminalOutcome outcome) {
  auto callback_result = dispatch_update_(
      generation,
      GenerationUpdate{
          std::move(outcome.tokens), outcome.reason, outcome.error_message});
  if (!callback_result.succeeded) {
    outcome = TerminalOutcome::failed(std::move(callback_result.error_message));
  }
  const FinishReason reason = outcome.reason;
  completion.finish(std::move(outcome), generation.m);
  return reason;
}

void RunnerImpl::complete_generation_(
    Generation generation,
    TerminalOutcome outcome,
    bool on_engine_thread) {
  generation.m.t_end = MetricsClock::now();
  std::optional<TerminalCompletion> completion;
  {
    // The claim is taken under the runner lock. User code still runs only after
    // both runner and handle locks are released.
    //
    // The outcome stands as given, whether or not shutdown is racing it.
    // A generation that reached its stop token or its budget keeps those
    // tokens, which the session already retained and which Cancelled would drop
    // from the caller's view of the context. A generation that failed keeps its
    // reason and diagnostic: the executor must outlive shutdown, and a result
    // produced after the stop boundary is discarded in execute_one_batch_, so a
    // failure that reaches here is a real contract violation rather than
    // teardown noise, and reporting it as Cancelled would hide it.
    std::lock_guard<std::mutex> lock(control_mutex_);
    auto claimed = TerminalCompletion::try_claim(generation.state);
    if (!claimed) {
      return;
    }
    completion.emplace(std::move(*claimed));
  }
  const FinishReason reason = deliver_claimed_terminal_(
      generation, std::move(*completion), std::move(outcome));
  if (on_engine_thread) {
    record_completion_(generation.m, reason);
  }
}

void RunnerImpl::record_completion_(
    const GenerationMetrics& m,
    FinishReason reason) {
  ++metrics_.generations_completed;
  switch (reason) {
    case FinishReason::StopToken:
      ++metrics_.finished_stop_token;
      break;
    case FinishReason::NewTokenLimit:
      ++metrics_.finished_token_limit;
      break;
    case FinishReason::Cancelled:
      ++metrics_.finished_cancelled;
      break;
    case FinishReason::Failed:
      ++metrics_.finished_failed;
      break;
  }
  metrics_.total_prompt_tokens += m.n_prompt_tokens;
  metrics_.total_generated_tokens += m.n_generated_tokens;
  // Zero for a generation that never reached a first token. Counted
  // separately from completions so the mean divides by the samples it has,
  // and so the minimum stays untouched when there are none.
  const std::int64_t ttft = m.ttft_us();
  if (ttft > 0) {
    ++metrics_.ttft_count;
    metrics_.ttft_sum_us += ttft;
    metrics_.ttft_min_us = std::min(metrics_.ttft_min_us, ttft);
    metrics_.ttft_max_us = std::max(metrics_.ttft_max_us, ttft);
  }
}

void RunnerImpl::complete_request_(
    GenerationRequest request,
    TerminalOutcome outcome,
    bool on_engine_thread) {
  complete_generation_(
      std::move(request.generation), std::move(outcome), on_engine_thread);
}

std::optional<RunnerImpl::Generation> RunnerImpl::detach_active_generation_(
    SessionId session_id) {
  auto session = sessions_.find(session_id);
  if (session == sessions_.end() || !session->second.active_generation) {
    return std::nullopt;
  }
  // Detach before callback delivery so a reentrant request sees an idle
  // session.
  auto active = std::move(*session->second.active_generation);
  session->second.active_generation.reset();

  for (auto& task : scheduler_->cancel(session_id)) {
    (void)task;
  }
  return active;
}

void RunnerImpl::complete_active_generation_(
    SessionId session_id,
    TerminalOutcome outcome) {
  auto active = detach_active_generation_(session_id);
  if (!active) {
    return;
  }
  complete_generation_(
      std::move(*active), std::move(outcome), /*on_engine_thread=*/true);
}

void RunnerImpl::fail_active_generation_after_callback_(
    SessionId session_id,
    std::string error_message) {
  // Once the callback returns an exception, that observed failure wins over
  // any shutdown that raced the callback. Do not invoke the failed callback
  // again.
  auto active = detach_active_generation_(session_id);
  if (!active) {
    return;
  }
  active->m.t_end = MetricsClock::now();
  auto completion = TerminalCompletion::try_claim(active->state);
  if (!completion) {
    return;
  }
  completion->finish(
      TerminalOutcome::failed(std::move(error_message)), active->m);
  // This path claims the terminal itself instead of going through
  // complete_generation_, so it has to count its own completion. Without this
  // the engine would report fewer completions than starts, and precisely for
  // the generations that failed most interestingly.
  record_completion_(active->m, FinishReason::Failed);
}

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
