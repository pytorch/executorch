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
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

namespace {

bool valid_positioned_tokens(
    Position position,
    const std::shared_ptr<const std::vector<Token>>& tokens,
    std::size_t extra = 0) {
  if (position < 0 || !tokens || tokens->empty()) {
    return false;
  }
  const auto room =
      static_cast<std::size_t>(std::numeric_limits<Position>::max() - position);
  return extra <= room && tokens->size() <= room - extra;
}

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

} // namespace

// Shared between the runner and every handle to one generation. Out of the
// public header so that callers see neither the lock nor what it guards.
struct GenerationHandleState {
  std::mutex mutex;
  std::condition_variable cv;
  bool done = false;
  FinishReason reason = FinishReason::Failed;
  std::atomic<bool> cancelled{false};
};

struct SessionStatus {
  std::atomic<bool> open{true};
  // The session's one position counter. It lives here rather than on the
  // engine thread's SessionRecord so that Session::position() can read it
  // without touching sessions_, which only the engine thread may. The engine
  // thread is the sole writer.
  std::atomic<Position> position{0};
};

bool publish_terminal_state(
    const std::shared_ptr<GenerationHandleState>& state,
    FinishReason reason) {
  std::lock_guard<std::mutex> lock(state->mutex);
  if (state->done) {
    return false;
  }
  state->reason = reason;
  state->done = true;
  return true;
}

void complete_terminal(
    const std::shared_ptr<GenerationHandleState>& state,
    const GenerationCallback& on_update,
    const std::vector<Token>& tokens,
    FinishReason reason) {
  if (!publish_terminal_state(state, reason)) {
    return;
  }
  state->cv.notify_all();
  if (on_update) {
    on_update(tokens, reason);
  }
}

// --- GenerationHandle ------------------------------------------------------

void GenerationHandle::cancel() const {
  if (state_) {
    state_->cancelled.store(true);
  }
}

bool GenerationHandle::done() const {
  if (!state_) {
    return true;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->done;
}

void GenerationHandle::wait() const {
  if (!state_) {
    return;
  }
  std::unique_lock<std::mutex> lock(state_->mutex);
  state_->cv.wait(lock, [this] { return state_->done; });
}

FinishReason GenerationHandle::finish_reason() const {
  if (!state_) {
    return FinishReason::Failed;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->reason;
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
  std::future<std::optional<Session>> open_session();
  void request_close(SessionId session) noexcept;
  GenerationHandle generate_async(
      SessionId session,
      std::vector<Token> delta,
      GenConfig config,
      GenerationCallback on_update);

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
    // Accepted and emitted by the last generation but never fed back, because
    // that generation ended while it was outstanding. The executor therefore
    // sits one below `position` until the next delta carries it.
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
    // Clears any carried token, because the input that just ran is the one
    // that carried it. Nothing else could have: once `pending` is set, the
    // generation that set it has ended and its queued tasks are cancelled.
    // Clearing at submit instead would lose the token if that input never ran.
    void advance(std::size_t tokens) {
      status->position.store(
          position() + static_cast<Position>(tokens),
          std::memory_order_release);
      pending.reset();
    }
  };

  struct Command {
    enum class Kind { Open, Close, Start } kind = Kind::Open;
    SessionId session = 0;
    std::optional<GenerationRequest> generation_request;
    std::shared_ptr<std::promise<std::optional<Session>>> open_ack;
  };

  void run_();
  void process_pending_commands_();
  void reap_cancelled_();
  bool execute_one_batch_();
  void start_generation_(GenerationRequest request);

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
  // Emit what an output-producing task produced, then submit what follows. The
  // caller has already advanced past the slice the executor consumed, so this
  // advances only by however much of the output survives.
  void handle_output_(SessionId session, std::optional<Output> output);
  void submit_(SessionId session, std::vector<Task> tasks);

  // Central callback seam. Call only after releasing runner and handle locks.
  void deliver_update_(
      const Generation& generation,
      const std::vector<Token>& tokens,
      std::optional<FinishReason> reason);

  // Publish terminal state and invoke the callback for a detached generation.
  void complete_generation_(
      Generation generation,
      FinishReason reason,
      std::vector<Token> final_tokens = {});
  void complete_request_(GenerationRequest request, FinishReason reason);
  void complete_active_generation_(
      SessionId session,
      FinishReason reason,
      std::vector<Token> final_tokens = {});
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
    complete_terminal(handle_state, on_update, {}, FinishReason::Failed);
    return handle;
  }
  if (!state_->status->open.load(std::memory_order_acquire)) {
    auto handle_state = std::make_shared<GenerationHandleState>();
    GenerationHandle handle(handle_state);
    complete_terminal(handle_state, on_update, {}, FinishReason::Cancelled);
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

std::future<std::optional<Session>> Runner::open_session() {
  return impl_->open_session();
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

  std::vector<std::pair<SessionId, std::optional<Generation>>> open_sessions;
  open_sessions.reserve(sessions_.size());
  for (auto& entry : sessions_) {
    entry.second.status->open.store(false, std::memory_order_release);
    open_sessions.emplace_back(
        entry.first, std::move(entry.second.active_generation));
  }
  // Retire sessions before callbacks run so reentrant requests see them closed.
  sessions_.clear();

  for (auto& entry : open_sessions) {
    if (entry.second) {
      complete_generation_(std::move(*entry.second), FinishReason::Cancelled);
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
    complete_active_generation_(session, FinishReason::Cancelled);
  }
}

void RunnerImpl::process_pending_commands_() {
  std::vector<Command> commands;
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    commands.swap(inbox_);
  }
  for (Command& cmd : commands) {
    switch (cmd.kind) {
      case Command::Kind::Open: {
        std::optional<SessionId> sid =
            is_running_() ? executor_.open_session() : std::nullopt;
        bool newly_issued = false;
        bool published = false;
        if (sid) {
          newly_issued = issued_session_ids_.insert(*sid).second;
          assert(
              newly_issued &&
              "Executor::open_session must return lifetime-unique ids");
          if (newly_issued) {
            std::lock_guard<std::mutex> lock(control_mutex_);
            if (lifecycle_.load(std::memory_order_relaxed) ==
                Lifecycle::Running) {
              auto status = std::make_shared<SessionStatus>();
              SessionRecord record;
              record.status = status;
              sessions_.emplace(*sid, std::move(record));
              cmd.open_ack->set_value(Session(std::make_unique<SessionState>(
                  shared_from_this(), *sid, std::move(status))));
              published = true;
            }
          }
        }
        if (!published) {
          if (newly_issued) {
            executor_.close_session(*sid);
          }
          cmd.open_ack->set_value(std::nullopt);
        }
        break;
      }
      case Command::Kind::Close: {
        auto session = sessions_.find(cmd.session);
        if (session == sessions_.end()) {
          break;
        }
        session->second.status->open.store(false, std::memory_order_release);
        std::optional<Generation> active =
            std::move(session->second.active_generation);
        // Retire the session before its terminal callback can enqueue new work.
        sessions_.erase(session);
        // Unconditional: a task can outlive the generation that submitted it,
        // so an empty active_generation does not mean an empty queue. Anything
        // left would run against a session the executor has released.
        (void)scheduler_->cancel(cmd.session);
        if (active) {
          complete_generation_(std::move(*active), FinishReason::Cancelled);
        }
        executor_.close_session(cmd.session);
        break;
      }
      case Command::Kind::Start:
        start_generation_(std::move(*cmd.generation_request));
        break;
    }
  }
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

  BatchInput batch = to_batch_input(tasks);
  BatchOutput out;
  const bool ok = executor_.execute(batch, out);
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
    for (SessionId session : failed_sessions) {
      complete_active_generation_(session, FinishReason::Failed);
    }
    return true;
  }

  for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
    if (!is_running_()) {
      break; // an earlier output callback requested shutdown
    }
    const Input& input = batch.inputs[i];
    auto session = sessions_.find(input.sid);
    if (session == sessions_.end()) {
      continue;
    }
    // The executor consumed this slice, so the session moves past it before
    // anything below decides the generation's fate. This sits above every early
    // return so a session with several chunks in one batch counts all of them,
    // not just those preceding the chunk that ended it.
    session->second.advance(input.size);
    if (!session->second.active_generation) {
      continue;
    }
    // Session destruction publishes logical closure immediately, before its
    // queued Close command can wait behind this execute. Do not let an
    // in-flight final result beat that close and complete successfully.
    if (!session->second.status->open.load(std::memory_order_acquire)) {
      complete_active_generation_(input.sid, FinishReason::Cancelled);
      continue;
    }
    if (session->second.active_generation->state->cancelled.load()) {
      // Nothing this step produced is kept, so the slice above is the whole
      // advance.
      complete_active_generation_(input.sid, FinishReason::Cancelled);
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

std::future<std::optional<Session>> RunnerImpl::open_session() {
  auto ack = std::make_shared<std::promise<std::optional<Session>>>();
  std::future<std::optional<Session>> f = ack->get_future();
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (lifecycle_.load(std::memory_order_relaxed) != Lifecycle::Running) {
      ack->set_value(std::nullopt);
      return f;
    }
    Command cmd;
    cmd.kind = Command::Kind::Open;
    cmd.open_ack = std::move(ack);
    inbox_.push_back(std::move(cmd));
  }
  notify_engine_();
  return f;
}

void RunnerImpl::request_close(SessionId session) noexcept {
  try {
    {
      std::lock_guard<std::mutex> lock(control_mutex_);
      if (lifecycle_.load(std::memory_order_relaxed) != Lifecycle::Running) {
        return;
      }
      Command cmd;
      cmd.kind = Command::Kind::Close;
      cmd.session = session;
      inbox_.push_back(std::move(cmd));
    }
    notify_engine_();
  } catch (...) {
    // Destruction cannot report admission failure. Shutdown remains the final
    // cleanup boundary for a close request that could not be queued.
  }
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

  GenerationHandle handle(state);
  bool admitted = false;
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Running) {
      Command cmd;
      cmd.kind = Command::Kind::Start;
      cmd.generation_request.emplace(std::move(request));
      inbox_.push_back(std::move(cmd));
      admitted = true;
    }
  }
  if (admitted) {
    notify_engine_();
    return handle;
  }

  // After shutdown nothing drains the inbox, so complete synchronously instead
  // of admitting a start that can never report completion.
  complete_request_(std::move(request), FinishReason::Cancelled);
  return handle;
}

void RunnerImpl::start_generation_(GenerationRequest request) {
  if (!is_running_() || request.generation.state->cancelled.load()) {
    complete_request_(std::move(request), FinishReason::Cancelled);
    return;
  }
  auto session = sessions_.find(request.session);
  if (session == sessions_.end()) {
    complete_request_(std::move(request), FinishReason::Failed);
    return;
  }
  SessionRecord& record = session->second;
  // Where the session left off, 0 for one just opened. A token the last
  // generation emitted but never fed belongs exactly here, so the delta carries
  // it and the range needs room for one more.
  const Position start_position = record.position();
  if (request.generation.remaining_tokens <= 0 ||
      !valid_positioned_tokens(
          start_position, request.delta, record.pending ? 1u : 0u)) {
    complete_request_(std::move(request), FinishReason::Failed);
    return;
  }
  // A step on this session failed mid-execute, so what the executor holds for
  // it is unknown. Anything built on that would be silently wrong.
  if (record.poisoned || record.active_generation) {
    complete_request_(std::move(request), FinishReason::Failed);
    return;
  }

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
    complete_request_(std::move(request), FinishReason::Cancelled);
    return;
  }

  // Carry a token the last generation emitted but never fed. Not cleared here:
  // advance() does that once the input holding it runs, so a generation
  // cancelled or rejected before then does not drop it.
  std::shared_ptr<const std::vector<Token>> delta = std::move(request.delta);
  if (record.pending) {
    std::vector<Token> carried;
    carried.reserve(delta->size() + 1);
    carried.push_back(*record.pending);
    carried.insert(carried.end(), delta->begin(), delta->end());
    delta = std::make_shared<const std::vector<Token>>(std::move(carried));
  }

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
  const std::size_t limit = scheduler_->max_prefill_chunk_size();
  const std::int32_t chunk = limit >= static_cast<std::size_t>(total)
      ? total
      : static_cast<std::int32_t>(limit);
  // A one-token continuation is the decode step. Anything else is prefill,
  // including an opening delta that happens to be a single token.
  const bool decode = is_continuation && total == 1;
  std::vector<Task> tasks;

  for (std::int32_t i = 0; i < total; i += chunk) {
    const std::int32_t n = std::min(chunk, total - i);
    const bool last = (i + n) == total;

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
    complete_active_generation_(
        session, running ? FinishReason::Failed : FinishReason::Cancelled);
  }
}

void RunnerImpl::handle_output_(
    SessionId session_id,
    std::optional<Output> output) {
  auto session = sessions_.find(session_id);
  if (session == sessions_.end() || !session->second.active_generation) {
    return;
  }
  SessionRecord& record = session->second;
  Generation& generation = *record.active_generation;

  // The caller already advanced past the slice the executor consumed, so the
  // session stands at the input's end, and the run below is measured from
  // there.
  const auto room = static_cast<std::size_t>(
      std::numeric_limits<Position>::max() - record.position());
  if (!output || !valid_output(*output, session_id, room)) {
    // The forward ran, so what the executor holds no longer matches anything
    // the runner can name. Same standing as a batch that failed outright.
    record.poisoned = true;
    complete_active_generation_(session_id, FinishReason::Failed);
    return;
  }

  // A run can hit a stop token or the budget part way through, since a
  // speculative executor answers with several tokens, so take the prefix up to
  // whichever comes first and emit it in one call.
  std::vector<Token> emit;
  FinishReason reason = FinishReason::NewTokenLimit;
  bool ends = false;

  for (Token token : output->tokens) {
    if (std::find(
            generation.stop_tokens.begin(),
            generation.stop_tokens.end(),
            token) != generation.stop_tokens.end()) {
      reason = FinishReason::StopToken;
      ends = true;
      break;
    }
    emit.push_back(token);
    --generation.remaining_tokens;
    if (generation.remaining_tokens <= 0) {
      ends = true;
      break;
    }
  }

  // The transcript grows by what the caller keeps, capped by what the executor
  // committed: every produced token but the last, which lands only when fed
  // back. Anything committed past this is rewound when the session's next input
  // arrives below where it stands.
  record.advance(std::min(emit.size(), output->tokens.size() - 1));

  if (emit.size() == output->tokens.size()) {
    // The last token was delivered but not fed, so the executor sits one below
    // the position above. Recorded whether or not the generation continues: the
    // continuation submitted below normally feeds it, but cancelling or
    // dropping that task would otherwise lose a token the caller already has.
    record.pending = emit.back();
  }

  if (ends) {
    complete_active_generation_(session_id, reason, std::move(emit));
    return;
  }

  // Only the last token still has to reach the executor; it committed the rest
  // while producing them. It belongs where the session now stands.
  const Position continuation_position = record.position();
  auto continuation = std::make_shared<const std::vector<Token>>(
      std::vector<Token>{emit.back()});

  // Runs arbitrary user code, and `generation` lives in sessions_. Callbacks
  // reach the runner only by queueing commands, which are not drained until the
  // next engine pass, so the record cannot be retired underneath this call.
  deliver_update_(generation, emit, std::nullopt);
  if (!is_running_()) {
    return; // shutdown requested from the callback; cleanup cancels this
  }

  // Re-found rather than reusing `generation`, so nothing below depends on what
  // the callback did or did not do.
  session = sessions_.find(session_id);
  if (session == sessions_.end() || !session->second.active_generation) {
    return;
  }

  submit_(
      session_id,
      create_tasks_(
          session_id,
          std::move(continuation),
          continuation_position,
          /*is_continuation=*/true));
}

void RunnerImpl::deliver_update_(
    const Generation& generation,
    const std::vector<Token>& tokens,
    std::optional<FinishReason> reason) {
  if (generation.on_update) {
    // TODO: Dispatch user callbacks through a callback pool.
    generation.on_update(tokens, reason);
  }
}

void RunnerImpl::complete_generation_(
    Generation generation,
    FinishReason reason,
    std::vector<Token> final_tokens) {
  bool published = false;
  {
    // The stop transition and terminal publication have a total order. User
    // code still runs only after both runner and handle locks are released.
    std::lock_guard<std::mutex> lock(control_mutex_);
    // A generation that genuinely reached its stop token or its budget keeps
    // that outcome even when shutdown races it. The session already counted
    // those tokens, so reporting Cancelled and dropping them would leave the
    // caller believing less is in context than there is.
    const bool finished = reason == FinishReason::StopToken ||
        reason == FinishReason::NewTokenLimit;
    if (lifecycle_.load(std::memory_order_relaxed) != Lifecycle::Running &&
        !finished) {
      reason = FinishReason::Cancelled;
      final_tokens.clear();
    }
    published = publish_terminal_state(generation.state, reason);
  }
  if (!published) {
    return;
  }
  generation.state->cv.notify_all();
  // Handle state is visible before user code inspects it from the callback.
  deliver_update_(generation, final_tokens, reason);
}

void RunnerImpl::complete_request_(
    GenerationRequest request,
    FinishReason reason) {
  complete_generation_(std::move(request.generation), reason);
}

void RunnerImpl::complete_active_generation_(
    SessionId session_id,
    FinishReason reason,
    std::vector<Token> final_tokens) {
  auto session = sessions_.find(session_id);
  if (session == sessions_.end() || !session->second.active_generation) {
    return;
  }
  // Detached before the callback runs, so a reentrant request sees the session
  // idle rather than mid-completion.
  Generation active = std::move(*session->second.active_generation);
  session->second.active_generation.reset();

  // Anything of this generation still queued would otherwise reach a batch.
  for (Task& task : scheduler_->cancel(session_id)) {
    (void)task;
  }
  complete_generation_(std::move(active), reason, std::move(final_tokens));
}

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
