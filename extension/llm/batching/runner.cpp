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
    const std::shared_ptr<const std::vector<Token>>& tokens) {
  if (position < 0 || !tokens || tokens->empty()) {
    return false;
  }
  return tokens->size() <=
      static_cast<std::size_t>(std::numeric_limits<Position>::max() - position);
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

// Everything the runner owns.
// Runner finds a stopped object rather than a dead one.
class RunnerImpl : public std::enable_shared_from_this<RunnerImpl> {
 public:
  RunnerImpl(Executor& executor, Scheduler& scheduler, RunnerConfig config)
      : executor_(executor), scheduler_(scheduler), config_(config) {}

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
  std::future<void> close_session(SessionId session);
  GenerationHandle generate_async(
      SessionId session,
      std::vector<Token> delta,
      GenConfig config,
      GenerationCallback on_update);

 private:
  enum class Lifecycle { Running, Stopping, Stopped };

  // Engine-thread-only, except `state` and the fields read by generate_async
  // before the start command is picked up.
  struct GenState {
    SessionId session = 0;
    std::shared_ptr<const std::vector<Token>> delta;
    std::int32_t generated = 0;
    GenConfig config;
    GenerationCallback on_update;
    // Shared with every handle, so cancelling needs no route back to the
    // runner and works after it is gone.
    std::shared_ptr<GenerationHandleState> state =
        std::make_shared<GenerationHandleState>();
    bool registered = false; // owns its session slot in gens_
    bool done = false;
  };
  using GenPtr = std::shared_ptr<GenState>;

  struct Command {
    enum class Kind { Open, Close, Start } kind = Kind::Open;
    SessionId session = 0;
    GenPtr gen;
    std::shared_ptr<std::promise<std::optional<Session>>> open_ack;
    std::shared_ptr<std::promise<void>> close_ack;
  };

  void run_();
  void process_pending_commands_();
  void reap_cancelled_();
  bool execute_one_batch_();
  void start_generation_(GenPtr gen);

  // Split `tokens` into chunks, the last of which produces output.
  //
  // `is_continuation` distinguishes what the executor handed back from the
  // caller's opening delta. Only a continuation of exactly one token is a
  // decode: an opening delta is prefill however short it is, and a longer
  // continuation is re-fed as prefill too.
  std::vector<Task> create_tasks_(
      const GenPtr& gen,
      std::shared_ptr<const std::vector<Token>> tokens,
      Position position,
      bool is_continuation);
  // Emit what an output-producing task produced, then submit what follows.
  void handle_output_(const GenPtr& gen, std::optional<Output> output);
  void submit_(const GenPtr& gen, std::vector<Task> tasks);

  // Session position bookkeeping. Engine thread only; a session that has
  // closed is ignored.
  void advance_session_(SessionId session, std::size_t consumed);
  void set_session_position_(SessionId session, Position position);

  // Central callback seam. Call only after releasing runner and handle locks.
  void deliver_update_(
      const GenPtr& gen,
      const std::vector<Token>& tokens,
      std::optional<FinishReason> reason);

  // Takes a GenPtr because removing a registered generation may release its
  // owning table entry. Also completes generations rejected before registration.
  void complete_generation_(
      const GenPtr& gen,
      FinishReason reason,
      std::vector<Token> final_tokens = {});
  bool is_running_() const;
  void notify_engine_();

  Executor& executor_;
  Scheduler& scheduler_;
  RunnerConfig config_;

  // Lifecycle writes and operations that must linearize with shutdown hold this
  // mutex. Lock-free lifecycle observations use acquire loads.
  std::mutex control_mutex_;
  std::condition_variable engine_cv_;
  std::condition_variable stopped_cv_;
  std::vector<Command> inbox_;
  std::atomic<Lifecycle> lifecycle_{Lifecycle::Running};
  bool join_in_progress_ = false;
  std::thread::id engine_thread_id_;

  // Engine thread only.
  // Open sessions and the logical position each has reached. A session opens
  // at 0 and advances by what the executor consumes, so a later generation
  // continues where the previous one ended.
  std::unordered_map<SessionId, Position> open_sessions_;
  std::unordered_set<SessionId> issued_session_ids_;
  std::unordered_map<SessionId, GenPtr> gens_;
  // Sessions whose executor state may no longer match what was asked for,
  // because a step on them failed mid-execute. Cleared only by close_session().
  std::unordered_set<SessionId> poisoned_;
  TaskId next_tid_ = 1;

  std::thread engine_;
};

// --- Session ---------------------------------------------------------------

GenerationHandle Session::generate_async(
    std::vector<Token> delta,
    GenConfig config,
    GenerationCallback on_update) const {
  return impl_->generate_async(
      sid_,
      std::move(delta),
      std::move(config),
      std::move(on_update));
}

std::future<void> Session::close() const {
  return impl_->close_session(sid_);
}

// --- lifecycle -------------------------------------------------------------

Runner::Runner(
    Executor& executor,
    Scheduler& scheduler,
    RunnerConfig config)
    : impl_(std::make_shared<RunnerImpl>(executor, scheduler, config)) {
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

std::future<void> Runner::close_session(SessionId session) {
  return impl_->close_session(session);
}

GenerationHandle Runner::generate_async(
    SessionId session,
    std::vector<Token> delta,
    GenConfig config,
    GenerationCallback on_update) {
  return impl_->generate_async(
      session,
      std::move(delta),
      std::move(config),
      std::move(on_update));
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
    if (std::this_thread::get_id() == engine_thread_id_) {
      lock.unlock();
      notify_engine_();
      return; // the engine thread cannot wait for or join itself
    }
    if (!join_in_progress_) {
      join_in_progress_ = true;
      owns_join = true;
    } else {
      stopped_cv_.wait(
          lock,
          [this] {
            return lifecycle_.load(std::memory_order_relaxed) ==
                Lifecycle::Stopped;
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
    engine_cv_.wait_for(
        lock,
        std::chrono::milliseconds(20),
        [this] {
          return lifecycle_.load(std::memory_order_relaxed) !=
              Lifecycle::Running ||
              !inbox_.empty() || scheduler_.has_work();
        });
  }

  // Commands that raced the stop still need answering, or a caller waiting on
  // an ack would hang.
  process_pending_commands_();

  scheduler_.clear();

  std::vector<SessionId> open_sessions;
  open_sessions.reserve(open_sessions_.size());
  for (const auto& entry : open_sessions_) {
    open_sessions.push_back(entry.first);
  }
  // Retire sessions before callbacks run so reentrant requests see them closed.
  open_sessions_.clear();

  std::vector<GenPtr> live;
  live.reserve(gens_.size());
  for (auto& entry : gens_) {
    live.push_back(entry.second);
  }
  for (const GenPtr& gen : live) {
    complete_generation_(gen, FinishReason::Cancelled);
  }
  for (SessionId session : open_sessions) {
    executor_.close_session(session);
    poisoned_.erase(session);
  }
}

// Ends generations cancelled since the last pass, before the next get_work()
// can batch their waiting tasks.
void RunnerImpl::reap_cancelled_() {
  std::vector<GenPtr> doomed;
  for (auto& entry : gens_) {
    if (entry.second->state->cancelled.load()) {
      doomed.push_back(entry.second);
    }
  }
  for (const GenPtr& gen : doomed) {
    complete_generation_(gen, FinishReason::Cancelled);
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
              open_sessions_.emplace(*sid, Position{0});
              published = true;
            }
          }
        }
        if (published) {
          cmd.open_ack->set_value(Session(shared_from_this(), *sid));
        } else {
          if (newly_issued) {
            executor_.close_session(*sid);
          }
          cmd.open_ack->set_value(std::nullopt);
        }
        break;
      }
      case Command::Kind::Close: {
        if (open_sessions_.erase(cmd.session) == 0) {
          cmd.close_ack->set_value();
          break;
        }
        auto it = gens_.find(cmd.session);
        if (it != gens_.end()) {
          // Copy before completing: completion erases this very entry.
          GenPtr gen = it->second;
          complete_generation_(gen, FinishReason::Cancelled);
        }
        executor_.close_session(cmd.session);
        // Session ids are lifetime-unique, so no later session can inherit this
        // poisoned state.
        poisoned_.erase(cmd.session);
        cmd.close_ack->set_value();
        break;
      }
      case Command::Kind::Start:
        start_generation_(std::move(cmd.gen));
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
    tasks = scheduler_.get_work();
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
  const bool answered_all = out.outputs.size() == tasks.size();
  // Succeeding without answering every input is a broken executor rather than
  // a failed forward, so trip in debug instead of quietly ending generations.
  assert(
      (!ok || answered_all) &&
      "Executor::execute must fill one output per input");

  if (!ok || !answered_all) {
    // Whatever the executor did before failing is unknown, so what it holds
    // may no longer match what was asked for. A session can have several
    // chunks in this batch, but its generation is finished only once.
    std::unordered_set<SessionId> failed_sessions;
    for (const Input& in : batch.inputs) {
      poisoned_.insert(in.sid);
      failed_sessions.insert(in.sid);
    }
    for (SessionId session : failed_sessions) {
      auto gen = gens_.find(session);
      if (gen != gens_.end()) {
        GenPtr failed = gen->second;
        complete_generation_(failed, FinishReason::Failed);
      }
    }
    return true;
  }

  for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
    if (!is_running_()) {
      break; // an earlier output callback requested shutdown
    }
    const Input& input = batch.inputs[i];
    // The executor consumed this slice, so the session has moved past it
    // whether or not its generation is still alive. A cancelled generation
    // therefore leaves the session at the right place rather than drifting
    // back to where its first chunk started.
    advance_session_(input.sid, input.size);
    auto gen = gens_.find(input.sid);
    if (gen == gens_.end()) {
      continue;
    }
    GenPtr current = gen->second;
    if (current->state->cancelled.load()) {
      complete_generation_(current, FinishReason::Cancelled);
      continue;
    }
    if (input.produce_output) {
      handle_output_(current, std::move(out.outputs[i]));
    }
  }
  return true;
}

void RunnerImpl::advance_session_(SessionId session, std::size_t consumed) {
  auto it = open_sessions_.find(session);
  if (it != open_sessions_.end()) {
    // Admission bounded the generation's whole range inside Position, so
    // summing the slices it was split into cannot overflow.
    it->second += static_cast<Position>(consumed);
  }
}

void RunnerImpl::set_session_position_(SessionId session, Position position) {
  auto it = open_sessions_.find(session);
  if (it != open_sessions_.end()) {
    it->second = position;
  }
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

std::future<void> RunnerImpl::close_session(SessionId session) {
  auto ack = std::make_shared<std::promise<void>>();
  std::future<void> f = ack->get_future();
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (lifecycle_.load(std::memory_order_relaxed) != Lifecycle::Running) {
      ack->set_value();
      return f;
    }
    Command cmd;
    cmd.kind = Command::Kind::Close;
    cmd.session = session;
    cmd.close_ack = std::move(ack);
    inbox_.push_back(std::move(cmd));
  }
  notify_engine_();
  return f;
}

// --- generations -----------------------------------------------------------

GenerationHandle RunnerImpl::generate_async(
    SessionId session,
    std::vector<Token> delta,
    GenConfig config,
    GenerationCallback on_update) {
  auto gen = std::make_shared<GenState>();
  gen->session = session;
  gen->delta = std::make_shared<const std::vector<Token>>(std::move(delta));
  gen->config = std::move(config);
  gen->on_update = std::move(on_update);

  GenerationHandle handle(gen->state);
  bool admitted = false;
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Running) {
      Command cmd;
      cmd.kind = Command::Kind::Start;
      cmd.gen = gen;
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
  complete_generation_(gen, FinishReason::Cancelled);
  return handle;
}

void RunnerImpl::start_generation_(GenPtr gen) {
  if (!is_running_() || gen->state->cancelled.load()) {
    complete_generation_(gen, FinishReason::Cancelled);
    return;
  }
  auto session = open_sessions_.find(gen->session);
  if (session == open_sessions_.end()) {
    complete_generation_(gen, FinishReason::Failed);
    return;
  }
  // Where the session left off, which is 0 for one that has just opened.
  const Position start_position = session->second;
  if (gen->config.max_new_tokens <= 0 ||
      !valid_positioned_tokens(start_position, gen->delta)) {
    complete_generation_(gen, FinishReason::Failed);
    return;
  }
  // A step on this session failed mid-execute, so what the executor holds for
  // it is unknown. Anything built on that would be silently wrong.
  if (poisoned_.count(gen->session) != 0) {
    complete_generation_(gen, FinishReason::Failed);
    return;
  }
  if (gens_.count(gen->session) != 0) {
    complete_generation_(gen, FinishReason::Failed); // one generation per session
    return;
  }

  executor_.set_sampling_seed(gen->session, gen->config.seed);

  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (lifecycle_.load(std::memory_order_relaxed) != Lifecycle::Running) {
      // Sampling began before the stop transition, but no task was submitted.
    } else {
      gens_[gen->session] = gen;
      gen->registered = true;
    }
  }
  if (!gen->registered) {
    complete_generation_(gen, FinishReason::Cancelled);
    return;
  }
  submit_(
      gen,
      create_tasks_(
          gen, gen->delta, start_position, /*is_continuation=*/false));
}

std::vector<Task> RunnerImpl::create_tasks_(
    const GenPtr& gen,
    std::shared_ptr<const std::vector<Token>> tokens,
    Position position,
    bool is_continuation) {
  const auto total = static_cast<std::int32_t>(tokens->size());
  const std::int32_t chunk = config_.max_prefill_chunk_size;
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
    t.input.sid = gen->session;
    t.input.produce_output = last;
    t.input.offset = static_cast<size_t>(i);
    t.input.size = static_cast<size_t>(n);
    t.input.tokens = tokens;
    t.input.position = position;
    t.input.sampling_params = gen->config.sampling;
    tasks.push_back(std::move(t));
  }
  return tasks;
}

void RunnerImpl::submit_(const GenPtr& gen, std::vector<Task> tasks) {
  bool running = false;
  bool accepted = false;
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    running =
        lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Running;
    if (running && !tasks.empty()) {
      accepted = scheduler_.submit(std::move(tasks));
    }
  }
  if (!accepted) {
    complete_generation_(
        gen, running ? FinishReason::Failed : FinishReason::Cancelled);
  }
}

void RunnerImpl::handle_output_(
    const GenPtr& gen_ptr,
    std::optional<Output> output) {
  GenState& gen = *gen_ptr;
  if (gen.done) {
    return;
  }
  if (!output || !output->tokens || output->tokens->empty()) {
    complete_generation_(gen_ptr, FinishReason::Failed);
    return;
  }

  // A run can hit a stop token or the budget part way through, since a
  // speculative executor answers with several tokens, so take the prefix up to
  // whichever comes first and emit it in one call.
  const auto& stops = gen.config.stop_tokens;
  const std::int32_t room = gen.config.max_new_tokens - gen.generated;
  std::vector<Token> emit;
  FinishReason reason = FinishReason::NewTokenLimit;
  bool ends = false;

  for (Token token : *output->tokens) {
    if (std::find(stops.begin(), stops.end(), token) != stops.end()) {
      reason = FinishReason::StopToken;
      ends = true;
      break;
    }
    emit.push_back(token);
    if (static_cast<std::int32_t>(emit.size()) >= room) {
      ends = true;
      break;
    }
  }
  gen.generated += static_cast<std::int32_t>(emit.size());
  if (ends) {
    complete_generation_(gen_ptr, reason, std::move(emit));
    return;
  }
  if (!emit.empty()) {
    deliver_update_(gen_ptr, emit, std::nullopt);
  }
  if (!is_running_()) {
    return; // callback requested shutdown; final cleanup cancels this generation
  }

  // The executor says where the session continues, so its answer replaces the
  // slice-by-slice tally: only the executor knows what a rejected speculative
  // round rolled back.
  if (!output->next ||
      !valid_positioned_tokens(
          output->next->position, output->next->tokens)) {
    complete_generation_(gen_ptr, FinishReason::Failed);
    return;
  }
  set_session_position_(gen.session, output->next->position);
  submit_(
      gen_ptr,
      create_tasks_(
          gen_ptr,
          output->next->tokens,
          output->next->position,
          /*is_continuation=*/true));
}

void RunnerImpl::deliver_update_(
    const GenPtr& gen,
    const std::vector<Token>& tokens,
    std::optional<FinishReason> reason) {
  if (gen->on_update) {
    // TODO: Dispatch user callbacks through a callback pool.
    gen->on_update(tokens, reason);
  }
}

void RunnerImpl::complete_generation_(
    const GenPtr& gen_ptr,
    FinishReason reason,
    std::vector<Token> final_tokens) {
  GenState& gen = *gen_ptr;
  bool cancel_queued = false;
  {
    std::lock_guard<std::mutex> lock(control_mutex_);
    if (gen.done) {
      return;
    }
    if (lifecycle_.load(std::memory_order_relaxed) != Lifecycle::Running) {
      reason = FinishReason::Cancelled;
      final_tokens.clear();
    }
    gen.done = true;
    if (gen.registered) {
      gens_.erase(gen.session);
      gen.registered = false;
      cancel_queued = true;
    }
  }
  if (cancel_queued) {
    // Anything of this generation still queued would otherwise reach a batch.
    for (Task& t : scheduler_.cancel(gen.session)) {
      (void)t;
    }
  }
  {
    std::lock_guard<std::mutex> lock(gen.state->mutex);
    gen.state->reason = reason;
    gen.state->done = true;
    // notify_all: every waiter must observe the end, and no signal follows.
    gen.state->cv.notify_all();
  }
  // Handle state is visible before user code inspects it from the callback.
  deliver_update_(gen_ptr, final_tokens, reason);
}

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
