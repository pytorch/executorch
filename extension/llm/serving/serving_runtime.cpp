/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/serving/serving_runtime.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include <executorch/extension/llm/batching/runner.h>
#include <executorch/extension/llm/serving/detail/generation_bridge.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace extension {
namespace llm {
namespace serving {

namespace detail {

struct RequestState {
  enum class Delivery { Streaming, Settling, Finalizing };

  RequestState(
      RequestId id,
      GenerationRequest request,
      const ServingRuntimeConfig& limits)
      : id(id),
        request(std::move(request)),
        event_limit(limits.max_events_per_request),
        token_limit(limits.max_tokens_per_request) {}

  void cancel() {
    batching::GenerationHandle handle;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (terminal || done) {
        return;
      }
      cancelled.store(true);
      handle = engine;
    }
    handle.cancel();
  }

  void bind(batching::GenerationHandle handle = {}) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      engine = handle;
      bound = true;
      if (!handle.valid()) {
        settled = true;
      }
    }
    if (cancelled.load()) {
      handle.cancel();
    }
  }

  void emit(const batching::GenerationUpdate& update) {
    batching::GenerationHandle cancel_handle;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (terminal) {
        return;
      }
      if ((!update.finish_reason && updates.size() == event_limit) ||
          update.tokens.size() > token_limit - queued_tokens) {
        error = ServingError{
            ErrorCode::CapacityExceeded, "request output queue overflow"};
        terminal = batching::GenerationUpdate{
            {}, batching::FinishReason::Failed, error->message};
        updates.clear();
        queued_tokens = 0;
        cancelled.store(true);
        cancel_handle = engine;
      } else {
        queued_tokens += update.tokens.size();
        if (update.finish_reason) {
          terminal = update;
          if (cancelled.load() &&
              *terminal->finish_reason != batching::FinishReason::Failed) {
            terminal->finish_reason = batching::FinishReason::Cancelled;
          }
          if (*terminal->finish_reason == batching::FinishReason::Failed &&
              !error) {
            error = ServingError{ErrorCode::Internal, terminal->error_message};
          }
        } else {
          updates.push_back(update);
        }
      }
    }
    cancel_handle.cancel();
  }

  void reject(std::optional<ServingError> failure = std::nullopt) {
    batching::GenerationUpdate update;
    update.finish_reason = failure ? batching::FinishReason::Failed
                                   : batching::FinishReason::Cancelled;
    if (failure) {
      update.error_message = failure->message;
      std::lock_guard<std::mutex> lock(mutex);
      error = std::move(failure);
    }
    bind();
    emit(update);
  }

  void sink_failed() {
    batching::GenerationHandle handle;
    {
      std::lock_guard<std::mutex> lock(mutex);
      error = ServingError{ErrorCode::Internal, "request sink threw"};
      updates.clear();
      queued_tokens = 0;
      terminal = batching::GenerationUpdate{
          {}, batching::FinishReason::Failed, error->message};
      cancelled.store(true);
      handle = engine;
    }
    handle.cancel();
  }

  const RequestId id;
  GenerationRequest request;
  const std::size_t event_limit;
  const std::size_t token_limit;
  std::atomic<bool> cancelled{false};
  mutable std::mutex mutex;
  std::condition_variable cv;
  batching::GenerationHandle engine;
  bool bound = false;
  bool settled = false;
  bool done = false;
  bool finalized = false;
  // Queue membership belongs to the runtime mutex; delivery state to its
  // thread.
  bool queued = false;
  Delivery delivery = Delivery::Streaming;
  bool sink_ok = true;
  std::size_t queued_tokens = 0;
  std::deque<batching::GenerationUpdate> updates;
  std::optional<batching::GenerationUpdate> terminal;
  std::optional<ServingError> error;
  // Transferred through preparation and control finalization; immutable once
  // finalized is published, until delivery cleanup. Engine callbacks and
  // handles never read it.
  GenerationCompletion completion;
};

} // namespace detail

struct ServingRuntime::Impl {
  using Request = std::shared_ptr<detail::RequestState>;
  using SessionKey = std::variant<std::string, RequestId>;
  enum class Operation { Open, Close, Reset, Generate };
  enum class Lifecycle { Running, Stopping, Stopped };
  enum class SessionPhase { Opening, Open, Reopening, Unavailable };

  struct Slot {
    SessionPhase phase = SessionPhase::Opening;
    std::optional<batching::Session> session;
    std::uint64_t incarnation = 0;
    RequestId active_request = 0;
  };

  struct Command {
    Operation operation;
    std::string key;
    // Queue transfers must not destroy user captures under mutex_.
    std::unique_ptr<LifecycleCallback> completion;
    Request request;
  };

  Impl(
      batching::Executor& executor,
      std::unique_ptr<batching::Scheduler> scheduler,
      ServingRuntimeConfig config)
      : config_(config),
        valid_config_(
            scheduler && config.max_sessions > 0 &&
            config.max_pending_operations > 0 && config.max_requests > 0 &&
            config.max_events_per_request > 0 &&
            config.max_tokens_per_request > 0) {
    if (valid_config_) {
      runner_ =
          std::make_unique<batching::Runner>(executor, std::move(scheduler));
#if ET_HAS_EXCEPTIONS
      try {
#endif
        dispatcher_ = std::thread([this] { dispatch(); });
        control_ = std::thread([this] { run(); });
#if ET_HAS_EXCEPTIONS
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          dispatch_stopping_ = true;
        }
        dispatch_cv_.notify_all();
        if (dispatcher_.joinable()) {
          dispatcher_.join();
        }
        throw;
      }
#endif
    }
  }

  static void complete(
      std::unique_ptr<LifecycleCallback> completion,
      LifecycleResult result) {
    if (!completion || !*completion) {
      return;
    }
#if ET_HAS_EXCEPTIONS
    try {
#endif
      (*completion)(std::move(result));
#if ET_HAS_EXCEPTIONS
    } catch (...) {
      ET_LOG(Error, "Lifecycle completion callback threw");
    }
#endif
  }

  std::future<LifecycleResult> submit(Operation operation, std::string key) {
    auto completion = std::make_shared<std::promise<LifecycleResult>>();
    auto future = completion->get_future();
    submit(
        operation,
        std::move(key),
        [completion = std::move(completion)](LifecycleResult result) {
          completion->set_value(std::move(result));
        });
    return future;
  }

  void
  submit(Operation operation, std::string key, LifecycleCallback on_complete) {
    Command command{
        operation,
        std::move(key),
        std::make_unique<LifecycleCallback>(std::move(on_complete)),
        {}};
    LifecycleResult error;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!valid_config_) {
        error =
            ServingError{ErrorCode::InvalidArgument, "invalid runtime config"};
      } else if (
          lifecycle_ != Lifecycle::Running ||
          runner_->initialization_state() ==
              batching::InitializationState::Failed) {
        error = ServingError{ErrorCode::NotReady, "runtime is not ready"};
      } else if (command.key.empty()) {
        error =
            ServingError{ErrorCode::InvalidArgument, "session key is empty"};
      } else if (outstanding_ == config_.max_pending_operations) {
        error = ServingError{
            ErrorCode::CapacityExceeded, "lifecycle operation limit reached"};
      }
      if (!error) {
        inbox_.push_back(std::move(command));
        ++outstanding_;
      }
    }
    if (error) {
      complete(std::move(command.completion), std::move(error));
    } else {
      cv_.notify_one();
    }
  }

  LifecycleResult process(Command& command) {
    std::optional<batching::Session> retired;
    Request cancelled;
    std::unique_lock<std::mutex> lock(mutex_);
    if (lifecycle_ != Lifecycle::Running) {
      return ServingError{ErrorCode::NotReady, "runtime is stopping"};
    }
    auto it = sessions_.find(command.key);
    if (command.operation == Operation::Close) {
      if (it != sessions_.end()) {
        if (it->second.active_request) {
          cancelled = requests_.at(it->second.active_request);
        }
        retired = std::move(it->second.session);
        sessions_.erase(it);
      }
      lock.unlock();
      if (cancelled) {
        cancelled->cancel();
      }
      retired.reset();
      return std::nullopt;
    }
    const bool opening = command.operation == Operation::Open;
    if (opening && it != sessions_.end()) {
      if (it->second.phase == SessionPhase::Open) {
        return std::nullopt;
      }
      return ServingError{
          ErrorCode::NotReady,
          "session is unavailable; reset to retry or close to release its slot"};
    }
    if (opening) {
      if (sessions_.size() == config_.max_sessions) {
        return ServingError{
            ErrorCode::CapacityExceeded, "session limit reached"};
      }
      it = sessions_.try_emplace(command.key).first;
      it->second.incarnation = next_incarnation_++;
    } else {
      if (it == sessions_.end()) {
        return ServingError{
            ErrorCode::SessionNotFound, "session key not found"};
      }
      it->second.phase = SessionPhase::Reopening;
      if (it->second.active_request) {
        cancelled = requests_.at(it->second.active_request);
      }
      it->second.active_request = 0;
      it->second.incarnation = next_incarnation_++;
      retired = std::move(it->second.session);
      it->second.session.reset();
    }

    // Only this thread changes registry entries. Keep the slot reserved while
    // waiting, but never hold the admission mutex across Runner calls or waits.
    lock.unlock();
    if (cancelled) {
      cancelled->cancel();
    }
    retired.reset();
    std::optional<batching::Session> opened;
    LifecycleResult error;
#if ET_HAS_EXCEPTIONS
    try {
#endif
      opened = runner_->open_session_async().get();
#if ET_HAS_EXCEPTIONS
    } catch (...) {
      error = ServingError{ErrorCode::Internal, "session open failed"};
    }
#endif
    lock.lock();
    if (lifecycle_ != Lifecycle::Running) {
      lock.unlock();
      opened.reset();
      return ServingError{ErrorCode::NotReady, "runtime is stopping"};
    }
    if (opened) {
      it->second.session = std::move(opened);
      it->second.phase = SessionPhase::Open;
      return std::nullopt;
    }
    if (opening) {
      sessions_.erase(it);
    } else {
      it->second.phase = SessionPhase::Unavailable;
    }
    const auto code = error ? error->code
        : runner_->initialization_state() ==
            batching::InitializationState::Failed
        ? ErrorCode::NotReady
        : ErrorCode::CapacityExceeded;
    return ServingError{
        code,
        opening
            ? "session open failed"
            : "session reopen failed; old state is gone and key is unavailable"};
  }

  detail::SubmissionResult submit(detail::GenerationRequest request) {
    Request state;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!valid_config_) {
        return ServingError{
            ErrorCode::InvalidArgument, "invalid runtime config"};
      }
      if (lifecycle_ != Lifecycle::Running ||
          runner_->initialization_state() ==
              batching::InitializationState::Failed) {
        return ServingError{ErrorCode::NotReady, "runtime is not ready"};
      }
      if ((request.key && request.key->empty()) || request.delta.empty() ||
          request.config.max_new_tokens <= 0) {
        return ServingError{
            ErrorCode::InvalidArgument, "invalid generation request"};
      }
      if (requests_.size() == config_.max_requests ||
          outstanding_ == config_.max_pending_operations) {
        return ServingError{
            ErrorCode::CapacityExceeded, "request admission limit reached"};
      }
      state = std::make_shared<detail::RequestState>(
          next_request_id_++, std::move(request), config_);
      state->completion.request_id = state->id;
      requests_.emplace(state->id, state);
      inbox_.push_back(Command{Operation::Generate, {}, {}, state});
      ++outstanding_;
    }
    cv_.notify_one();
    return RequestHandle(state);
  }

  void generate(const Request& request) {
    const SessionKey key = request->request.key
        ? SessionKey(*request->request.key)
        : SessionKey(request->id);
    std::unique_lock<std::mutex> lock(mutex_);
    if (lifecycle_ != Lifecycle::Running || request->cancelled.load()) {
      lock.unlock();
      request->reject();
      return;
    }
    auto it = sessions_.find(key);
    const bool opening = it == sessions_.end();
    if (!opening &&
        (it->second.phase != SessionPhase::Open || it->second.active_request)) {
      const bool busy = it->second.active_request != 0;
      lock.unlock();
      request->reject(ServingError{
          busy ? ErrorCode::SessionBusy : ErrorCode::NotReady,
          busy ? "session already has an active request"
               : "session is unavailable; reset first"});
      return;
    }
    if (opening) {
      if (sessions_.size() == config_.max_sessions) {
        lock.unlock();
        request->reject(
            ServingError{ErrorCode::CapacityExceeded, "session limit reached"});
        return;
      }
      it = sessions_.try_emplace(key).first;
      it->second.incarnation = next_incarnation_++;
    }
    it->second.active_request = request->id;
    request->completion.incarnation = it->second.incarnation;
    if (opening) {
      lock.unlock();
      std::optional<batching::Session> opened;
#if ET_HAS_EXCEPTIONS
      try {
#endif
        opened = runner_->open_session_async().get();
#if ET_HAS_EXCEPTIONS
      } catch (...) {
        // Refusal below also releases the initial reservation.
      }
#endif
      lock.lock();
      const bool cancelled =
          lifecycle_ != Lifecycle::Running || request->cancelled.load();
      if (!opened || cancelled) {
        sessions_.erase(it);
        const auto code = runner_->initialization_state() ==
                batching::InitializationState::Failed
            ? ErrorCode::NotReady
            : ErrorCode::CapacityExceeded;
        lock.unlock();
        opened.reset();
        request->reject(
            cancelled
                ? std::nullopt
                : std::optional<ServingError>({code, "session open failed"}));
        return;
      }
      it->second.session = std::move(opened);
      it->second.phase = SessionPhase::Open;
    }
    auto& session = *it->second.session;
    const bool cancelled =
        lifecycle_ != Lifecycle::Running || request->cancelled.load();
    lock.unlock();
    if (cancelled) {
      request->reject();
      return;
    }
    // The control thread alone owns Session objects. Rejection can invoke the
    // callback inline before this call returns; the mailbox needs no binding.
#if ET_HAS_EXCEPTIONS
    try {
#endif
      auto handle = session.generate_async(
          std::move(request->request.delta),
          std::move(request->request.config),
          [this, request](const batching::GenerationUpdate& update) {
            request->emit(update);
            schedule(request);
          },
          [this, weak = std::weak_ptr<detail::RequestState>(request)] {
            if (auto request = weak.lock()) {
              {
                std::lock_guard<std::mutex> lock(request->mutex);
                request->settled = true;
              }
              schedule(request);
            }
          });
      request->bind(std::move(handle));
#if ET_HAS_EXCEPTIONS
    } catch (...) {
      request->reject(
          ServingError{ErrorCode::Internal, "generation submission failed"});
    }
#endif
  }

  void finalize(const Request& request) {
    const SessionKey key = request->request.key
        ? SessionKey(*request->request.key)
        : SessionKey(request->id);
    bool owns_session = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = sessions_.find(key);
      owns_session = it != sessions_.end() &&
          it->second.incarnation == request->completion.incarnation &&
          it->second.active_request == request->id;
      request->completion.current_session =
          owns_session && lifecycle_ == Lifecycle::Running;
      if (request->completion.current_session) {
        request->completion.position = it->second.session->position();
      }
    }
    bool failed = false;
#if ET_HAS_EXCEPTIONS
    try {
#endif
      if (request->request.on_complete) {
        request->request.on_complete(request->completion);
      }
#if ET_HAS_EXCEPTIONS
    } catch (...) {
      failed = true;
      std::lock_guard<std::mutex> lock(request->mutex);
      request->error =
          ServingError{ErrorCode::Internal, "request commit hook threw"};
      request->completion.error = request->error;
      request->completion.terminal = {
          {}, batching::FinishReason::Failed, request->error->message};
    }
#endif
    std::optional<batching::Session> retired;
    if (owns_session) {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = sessions_.find(key);
      it->second.active_request = 0;
      if (!request->request.key || failed) {
        retired = std::move(it->second.session);
        it->second.session.reset();
        if (!request->request.key) {
          sessions_.erase(it);
        } else {
          it->second.phase = SessionPhase::Unavailable;
        }
      }
    }
    retired.reset();
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      // The session claim is released and completion/error are now fixed.
      request->finalized = true;
    }
    schedule(request);
  }

  void schedule(const Request& request) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!requests_.count(request->id) || request->queued) {
        return;
      }
      dispatch_queue_.push_back(request);
      request->queued = true;
    }
    dispatch_cv_.notify_one();
  }

  void deliver(const Request& request) {
    using Delivery = detail::RequestState::Delivery;
    const RequestHandle handle(request);
    if (request->delivery == Delivery::Streaming) {
      batching::GenerationUpdate update;
      {
        std::lock_guard<std::mutex> lock(request->mutex);
        if (!request->updates.empty()) {
          update = std::move(request->updates.front());
          request->updates.pop_front();
        } else {
          // A throwing engine callback can settle without a terminal update.
          if (!request->terminal && request->bound && request->settled) {
            request->terminal = batching::GenerationUpdate{
                {},
                request->engine.finish_reason(),
                request->engine.error_message()};
          }
          if (!request->terminal) {
            return;
          }
          update.tokens = std::move(request->terminal->tokens);
          request->delivery = Delivery::Settling;
        }
        request->queued_tokens -= update.tokens.size();
      }
#if ET_HAS_EXCEPTIONS
      try {
#endif
        if (request->sink_ok && request->request.on_update &&
            (request->delivery == Delivery::Streaming ||
             !update.tokens.empty())) {
          request->request.on_update(update, handle);
        }
#if ET_HAS_EXCEPTIONS
      } catch (...) {
        request->sink_ok = false;
        request->sink_failed();
      }
#endif
      schedule(request);
      return;
    }

    if (request->delivery == Delivery::Settling) {
      {
        std::lock_guard<std::mutex> lock(request->mutex);
        if (!request->bound || !request->settled) {
          return;
        }
        if (!request->error &&
            request->engine.finish_reason() == batching::FinishReason::Failed) {
          request->error = ServingError{
              ErrorCode::Internal, request->engine.error_message()};
          request->terminal = batching::GenerationUpdate{
              {}, batching::FinishReason::Failed, request->error->message};
        }
        request->completion.terminal = std::move(*request->terminal);
        request->completion.metrics = request->engine.metrics();
        request->completion.error = request->error;
      }
#if ET_HAS_EXCEPTIONS
      try {
#endif
        if (request->sink_ok && request->request.on_prepare_complete) {
          request->request.on_prepare_complete(request->completion);
        }
#if ET_HAS_EXCEPTIONS
      } catch (...) {
        std::lock_guard<std::mutex> lock(request->mutex);
        request->error =
            ServingError{ErrorCode::Internal, "request preparation threw"};
        request->completion.error = request->error;
        request->completion.terminal.finish_reason =
            batching::FinishReason::Failed;
        request->completion.terminal.error_message = request->error->message;
      }
#endif
      request->delivery = Delivery::Finalizing;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        finalizers_.push_back(request);
      }
      cv_.notify_one();
      return;
    }

    {
      std::lock_guard<std::mutex> lock(request->mutex);
      if (!request->finalized) {
        return;
      }
    }
    // Retire only at terminal dispatch, not control finalization: queued
    // terminals still count, leaving at most max_requests plus this envelope.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      requests_.erase(request->id);
    }
    cv_.notify_one();
#if ET_HAS_EXCEPTIONS
    try {
#endif
      if (request->sink_ok && request->request.on_update) {
        request->request.on_update(request->completion.terminal, handle);
      }
#if ET_HAS_EXCEPTIONS
    } catch (...) {
      ET_LOG(Error, "Terminal sink threw after request finalization");
    }
#endif
    // Callback quiescence includes capture destruction, outside runtime locks.
    request->request = {};
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      request->terminal.reset();
      request->completion = {};
      request->done = true;
    }
    request->cv.notify_all();
  }

  void dispatch() {
    for (;;) {
      Request request;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        dispatch_cv_.wait(lock, [this] {
          return dispatch_stopping_ || !dispatch_queue_.empty();
        });
        if (dispatch_queue_.empty()) {
          return;
        }
        request = std::move(dispatch_queue_.front());
        dispatch_queue_.pop_front();
        // Clear before delivery so a racing notification queues the next turn.
        request->queued = false;
        if (!requests_.count(request->id)) {
          continue;
        }
      }
      deliver(request);
    }
  }

  void run() {
    for (;;) {
      Command command;
      Request finalizer;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
          return !finalizers_.empty() || !inbox_.empty() ||
              (lifecycle_ != Lifecycle::Running && requests_.empty());
        });
        if (!finalizers_.empty()) {
          finalizer = std::move(finalizers_.front());
          finalizers_.pop_front();
        } else if (!inbox_.empty()) {
          command = std::move(inbox_.front());
          inbox_.pop_front();
        } else {
          break;
        }
      }
      if (finalizer) {
        finalize(finalizer);
        continue;
      }
      LifecycleResult result;
      if (command.operation == Operation::Generate) {
        generate(command.request);
        schedule(command.request);
      } else {
        result = process(command);
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        --outstanding_;
        if (command.operation != Operation::Generate &&
            lifecycle_ != Lifecycle::Running) {
          result = ServingError{ErrorCode::NotReady, "runtime is stopping"};
        }
      }
      complete(std::move(command.completion), std::move(result));
    }
    decltype(sessions_) retired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      retired.swap(sessions_);
    }
    retired.clear();
  }

  ServingInfo info() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {
        valid_config_ && lifecycle_ == Lifecycle::Running &&
            runner_->initialization_state() ==
                batching::InitializationState::Ready,
        config_.max_context_length,
        config_.max_sessions,
        sessions_.size()};
  }

  void shutdown() {
    std::vector<Request> cancelled;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (lifecycle_ != Lifecycle::Running) {
        stopped_cv_.wait(
            lock, [this] { return lifecycle_ == Lifecycle::Stopped; });
        return;
      }
      lifecycle_ = Lifecycle::Stopping;
      for (const auto& entry : requests_) {
        cancelled.push_back(entry.second);
      }
    }
    for (const auto& request : cancelled) {
      request->cancel();
    }
    cv_.notify_one();
    // Stop Runner before joining control: control may be awaiting an open.
    // Runner settles that future and closes even an unpublished late session.
    if (runner_) {
      runner_->shutdown();
      control_.join();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        dispatch_stopping_ = true;
      }
      dispatch_cv_.notify_all();
      // An empty registry can still have a terminal callback/cleanup in flight.
      dispatcher_.join();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      lifecycle_ = Lifecycle::Stopped;
    }
    stopped_cv_.notify_all();
  }

  const ServingRuntimeConfig config_;
  const bool valid_config_;
  std::unique_ptr<batching::Runner> runner_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable stopped_cv_;
  Lifecycle lifecycle_ = Lifecycle::Running;
  std::size_t outstanding_ = 0;
  std::deque<Command> inbox_;
  std::unordered_map<SessionKey, Slot> sessions_;
  std::uint64_t next_incarnation_ = 1;
  RequestId next_request_id_ = 1;
  std::unordered_map<RequestId, Request> requests_;
  std::deque<Request> dispatch_queue_;
  std::deque<Request> finalizers_;
  std::condition_variable dispatch_cv_;
  bool dispatch_stopping_ = false;
  std::thread dispatcher_;
  std::thread control_;
};

RequestHandle::RequestHandle(std::shared_ptr<detail::RequestState> state)
    : state_(std::move(state)) {}

RequestId RequestHandle::id() const noexcept {
  return state_ ? state_->id : 0;
}

void RequestHandle::cancel() const {
  if (state_) {
    state_->cancel();
  }
}

bool RequestHandle::done() const {
  if (!state_) {
    return false;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->done;
}

void RequestHandle::wait() const {
  if (state_) {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->cv.wait(lock, [this] { return state_->done; });
  }
}

std::optional<ServingError> RequestHandle::error() const {
  if (!state_) {
    return std::nullopt;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->error;
}

detail::SubmissionResult detail::GenerationBridge::submit(
    ServingRuntime& runtime,
    GenerationRequest request) {
  return runtime.impl_->submit(std::move(request));
}

ServingRuntime::ServingRuntime(
    batching::Executor& executor,
    std::unique_ptr<batching::Scheduler> scheduler,
    ServingRuntimeConfig config)
    : impl_(std::make_unique<Impl>(executor, std::move(scheduler), config)) {}

ServingRuntime::~ServingRuntime() {
  shutdown();
}

void ServingRuntime::open_session_async(
    std::string key,
    LifecycleCallback on_complete) {
  impl_->submit(Impl::Operation::Open, std::move(key), std::move(on_complete));
}

std::future<LifecycleResult> ServingRuntime::open_session_async(
    std::string key) {
  return impl_->submit(Impl::Operation::Open, std::move(key));
}

void ServingRuntime::close_session_async(
    std::string key,
    LifecycleCallback on_complete) {
  impl_->submit(Impl::Operation::Close, std::move(key), std::move(on_complete));
}

std::future<LifecycleResult> ServingRuntime::close_session_async(
    std::string key) {
  return impl_->submit(Impl::Operation::Close, std::move(key));
}

void ServingRuntime::reset_session_async(
    std::string key,
    LifecycleCallback on_complete) {
  impl_->submit(Impl::Operation::Reset, std::move(key), std::move(on_complete));
}

std::future<LifecycleResult> ServingRuntime::reset_session_async(
    std::string key) {
  return impl_->submit(Impl::Operation::Reset, std::move(key));
}

ServingInfo ServingRuntime::info() const {
  return impl_->info();
}

void ServingRuntime::shutdown() {
  impl_->shutdown();
}

} // namespace serving
} // namespace llm
} // namespace extension
} // namespace executorch
