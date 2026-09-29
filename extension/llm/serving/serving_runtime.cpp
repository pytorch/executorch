/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/serving/serving_runtime.h>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include <executorch/extension/llm/batching/runner.h>

namespace executorch {
namespace extension {
namespace llm {
namespace serving {

struct ServingRuntime::Impl {
  enum class Operation { Open, Close, Reset };
  enum class Lifecycle { Running, Stopping, Stopped };
  enum class SessionPhase { Opening, Open, Reopening, Unavailable };

  struct Slot {
    SessionPhase phase = SessionPhase::Opening;
    std::optional<batching::Session> session;
  };

  struct Command {
    Operation operation;
    std::string key;
    std::promise<LifecycleResult> completion;
  };

  Impl(
      batching::Executor& executor,
      std::unique_ptr<batching::Scheduler> scheduler,
      ServingRuntimeConfig config)
      : config_(config),
        valid_config_(
            scheduler && config.max_sessions > 0 &&
            config.max_pending_operations > 0) {
    if (valid_config_) {
      runner_ =
          std::make_unique<batching::Runner>(executor, std::move(scheduler));
      control_ = std::thread([this] { run(); });
    }
  }

  std::future<LifecycleResult> submit(Operation operation, std::string key) {
    Command command{operation, std::move(key), {}};
    auto future = command.completion.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      LifecycleResult error;
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
      if (error) {
        command.completion.set_value(std::move(error));
        return future;
      }
      inbox_.push_back(std::move(command));
      ++outstanding_;
    }
    cv_.notify_one();
    return future;
  }

  LifecycleResult process(Command& command) {
    std::optional<batching::Session> retired;
    std::unique_lock<std::mutex> lock(mutex_);
    if (lifecycle_ != Lifecycle::Running) {
      return ServingError{ErrorCode::NotReady, "runtime is stopping"};
    }
    auto it = sessions_.find(command.key);
    if (command.operation == Operation::Close) {
      if (it != sessions_.end()) {
        retired = std::move(it->second.session);
        sessions_.erase(it);
      }
      lock.unlock();
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
    } else {
      if (it == sessions_.end()) {
        return ServingError{
            ErrorCode::SessionNotFound, "session key not found"};
      }
      it->second.phase = SessionPhase::Reopening;
      retired = std::move(it->second.session);
      it->second.session.reset();
    }

    // Only this thread changes registry entries. Keep the slot reserved while
    // waiting, but never hold the admission mutex across Runner calls or waits.
    lock.unlock();
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

  void run() {
    for (;;) {
      Command command;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
          return lifecycle_ != Lifecycle::Running || !inbox_.empty();
        });
        if (inbox_.empty()) {
          break;
        }
        command = std::move(inbox_.front());
        inbox_.pop_front();
      }
      auto result = process(command);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (lifecycle_ != Lifecycle::Running) {
          result = ServingError{ErrorCode::NotReady, "runtime is stopping"};
        }
        --outstanding_;
        command.completion.set_value(std::move(result));
      }
    }
    decltype(sessions_) retired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      retired.swap(sessions_);
    }
    // Session destruction may enqueue Runner commands; keep it outside mutex_.
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
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (lifecycle_ != Lifecycle::Running) {
        stopped_cv_.wait(
            lock, [this] { return lifecycle_ == Lifecycle::Stopped; });
        return;
      }
      lifecycle_ = Lifecycle::Stopping;
    }
    cv_.notify_one();
    // Stop Runner before joining control: control may be awaiting an open.
    // Runner settles that future and closes even an unpublished late session.
    if (runner_) {
      runner_->shutdown();
      control_.join();
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
  std::unordered_map<std::string, Slot> sessions_;
  std::thread control_;
};

ServingRuntime::ServingRuntime(
    batching::Executor& executor,
    std::unique_ptr<batching::Scheduler> scheduler,
    ServingRuntimeConfig config)
    : impl_(std::make_unique<Impl>(executor, std::move(scheduler), config)) {}

ServingRuntime::~ServingRuntime() {
  shutdown();
}

std::future<LifecycleResult> ServingRuntime::open_session_async(
    std::string key) {
  return impl_->submit(Impl::Operation::Open, std::move(key));
}

std::future<LifecycleResult> ServingRuntime::close_session_async(
    std::string key) {
  return impl_->submit(Impl::Operation::Close, std::move(key));
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
