/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/serving/serving_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include <executorch/extension/llm/batching/prefix_cache.h>
#include <executorch/extension/llm/batching/runner.h>
#include <executorch/extension/llm/serving/detail/generation_bridge.h>
#include <executorch/extension/llm/serving/detail/prompt_preparer.h>
#include <executorch/extension/llm/serving/detail/text_output.h>
#include <executorch/extension/llm/serving/prompt_history.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace extension {
namespace llm {
namespace serving {

namespace detail {

struct TextRequest {
  PromptInput input;
  GenerationOptions options;
  std::function<void(GenerationEvent)> sink;
  std::vector<batching::Token> prompt;
  std::vector<batching::Token> raw_tokens;
  std::vector<batching::Token> history;
  std::unique_ptr<TextOutput> output;
  std::optional<ServingError> render_error;
  // Completed on control, then transferred unchanged by terminal delivery.
  TerminalEvent terminal;
  batching::MetricsTime submitted = batching::MetricsClock::now();
  std::size_t start_position = 0;
  bool started = false;
  bool clone_lane = false;
  std::optional<batching::PrefixCache::PromptCapture> prefix_capture;

  void emit(GenerationEvent event) {
    if (!sink) {
      return;
    }
#if ET_HAS_EXCEPTIONS
    try {
#endif
      sink(std::move(event));
#if ET_HAS_EXCEPTIONS
    } catch (...) {
      // Flush runs in the preparation hook, outside the raw sink's guard.
      // Disable user output here so neither path invokes a throwing sink again.
      sink = {};
      throw;
    }
#endif
  }
};

struct RequestState {
  enum class Delivery { Streaming, Settling, Finalizing };

  RequestState(GenerationRequest&& request, const ServingRuntimeConfig& limits)
      : fence_key(request.key),
        request(std::move(request)),
        event_limit(limits.max_events_per_request),
        token_limit(limits.max_tokens_per_request) {
    // libc++ can retain inline callable copies after move. Empty the runtime
    // staging request before admission so cleanup covers every owned callback.
    request = {};
  }

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

  // Assigned once under admission locking, before publication.
  RequestId id = 0;
  // Remains immutable while delivery destroys the callback-owning request.
  const std::optional<std::string> fence_key;
  GenerationRequest request;
  std::shared_ptr<TextRequest> text;
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
    std::vector<batching::Token> logical_history;
    bool dirty = false;
  };

  struct Command {
    Operation operation;
    std::string key;
    // Queue transfers must not destroy user captures under mutex_.
    std::unique_ptr<LifecycleCallback> completion;
    Request request;
    RequestId fence_through = 0;
    bool processed = false;
    LifecycleResult result = std::nullopt;
  };

  Impl(
      batching::Executor& executor,
      std::unique_ptr<batching::Scheduler> scheduler,
      ServingRuntimeConfig config,
      const tokenizers::Tokenizer* tokenizer = nullptr)
      : config_(config),
        valid_config_(
            scheduler && config.max_sessions > 0 &&
            config.max_pending_operations > 0 && config.max_requests > 0 &&
            config.max_events_per_request > 0 &&
            config.max_tokens_per_request > 0),
        tokenizer_(tokenizer),
        prefix_cache_(config.prefix_cache_capacity) {
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
        command.fence_through = next_request_id_ - 1;
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
      it->second.logical_history.clear();
      it->second.dirty = false;
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

  detail::SubmissionResult submit(
      detail::GenerationRequest&& request,
      std::shared_ptr<detail::TextRequest> text = {}) {
    // Even moving an inline std::function can copy a user callable. Construct
    // and unwind callback ownership outside admission locking.
    auto state =
        std::make_shared<detail::RequestState>(std::move(request), config_);
    state->text = std::move(text);
    if (state->text) {
      wire_text(state);
    }
    Command command{
        Operation::Generate, state->fence_key.value_or(""), {}, state};
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
      if (state->text && !tokenizer_) {
        return ServingError{
            ErrorCode::NotReady, "text generation requires a tokenizer"};
      }
      if ((state->fence_key && state->fence_key->empty()) ||
          (!state->text &&
           (state->request.delta.empty() ||
            state->request.config.max_new_tokens <= 0))) {
        return ServingError{
            ErrorCode::InvalidArgument, "invalid generation request"};
      }
      if (requests_.size() == config_.max_requests ||
          outstanding_ == config_.max_pending_operations) {
        return ServingError{
            ErrorCode::CapacityExceeded, "request admission limit reached"};
      }
      state->id = next_request_id_++;
      state->completion.request_id = state->id;
      requests_.emplace(state->id, state);
#if ET_HAS_EXCEPTIONS
      try {
#endif
        inbox_.push_back(std::move(command));
#if ET_HAS_EXCEPTIONS
      } catch (...) {
        requests_.erase(state->id);
        throw;
      }
#endif
      ++outstanding_;
    }
    cv_.notify_one();
    return RequestHandle(state);
  }

  void complete_text(
      detail::TextRequest& text,
      const std::optional<std::string>& key,
      const detail::GenerationCompletion& completion) {
    auto& terminal = text.terminal;
    terminal.error = completion.error;
    if (terminal.error) {
      terminal.finish_reason = FinishReason::Failed;
    } else if (text.output && text.output->stopped()) {
      terminal.finish_reason = FinishReason::Stop;
    } else {
      switch (*completion.terminal.finish_reason) {
        case batching::FinishReason::StopToken:
          terminal.finish_reason = FinishReason::Stop;
          break;
        case batching::FinishReason::NewTokenLimit:
          terminal.finish_reason = FinishReason::Length;
          break;
        case batching::FinishReason::Cancelled:
          terminal.finish_reason = FinishReason::Cancelled;
          break;
        case batching::FinishReason::Failed:
          terminal.finish_reason = FinishReason::Failed;
          terminal.error = ServingError{
              ErrorCode::Internal, completion.terminal.error_message};
          break;
      }
    }
    auto& stats = terminal.stats;
    stats.completion_tokens =
        text.output ? text.output->completion_tokens() : 0;
    stats.prefill_ms = completion.metrics.prefill_span_us() / 1000.0;
    stats.decode_ms = completion.metrics.decode_span_us() / 1000.0;
    stats.total_ms = std::chrono::duration<double, std::milli>(
                         batching::MetricsClock::now() - text.submitted)
                         .count();
    if (text.started) {
      // The carried pending token is not physically resident: charge its
      // forward pass as prompt work, rather than counting it as free reuse.
      stats.reused_prompt_tokens =
          std::min(text.start_position, stats.prompt_tokens);
      stats.prefilled_prompt_tokens = completion.metrics.n_prefilled_tokens;
    }
    const bool completed = terminal.finish_reason == FinishReason::Stop ||
        terminal.finish_reason == FinishReason::Length;
    stats.generated_token_ids = completed && text.output
        ? text.output->generated_token_ids()
        : std::nullopt;
    if (!completion.current_session || !text.started) {
      return;
    }
    // Execution can finish a batch after cancellation retires its logical
    // result. Physical work alone is not proof that Runner retained the prompt.
    const bool full_prompt = stats.prompt_tokens > 0 &&
        (completion.metrics.n_generated_tokens > 0 ||
         (completion.position &&
          *completion.position >=
              static_cast<batching::Position>(stats.prompt_tokens)));
    const bool all_output =
        static_cast<std::size_t>(completion.metrics.n_generated_tokens) ==
        text.raw_tokens.size();
    const SessionKey session_key =
        key ? SessionKey(*key) : SessionKey(completion.request_id);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session_key);
    if (it == sessions_.end() ||
        it->second.incarnation != completion.incarnation ||
        it->second.active_request != completion.request_id) {
      return;
    }
    if (full_prompt &&
        text.history.size() == stats.prompt_tokens + text.raw_tokens.size()) {
      it->second.logical_history = std::move(text.history);
    } else {
      it->second.logical_history.clear();
    }
    it->second.dirty = !completed || !full_prompt || !all_output ||
        (text.output && text.output->string_stopped());
  }

  void wire_text(const Request& request) {
    const auto text = request->text;
    request->request.on_update = [text](
                                     const batching::GenerationUpdate& update,
                                     const RequestHandle& handle) {
      if (update.finish_reason) {
        text->emit(std::move(text->terminal));
        return;
      }
      text->raw_tokens.insert(
          text->raw_tokens.end(), update.tokens.begin(), update.tokens.end());
      if (text->output && !text->render_error) {
        if (text->output->append(update.tokens) != runtime::Error::Ok) {
          text->render_error =
              ServingError{ErrorCode::Internal, "token decoding failed"};
          handle.cancel();
        } else if (text->output->stopped()) {
          handle.cancel();
        }
      }
    };
    request->request.on_prepare_complete =
        [text](const detail::GenerationCompletion&) {
          text->history = std::move(text->prompt);
          text->history.insert(
              text->history.end(),
              text->raw_tokens.begin(),
              text->raw_tokens.end());
          if (text->output) {
            text->output->finish();
          }
        };
    request->request.on_complete =
        [this, text, key = request->request.key](
            const detail::GenerationCompletion& completion) {
          complete_text(*text, key, completion);
        };
  }

  LifecycleResult prepare_text(const Request& request) {
    auto& text = *request->text;
    const SessionKey key = request->request.key
        ? SessionKey(*request->request.key)
        : SessionKey(request->id);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (lifecycle_ != Lifecycle::Running || request->cancelled.load()) {
        return std::nullopt;
      }
      auto it = sessions_.find(key);
      if (it != sessions_.end() && it->second.active_request) {
        return ServingError{
            ErrorCode::SessionBusy, "session already has an active request"};
      }
      if (it != sessions_.end() && it->second.phase != SessionPhase::Open) {
        return ServingError{
            ErrorCode::NotReady, "session is unavailable; reset first"};
      }
    }
    const auto& options = text.options;
    if ((options.max_new_tokens && *options.max_new_tokens <= 0) ||
        !std::isfinite(options.sampling.temperature) ||
        options.sampling.temperature < 0 ||
        !std::isfinite(options.sampling.top_p) || options.sampling.top_p <= 0 ||
        options.sampling.top_p > 1 || options.sampling.top_k < 0 ||
        std::any_of(
            options.stop_strings.begin(),
            options.stop_strings.end(),
            [](const std::string& stop) { return stop.empty(); })) {
      return ServingError{
          ErrorCode::InvalidArgument, "invalid generation options"};
    }
    const auto position_limit = static_cast<std::size_t>(
        std::numeric_limits<batching::Position>::max());
    const auto context_limit = config_.max_context_length
        ? std::min(config_.max_context_length, position_limit)
        : position_limit;
    auto prepared =
        detail::prepare_prompt(*tokenizer_, text.input, context_limit);
    if (!prepared.ok()) {
      return ServingError{
          ErrorCode::InvalidArgument, "prompt preparation failed"};
    }
    text.prompt = std::move(prepared->tokens);
    text.input = {};
    text.terminal.stats.prompt_tokens = text.prompt.size();
    const auto available = context_limit - text.prompt.size();
    const auto wanted = options.max_new_tokens.value_or(
        config_.max_context_length
            ? static_cast<std::int32_t>(
                  std::min(available, static_cast<std::size_t>(INT32_MAX)))
            : config_.default_max_new_tokens);
    if (wanted <= 0 || available == 0) {
      return ServingError{
          ErrorCode::InvalidArgument, "no generation budget available"};
    }
    auto& config = request->request.config;
    config.max_new_tokens = static_cast<std::int32_t>(
        std::min(available, static_cast<std::size_t>(wanted)));
    config.sampling = options.sampling;
    config.seed = options.seed;
    config.stop_tokens = config_.default_stop_tokens;
    config.stop_tokens.insert(
        config.stop_tokens.end(),
        options.stop_tokens.begin(),
        options.stop_tokens.end());
    text.output = std::make_unique<detail::TextOutput>(
        *tokenizer_,
        text.prompt.back(),
        config.stop_tokens,
        options.stop_strings,
        [state = &text](const std::string& piece) {
          state->emit(TextEvent{piece});
        });

    PrefillPlan plan{PrefillPlan::kFull, 0, "new"};
    bool reset = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = sessions_.find(key);
      if (it != sessions_.end()) {
        plan = plan_prefill(
            it->second.logical_history, text.prompt, it->second.dirty);
        reset = plan.action == PrefillPlan::kFull &&
            (it->second.dirty || !it->second.logical_history.empty() ||
             it->second.session->position() != 0);
      }
    }
    text.terminal.stats.session_reset_reason = plan.reason;
    request->request.delta.assign(
        text.prompt.begin() + plan.suffix_start, text.prompt.end());
    // All fallible preparation precedes destructive cold replacement.
    if (request->cancelled.load()) {
      return std::nullopt;
    }
    if (reset) {
      Command command{Operation::Reset, *request->request.key, {}, {}};
      return process(command);
    }
    return std::nullopt;
  }

  std::optional<batching::PrefixMatch> lookup_prefix(const Request& request) {
    if (!request->text || config_.prefix_cache_capacity == 0 ||
        request->text->options.sampling.temperature != 0 || clone_lane_busy_) {
      return std::nullopt;
    }
    clone_lane_busy_ = true;
    request->text->clone_lane = true;
#if ET_HAS_EXCEPTIONS
    try {
#endif
      return prefix_cache_.lookup(request->text->prompt);
#if ET_HAS_EXCEPTIONS
    } catch (...) {
      return std::nullopt;
    }
#endif
  }

  void finish_capture(const Request& request) {
    if (!request->text || !request->text->clone_lane) {
      return;
    }
    // Generation has already finished on the dispatcher. collect() combines
    // clone wait and insertion; keep both on control, outside all request and
    // admission locks. Settle even a cancelled source's queued clone before
    // releasing the transient snapshot slot.
#if ET_HAS_EXCEPTIONS
    try {
#endif
      if (request->text->prefix_capture) {
        request->text->prefix_capture->collect();
      }
#if ET_HAS_EXCEPTIONS
    } catch (...) {
      // Optional cache maintenance must not change the generation outcome.
    }
#endif
    request->text->prefix_capture.reset();
    request->text->clone_lane = false;
    clone_lane_busy_ = false;
  }

  void generate(const Request& request) {
    if (request->text) {
      LifecycleResult error;
#if ET_HAS_EXCEPTIONS
      try {
#endif
        error = prepare_text(request);
#if ET_HAS_EXCEPTIONS
      } catch (...) {
        error = ServingError{ErrorCode::Internal, "prompt preparation failed"};
      }
#endif
      if (error) {
        request->reject(std::move(error));
        return;
      }
    }
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
      auto match = lookup_prefix(request);
      if (match) {
        opened = std::move(match->session);
        // Lookup always leaves the final prompt token for a fresh forward.
        request->request.delta.erase(
            request->request.delta.begin(),
            request->request.delta.begin() + match->matched_tokens);
      }
#if ET_HAS_EXCEPTIONS
      try {
#endif
        if (!opened && !request->cancelled.load()) {
          opened = runner_->open_session_async().get();
        }
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
    if (!request->text) {
      it->second.dirty = true;
    }
    const bool cancelled =
        lifecycle_ != Lifecycle::Running || request->cancelled.load();
    lock.unlock();
    if (cancelled) {
      request->reject();
      return;
    }
    if (request->text) {
      request->text->start_position =
          static_cast<std::size_t>(session.position());
      request->text->started = true;
    }
    // The control thread alone owns Session objects. Rejection can invoke the
    // callback inline before this call returns; the mailbox needs no binding.
#if ET_HAS_EXCEPTIONS
    try {
#endif
      batching::GenerationCallback callback =
          [this, request](const batching::GenerationUpdate& update) {
            request->emit(update);
            schedule(request);
          };
      if (request->text && request->text->clone_lane) {
#if ET_HAS_EXCEPTIONS
        try {
#endif
          request->text->prefix_capture =
              prefix_cache_.capture_prompt(session, request->text->prompt);
          callback = request->text->prefix_capture->wrap(callback);
#if ET_HAS_EXCEPTIONS
        } catch (...) {
          // Keep the original callback when optional capture setup fails.
        }
#endif
      }
      auto handle = session.generate_async(
          std::move(request->request.delta),
          std::move(request->request.config),
          std::move(callback),
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
    finish_capture(request);
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
    if (request->text && request->text->render_error &&
        !request->completion.error) {
      std::lock_guard<std::mutex> lock(request->mutex);
      request->error = request->text->render_error;
      request->completion.error = request->error;
      request->completion.terminal = {
          {}, batching::FinishReason::Failed, request->error->message};
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
      if (request->text) {
        auto& terminal = request->text->terminal;
        terminal.finish_reason = FinishReason::Failed;
        terminal.error = request->completion.error;
        terminal.stats.generated_token_ids.reset();
      }
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
      retiring_ = request;
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
    request->text.reset();
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      request->terminal.reset();
      request->completion = {};
      request->done = true;
    }
    request->cv.notify_all();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      retiring_.reset();
    }
    cv_.notify_one();
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

  static bool is_fence(const Command& command) {
    return command.operation == Operation::Close ||
        command.operation == Operation::Reset;
  }

  static bool precedes(const Request& request, const Command& command) {
    return request && request->fence_key &&
        *request->fence_key == command.key &&
        request->id <= command.fence_through;
  }

  // mutex_ held. Admission excludes the dispatcher's retiring envelope, but a
  // lifecycle acknowledgement must include its terminal and capture cleanup.
  bool fence_pending(const Command& command) const {
    return precedes(retiring_, command) ||
        std::any_of(requests_.begin(), requests_.end(), [&](const auto& entry) {
             return precedes(entry.second, command);
           });
  }

  void cancel_predecessors(const Command& command) {
    std::vector<Request> cancelled;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& entry : requests_) {
        if (precedes(entry.second, command)) {
          cancelled.push_back(entry.second);
        }
      }
      if (precedes(retiring_, command)) {
        cancelled.push_back(retiring_);
      }
    }
    for (const auto& request : cancelled) {
      request->cancel();
    }
  }

  // mutex_ held. A deferred fence blocks only subsequent commands for its key.
  // Scanning this bounded inbox avoids another queue or a polling wakeup.
  auto runnable_command() {
    for (auto it = inbox_.begin(); it != inbox_.end(); ++it) {
      if (!it->key.empty() &&
          std::any_of(inbox_.begin(), it, [&](const Command& earlier) {
            return earlier.key == it->key;
          })) {
        continue;
      }
      if (!it->processed || !fence_pending(*it)) {
        return it;
      }
    }
    return inbox_.end();
  }

  void run() {
    for (;;) {
      Command command;
      std::size_t command_index = 0;
      Request finalizer;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
          return !finalizers_.empty() || runnable_command() != inbox_.end() ||
              (lifecycle_ != Lifecycle::Running && inbox_.empty() &&
               requests_.empty() && !retiring_);
        });
        if (!finalizers_.empty()) {
          finalizer = std::move(finalizers_.front());
          finalizers_.pop_front();
        } else if (auto it = runnable_command(); it != inbox_.end()) {
          command_index = it - inbox_.begin();
          command = std::move(*it);
          inbox_.erase(it);
        } else {
          break;
        }
      }
      if (finalizer) {
        finalize(finalizer);
        continue;
      }
      if (!command.processed) {
        if (command.operation == Operation::Generate) {
          generate(command.request);
          schedule(command.request);
        } else {
          if (is_fence(command)) {
            cancel_predecessors(command);
          }
          command.result = process(command);
        }
        command.processed = true;
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_fence(command) && fence_pending(command)) {
          // Only control removes commands; concurrent submissions append, so
          // this position still precedes every later command for the key.
          inbox_.insert(inbox_.begin() + command_index, std::move(command));
          continue;
        }
        --outstanding_;
        if (command.operation != Operation::Generate &&
            lifecycle_ != Lifecycle::Running) {
          command.result =
              ServingError{ErrorCode::NotReady, "runtime is stopping"};
        }
      }
      complete(std::move(command.completion), std::move(command.result));
    }
    decltype(sessions_) retired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      retired.swap(sessions_);
    }
    retired.clear();
    prefix_cache_.clear();
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
  const tokenizers::Tokenizer* const tokenizer_;
  std::unique_ptr<batching::Runner> runner_;
  batching::PrefixCache prefix_cache_;
  // Control-thread policy; held through capture collection, including clone
  // refusal and cancellation. At most one transient snapshot can exist.
  bool clone_lane_busy_ = false;
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
  Request retiring_;
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

ServingRuntime::ServingRuntime(
    batching::Executor& executor,
    std::unique_ptr<batching::Scheduler> scheduler,
    const tokenizers::Tokenizer& tokenizer,
    ServingRuntimeConfig config)
    : impl_(std::make_unique<Impl>(
          executor,
          std::move(scheduler),
          config,
          &tokenizer)) {}

GenerateResult ServingRuntime::generate(
    std::optional<std::string> key,
    PromptInput prompt,
    GenerationOptions options,
    std::function<void(GenerationEvent)> on_event) {
  auto text = std::make_shared<detail::TextRequest>();
  text->input = std::move(prompt);
  text->options = std::move(options);
  // Unlike move, swap leaves no inline callable copy in the incoming sink.
  text->sink.swap(on_event);
  detail::GenerationRequest request;
  request.key = std::move(key);
  return impl_->submit(std::move(request), std::move(text));
}

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
