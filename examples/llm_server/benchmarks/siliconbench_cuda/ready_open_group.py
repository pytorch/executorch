"""Candidate: issue cold opens for already queued scratch requests before waiting."""
from pathlib import Path


def patch(source):
    source = source.replace('#include <cmath>', '#include <cmath>\n#include <cstdio>\n#include <vector>', 1)
    source = source.replace('  void generate(const Request& request) {',
                            '  void generate(const Request& request, bool prepared = false) {', 1)
    old = '  void generate(const Request& request, bool prepared = false) {\n    if (request->text) {'
    assert source.count(old) == 1
    source = source.replace(old, old.replace('if (request->text)', 'if (request->text && !prepared)'))
    anchor = '  void generate(const Request& request, bool prepared = false) {'
    helper = '''  void generate_ready_group(std::vector<Command>& group) {
    struct Pending {
      Request request;
      std::future<std::optional<batching::Session>> opening;
    };
    std::vector<Pending> pending;
    pending.reserve(group.size());
    std::fprintf(stderr, "ET_HTTP_READY_OPEN_GROUP %zu\\n", group.size());
    for (const auto& command : group) {
      const auto& request = command.request;
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
      if (error || request->cancelled.load()) {
        request->reject(std::move(error));
        schedule(request);
        continue;
      }
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (lifecycle_ != Lifecycle::Running || request->cancelled.load()) {
          lock.unlock();
          request->reject();
          schedule(request);
          continue;
        }
        if (sessions_.size() == config_.max_sessions) {
          lock.unlock();
          request->reject(ServingError{ErrorCode::CapacityExceeded,
              "session limit reached"});
          schedule(request);
          continue;
        }
        const SessionKey key(request->id);
        auto& slot = sessions_.try_emplace(key).first->second;
        slot.incarnation = next_incarnation_++;
      }
#if ET_HAS_EXCEPTIONS
      try {
#endif
        pending.push_back({request, runner_->open_session_async()});
#if ET_HAS_EXCEPTIONS
      } catch (...) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          sessions_.erase(SessionKey(request->id));
        }
        request->reject(ServingError{ErrorCode::Internal,
            "session preparation failed"});
        schedule(request);
      }
#endif
    }
    for (auto& item : pending) {
      std::optional<batching::Session> opened;
      LifecycleResult error;
#if ET_HAS_EXCEPTIONS
      try {
#endif
        opened = item.opening.get();
#if ET_HAS_EXCEPTIONS
      } catch (...) {
        error = ServingError{ErrorCode::Internal, "session preparation failed"};
      }
#endif
      const auto& request = item.request;
      const SessionKey key(request->id);
      bool cancelled;
      bool installed = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled = lifecycle_ != Lifecycle::Running || request->cancelled.load();
        if (!opened || cancelled || error) {
          sessions_.erase(key);
        } else {
          auto& slot = sessions_.at(key);
          slot.session = std::move(opened);
          slot.phase = SessionPhase::Open;
          slot.active_request = request->id;
          request->completion.incarnation = slot.incarnation;
          installed = true;
        }
      }
      if (!installed) {
        opened.reset();
        request->reject(cancelled ? std::nullopt : error ? std::move(error) :
            std::optional<ServingError>({ErrorCode::CapacityExceeded,
                "session open failed"}));
        schedule(request);
        continue;
      }
      generate(request, true);
      schedule(request);
    }
  }

'''
    source = source.replace(anchor, helper + anchor, 1)
    old_busy = '(it->second.phase != SessionPhase::Open || it->second.active_request))'
    assert source.count(old_busy) == 1
    source = source.replace(old_busy, '(it->second.phase != SessionPhase::Open ||\n         (it->second.active_request &&\n          !(prepared && it->second.active_request == request->id))))', 1)
    source = source.replace('      Command command;\n      std::size_t command_index = 0;',
                            '      Command command;\n      std::vector<Command> ready_group;\n      std::size_t command_index = 0;', 1)
    old = '          command = std::move(*it);\n          inbox_.erase(it);'
    assert source.count(old) == 1
    source = source.replace(old, old + '''
          const auto eligible = [](const Command& queued) {
            return queued.operation == Operation::Generate && !queued.processed &&
                queued.request->text && !queued.request->request.key;
          };
          if (config_.prefix_cache_capacity == 0 && eligible(command) &&
              !inbox_.empty() && eligible(inbox_.front())) {
            ready_group.push_back(std::move(command));
            while (!inbox_.empty() && eligible(inbox_.front()) &&
                   ready_group.size() < config_.max_sessions) {
              ready_group.push_back(std::move(inbox_.front()));
              inbox_.erase(inbox_.begin());
            }
          }''')
    old = '      if (!command.processed) {'
    assert source.count(old) == 1
    source = source.replace(old, '''      if (!ready_group.empty()) {
        generate_ready_group(ready_group);
        std::lock_guard<std::mutex> lock(mutex_);
        outstanding_ -= ready_group.size();
        continue;
      }
''' + old)
    return source
