"""Generate an isolated optional Runner/ServingRuntime idle maintenance overlay."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

def replace(text, old, new):
    assert text.count(old)==1, 'Wrong source or duplicate patch anchor'
    return text.replace(old,new)

def transform(rel,text):
    if rel.endswith('/runner.h'):
        text=replace(text,'#include <cstdint>','#include <chrono>\n#include <cstdint>')
        text=replace(text,'class ET_EXPERIMENTAL Runner {',
'''struct ET_EXPERIMENTAL IdleMaintenanceConfig {
  std::chrono::milliseconds delay{100};
  // Engine-thread only, outside runner locks, after complete generations.
  // Return false to defer while a surrounding serving runtime remains busy.
  // Must not call back into the runner. Exceptions do not fail generations.
  std::function<bool()> callback;
};

class ET_EXPERIMENTAL Runner {''')
        text=replace(text,'  Runner(Executor& executor, std::unique_ptr<Scheduler> scheduler);',
'''  Runner(Executor& executor, std::unique_ptr<Scheduler> scheduler);
  Runner(
      Executor& executor,
      std::unique_ptr<Scheduler> scheduler,
      IdleMaintenanceConfig idle_maintenance);''')
    elif rel.endswith('/runner.cpp'):
        text=replace(text,
'''  RunnerImpl(Executor& executor, std::unique_ptr<Scheduler> scheduler)
      : executor_(executor), scheduler_(std::move(scheduler)) {''',
'''  RunnerImpl(
      Executor& executor,
      std::unique_ptr<Scheduler> scheduler,
      IdleMaintenanceConfig idle_maintenance)
      : executor_(executor),
        scheduler_(std::move(scheduler)),
        idle_maintenance_(std::move(idle_maintenance)) {''')
        text=replace(text,'  std::unique_ptr<Scheduler> scheduler_;',
'''  std::unique_ptr<Scheduler> scheduler_;
  const IdleMaintenanceConfig idle_maintenance_;''')
        text=replace(text,
'''Runner::Runner(Executor& executor, std::unique_ptr<Scheduler> scheduler)
    : impl_(std::make_shared<RunnerImpl>(executor, std::move(scheduler))) {
  impl_->start();
}''',
'''Runner::Runner(Executor& executor, std::unique_ptr<Scheduler> scheduler)
    : Runner(executor, std::move(scheduler), {}) {}

Runner::Runner(
    Executor& executor,
    std::unique_ptr<Scheduler> scheduler,
    IdleMaintenanceConfig idle_maintenance)
    : impl_(std::make_shared<RunnerImpl>(
          executor, std::move(scheduler), std::move(idle_maintenance))) {
  impl_->start();
}''')
        text=replace(text,
'''  while (is_running_()) {
    process_pending_commands_();
    reap_cancelled_();
    if (execute_one_batch_()) {
      continue; // more work may be ready
    }
    std::unique_lock<std::mutex> lock(control_mutex_);
    engine_cv_.wait_for(lock, std::chrono::milliseconds(20), [this] {''',
'''  std::optional<MetricsTime> quiet_since;
  bool maintained = false;
  while (is_running_()) {
    process_pending_commands_();
    reap_cancelled_();
    if (execute_one_batch_()) {
      quiet_since = MetricsClock::now();
      maintained = false;
      continue; // more work may be ready
    }
    std::unique_lock<std::mutex> lock(control_mutex_);
    if (!maintained && idle_maintenance_.callback && quiet_since &&
        MetricsClock::now() - *quiet_since >= idle_maintenance_.delay &&
        lifecycle_.load(std::memory_order_relaxed) == Lifecycle::Running &&
        inbox_.empty() && !scheduler_->has_work() &&
        std::none_of(sessions_.begin(), sessions_.end(), [](const auto& entry) {
          return entry.second.active_generation.has_value();
        })) {
      lock.unlock();
#if ET_HAS_EXCEPTIONS
      try {
#endif
        maintained = idle_maintenance_.callback();
#if ET_HAS_EXCEPTIONS
      } catch (...) {
        // Optional maintenance does not revise completed or queued requests.
        maintained = true;
      }
#endif
      lock.lock();
    }
    engine_cv_.wait_for(lock, std::chrono::milliseconds(20), [this] {''')
    elif rel.endswith('/serving_runtime.h'):
        text=replace(text,'#include <cstddef>','#include <chrono>\n#include <cstddef>')
        text=replace(text,
'''  batching::PrefixCache::RetentionPolicy prefix_cache_retention = {};
};''',
'''  batching::PrefixCache::RetentionPolicy prefix_cache_retention = {};
  // Optional engine-thread maintenance after the runtime is fully quiescent.
  // No inference runs concurrently. Must not reenter this runtime or runner.
  std::chrono::milliseconds idle_maintenance_delay{100};
  std::function<void()> idle_maintenance = {};
};''')
    elif rel.endswith('/serving_runtime.cpp'):
        text=replace(text,
'''      runner_ =
          std::make_unique<batching::Runner>(executor, std::move(scheduler));''',
'''      batching::IdleMaintenanceConfig idle;
      idle.delay = config_.idle_maintenance_delay;
      if (config_.idle_maintenance) {
        idle.callback = [this] {
          {
            std::lock_guard<std::mutex> lock(mutex_);
            if (lifecycle_ != Lifecycle::Running || outstanding_ ||
                control_active_.load(std::memory_order_acquire) ||
                !inbox_.empty() || !requests_.empty() || retiring_ ||
                !dispatch_queue_.empty() || !finalizers_.empty() ||
                pending_checkpoint_ || capture_lane_busy_.load()) {
              return false;
            }
          }
          config_.idle_maintenance();
          return true;
        };
      }
      runner_ = std::make_unique<batching::Runner>(
          executor, std::move(scheduler), std::move(idle));''')
        text=replace(text,'  std::size_t outstanding_ = 0;',
'''  struct ControlActivity {
    std::atomic<bool>& active;
    ~ControlActivity() {
      active.store(false, std::memory_order_release);
    }
  };
  std::atomic<bool> control_active_{false};
  std::size_t outstanding_ = 0;''')
        text=replace(text,
'''    for (;;) {
      Command command;
      std::size_t command_index = 0;''',
'''    for (;;) {
      ControlActivity activity{control_active_};
      Command command;
      std::size_t command_index = 0;''')
        text=replace(text,
'''        } else {
          break;
        }
      }
      if (checkpoint) {''',
'''        } else {
          break;
        }
        control_active_.store(true, std::memory_order_release);
      }
      if (checkpoint) {''')
    else:
        raise ValueError(rel)
    return text

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    records={};patch=[]
    for rel in ['extension/llm/batching/runner.h','extension/llm/batching/runner.cpp',
                'extension/llm/serving/serving_runtime.h','extension/llm/serving/serving_runtime.cpp']:
        before=(args.sdk/rel).read_text();after=transform(rel,before)
        target=args.output/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(after)
        records[rel]={'original_SHA256':hashlib.sha256(before.encode()).hexdigest(),'candidate_SHA256':hashlib.sha256(after.encode()).hexdigest()}
        patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    (args.output/'server-idle.patch').write_text(''.join(patch))
    (args.output/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
