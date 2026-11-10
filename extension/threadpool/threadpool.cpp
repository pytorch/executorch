/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>

#include <algorithm>
#include <memory>

#include <executorch/extension/threadpool/threadpool_guard.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/runtime.h>

#include <cpuinfo.h>

// At most one mode should be set.
#if (                                                       \
    defined(EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES) && \
    defined(EXECUTORCH_THREADPOOL_USE_PERFORMANCE_CORES))
#error Multiple \
            threadpool size specifiers are set.At most one of                \
    EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES,                             \
    and EXECUTORCH_THREADPOOL_USE_PERFORMANCE_CORES may be defined.
#endif

// Default to EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES if no mode is set.
#if !defined(EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES) && \
    !defined(EXECUTORCH_THREADPOOL_USE_PERFORMANCE_CORES)
#define EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES 1
#endif

namespace executorch::extension::threadpool {

namespace {
#if !defined(WIN32) && !defined(__EMSCRIPTEN__)
// After fork, the child process inherits the data-structures of the parent
// process' thread-pool, but since those threads don't exist, the thread-pool
// is corrupt. It's leaked in order to prevent segfaults.
// Ref: https://github.com/pytorch/pytorch/issues/54752#issuecomment-810315302
bool leak_corrupted_threadpool = false;
// Incremented in every child that fork() creates.
size_t fork_generation = 0;

void child_atfork() {
  leak_corrupted_threadpool = true;
  ++fork_generation;
}
#endif

std::shared_ptr<pthreadpool> create_pthreadpool(size_t thread_count) {
#if !defined(WIN32) && !defined(__EMSCRIPTEN__)
  // A holder of get_shared_pthreadpool() can release the last reference in a
  // forked child, where the threadpool is corrupt, so it is leaked there.
  return std::shared_ptr<pthreadpool>(
      pthreadpool_create(thread_count),
      [generation = fork_generation](pthreadpool_t threadpool) {
        if (generation == fork_generation) {
          pthreadpool_destroy(threadpool);
        }
      });
#else
  return std::shared_ptr<pthreadpool>(
      pthreadpool_create(thread_count), pthreadpool_destroy);
#endif
}
} // namespace

ThreadPool::ThreadPool(size_t thread_count)
    : threadpool_(create_pthreadpool(thread_count)),
      thread_count_(
          threadpool_ ? pthreadpool_get_threads_count(threadpool_.get()) : 0) {}

size_t ThreadPool::get_thread_count() const {
  std::lock_guard<std::mutex> lock{mutex_};

  return thread_count_;
}

bool ThreadPool::_unsafe_reset_threadpool(uint32_t new_thread_count) {
  ET_LOG(Info, "Resetting threadpool to %u threads.", new_thread_count);

  // No need to do anything if the count is same or 0
  if (new_thread_count == get_thread_count() || new_thread_count == 0) {
    return true;
  }

  std::lock_guard<std::mutex> lock{mutex_};
  std::lock_guard<std::mutex> threadpool_lock{threadpool_mutex_};

  if (threadpool_.use_count() > 1) {
    ET_LOG(
        Info,
        "Methods loaded before this reset keep the old threadpool of %zu threads until they are destroyed.",
        thread_count_);
  }
  threadpool_ = create_pthreadpool(new_thread_count);
  thread_count_ =
      threadpool_ ? pthreadpool_get_threads_count(threadpool_.get()) : 0;
  return true;
}

void ThreadPool::_unsafe_destroy_threadpool() {
  std::lock_guard<std::mutex> lock{mutex_};
  std::lock_guard<std::mutex> threadpool_lock{threadpool_mutex_};
  ET_LOG(Info, "Destroying threadpool.");
  threadpool_.reset();
  thread_count_ = 0;
}

void ThreadPool::run(
    runtime::FunctionRef<void(size_t)> fn,
    const size_t range) {
  // Run on same thread if NoThreadPoolGuard guard is enabled
  if (NoThreadPoolGuard::is_enabled()) {
    for (size_t i = 0; i < range; ++i) {
      fn(i);
    }
    return;
  }

  std::lock_guard<std::mutex> lock{mutex_};

  ET_CHECK_MSG(!NoThreadPoolGuard::is_enabled(), "Inside a threadpool guard!");
  ET_CHECK_MSG(threadpool_.get(), "Invalid threadpool!");

  struct Context final {
    const std::function<void(size_t)>& fn;
  } context{
      fn,
  };

  pthreadpool_parallelize_1d(
      threadpool_.get(),
      // Note: pthreadpool_parallelize_1d() is a blocking function.  The
      // function pointer to this lambda passed on to
      // pthreadpool_parallelize_1d() cannot go out of scope until
      // pthreadpool_parallelize_1d() returns.
      [](void* const context, const size_t item) {
        NoThreadPoolGuard guard;
        reinterpret_cast<Context*>(context)->fn(item);
      },
      &context,
      range,
      0u);
}

// get_threadpool is not thread safe due to leak_corrupted_threadpool
// Make this part threadsafe: TODO(kimishpatel)
ThreadPool* get_threadpool() {
  executorch::runtime::runtime_init();

  if (!cpuinfo_initialize()) {
    ET_LOG(Error, "cpuinfo initialization failed");
    return nullptr; // NOLINT(facebook-hte-NullableReturn)
  }

  static const int num_threads = ([]() {
#if defined(EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES)
    // Use threads=cores.
    auto result = cpuinfo_get_processors_count();
#else
    // Set threads equal to the number of performance cores.
    auto result = ::executorch::extension::cpuinfo::get_num_performant_cores();
#endif
#if defined(__linux__) && !defined(__ANDROID__)
    // Never start more threads than the CPUs in this thread's affinity mask,
    // which the pool's threads inherit, and use that number when the count is
    // unknown (0), which pthreadpool would fill in with every online CPU, past
    // the cap below.
    const auto allowed =
        ::executorch::extension::cpuinfo::get_num_allowed_cpus();
    if (allowed > 0) {
      result = result == 0 ? allowed : std::min(result, allowed);
    }
#endif

    /*
     * For llvm-tsan, holding limit for the number of locks for a single thread
     * is 63 (because of comparison < 64 instead of <=). pthreadpool's worst
     * case is the number of threads in a pool. So we want to limit the
     * threadpool size to 64 when running with tsan. However, sometimes it is
     * tricky to detect if we are running under tsan, for now capping the
     * default threadcount to the tsan limit unconditionally.
     */
    constexpr decltype(result) tsan_thread_limit = 63;
    return std::min(result, tsan_thread_limit);
  })();

#if defined(_WIN32)
  // Never destroyed. At process exit Windows terminates the worker threads
  // before running a DLL's static destructors, so tearing the pool down there
  // waits on a lock a terminated worker may hold and the process never exits.
  // A pool that a reset replaced is destroyed by its last holder instead, so
  // the same can happen if a DLL's static destructor releases it.
  static auto& threadpool = *new std::unique_ptr<ThreadPool>(
      std::make_unique<ThreadPool>(num_threads));
#else
  static auto threadpool = std::make_unique<ThreadPool>(num_threads);
#endif

// Inheriting from old threadpool to get around segfault issue
// commented above at child_atfork
#if !defined(WIN32) && !defined(__EMSCRIPTEN__)
  // @lint-ignore CLANGTIDY facebook-hte-std::once_flag
  static std::once_flag flag;
  // @lint-ignore CLANGTIDY facebook-hte-std::call_once
  std::call_once(
      flag, []() { pthread_atfork(nullptr, nullptr, child_atfork); });
  if ET_UNLIKELY (leak_corrupted_threadpool) {
    leak_corrupted_threadpool = false;
    if (auto leaked = threadpool.release()) {
      auto t = leaked->get_thread_count();
      threadpool = std::make_unique<ThreadPool>(t);
    }
  }
#endif
  return threadpool.get();
}

pthreadpool_t get_pthreadpool() {
  if (NoThreadPoolGuard::is_enabled()) {
    return nullptr;
  }
  ThreadPool* const threadpool = get_threadpool();
  ET_CHECK_MSG(threadpool, "Failed to acquire an instance of ThreadPool!");
  std::lock_guard<std::mutex> lock{threadpool->threadpool_mutex_};
  return threadpool->threadpool_.get();
}

std::shared_ptr<pthreadpool> get_shared_pthreadpool() {
  if (NoThreadPoolGuard::is_enabled()) {
    return nullptr;
  }
  ThreadPool* const threadpool = get_threadpool();
  ET_CHECK_MSG(threadpool, "Failed to acquire an instance of ThreadPool!");
  std::lock_guard<std::mutex> lock{threadpool->threadpool_mutex_};
  return threadpool->threadpool_;
}

} // namespace executorch::extension::threadpool
