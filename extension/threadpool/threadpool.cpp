/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/threadpool/threadpool.h>

#include <algorithm>
#include <atomic>
#include <memory>

#include <executorch/extension/threadpool/threadpool_guard.h>
#include <executorch/runtime/platform/assert.h>

#include <cpuinfo.h>

namespace executorch::extension::threadpool {

#if !(defined(WIN32))
namespace {
// After fork, the child process inherits the data-structures of the parent
// process' thread-pool, but since those threads don't exist, the thread-pool
// is corrupt. It's leaked in order to prevent segfaults.
// Ref: https://github.com/pytorch/pytorch/issues/54752#issuecomment-810315302
bool leak_corrupted_threadpool = false;

void child_atfork() {
  leak_corrupted_threadpool = true;
}

} // namespace
#endif

ThreadPool::ThreadPool(size_t thread_count)
    : threadpool_(pthreadpool_create(thread_count), pthreadpool_destroy) {}

size_t ThreadPool::get_thread_count() const {
  std::lock_guard<std::mutex> lock{mutex_};

  ET_CHECK_MSG(threadpool_.get(), "Invalid threadpool!");
  return pthreadpool_get_threads_count(threadpool_.get());
}

bool ThreadPool::_unsafe_reset_threadpool(uint32_t new_thread_count) {
  // No need to do anything if the count is same or 0
  if (new_thread_count == get_thread_count() || new_thread_count == 0) {
    return true;
  }

  std::lock_guard<std::mutex> lock{mutex_};

  threadpool_.reset(pthreadpool_create(new_thread_count));
  return true;
}

void ThreadPool::run(
    const std::function<void(size_t)>& fn,
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
        reinterpret_cast<Context*>(context)->fn(item);
      },
      &context,
      range,
      0u);
}

// get_threadpool is not thread safe due to leak_corrupted_threadpool
// Make this part threadsafe: TODO(kimishpatel)
ThreadPool* get_threadpool() {
  ET_CHECK_MSG(cpuinfo_initialize(), "cpuinfo initialization failed");
  int num_threads = cpuinfo_get_processors_count();
  /*
   * For llvm-tsan, holding limit for the number of locks for a single thread
   * is 63 (because of comparison < 64 instead of <=). pthreadpool's worst
   * case is the number of threads in a pool. So we want to limit the threadpool
   * size to 64 when running with tsan. However, sometimes it is tricky to
   * detect if we are running under tsan, for now capping the default
   * threadcount to the tsan limit unconditionally.
   */
  constexpr int tsan_thread_limit = 63;
  num_threads = std::min(num_threads, tsan_thread_limit);
  static auto threadpool = std::make_unique<ThreadPool>(num_threads);

// Inheriting from old threadpool to get around segfault issue
// commented above at child_atfork
#if !(defined(WIN32))
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
  return threadpool->threadpool_.get();
}

} // namespace executorch::extension::threadpool
