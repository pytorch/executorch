/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <pthreadpool.h>

#include <executorch/runtime/core/function_ref.h>

/*
 * Threadpool Options:
 *
 * Threadpool size has a sizble affect on performance. By default, the
 * threadpool will be sized according to the number of performance cores. This
 * behavior can be overriden with the following build-time options. Note that
 * these options are mutually exclusive.
 *
 * - EXECUTORCH_THREADPOOL_USE_PERFORMANCE_CORES (flag) - Sizes the threadpool
 * equal to the number of performance cores on the system. This is the default
 * behavior.
 * - EXECUTORCH_THREADPOOL_USE_ALL_LOGICAL_CORES (flag) - Sizes the threadpool
 * equal to the number of logical cores on system. This is the historical
 * behavior.
 *
 * On Linux other than Android, the threadpool starts with no more threads than
 * the CPUs in the affinity mask of the thread that first calls
 * get_threadpool(). taskset and cgroup cpusets shrink that mask, CPU quotas do
 * not. The mask also sizes the threadpool when the performance core count is
 * unknown, as on x86. It is read once: the threadpool does not follow later
 * changes to the mask, and a child created by fork() keeps its parent's size.
 * If the mask can grow later, set the thread count you need with
 * _unsafe_reset_threadpool() before loading any model.
 */

namespace executorch::extension::threadpool {

class ThreadPool final {
 public:
  explicit ThreadPool(size_t thread_count = 0);
  ~ThreadPool() = default;

  // Make threadpool non copyable
  // Non-copyable: threadpool cannot be copied because it will
  // effectively require cloning of threadpool.
  // Cloning can be done by just calling create_thread_pool.
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // Make threadpool non-movable.
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  /**
   * Returns the number of threads in the threadpool. This number does not
   * reflect temporary lowering via UseNThreadsThreadPoolGuard.
   */
  size_t get_thread_count() const;

  /**
   * INTERNAL: Resets the threadpool to the requested # of threads. This is not
   * a thread safe call. A threadpool that get_pthreadpool() returned is not
   * destroyed but kept with its size, because delegates such as XNNPACK keep
   * using it for the methods they loaded. So the new size applies to such
   * methods only when they are loaded after this call. A later reset to a kept
   * size reuses that threadpool. Kept threadpools live until
   * _unsafe_destroy_threadpool() or the end of the process. This
   * is a private API, which will later be replaced by something that allows
   * creating of threadpool with requested size and use such a threadpool with
   * backend delegates, custom ops or optimized lib.
   * For Meta internal use, there is
   * executorch::extension::threadpool::UseNThreadsThreadPoolGuard API that
   * provides a safer way to select a subset of threads, from threadpool, to run
   * the model on.
   */
  [[deprecated(
      "This API is experimental and may change without notice. Consider using UseNThreadsThreadPoolGuard")]]
  bool _unsafe_reset_threadpool(uint32_t num_threads);

  /**
   * INTERNAL: Destroys the threadpool. This is not a thread safe call. When
   * calling this method, threads of the threadpool might be doing some work.
   * Some other code may also be holding on to the threadpool pointer, that is
   * no longer valid.
   */
  [[deprecated("This API is experimental and may change without notice.")]]
  void _unsafe_destroy_threadpool();

  /**
   * Run, in parallel, function fn(task_id) over task_id in range [0, range).
   * This function is blocking.  All input is processed by the time it returns.
   * NoThreadPoolGuard (see threadpool_guard.h) can used to disable use of
   * multiple threads with the scope of the guard When NoThreadPoolGuard is not
   * used all calls to run method are serialized.
   */
  void run(runtime::FunctionRef<void(size_t)> fn, size_t range);

 private:
  friend pthreadpool_t get_pthreadpool();

 private:
  using PthreadpoolPtr =
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>;

  // Guards the data members below.
  mutable std::mutex mutex_;
  PthreadpoolPtr threadpool_;
  // The number of threads in the threadpool.
  size_t thread_count_;
  // Whether get_pthreadpool() has returned threadpool_.
  bool threadpool_shared_ = false;
  // Shared threadpools that a reset replaced, with their thread counts.
  std::vector<std::pair<size_t, PthreadpoolPtr>> retired_threadpools_;
};

/**
 * Returns the singleton instance of ThreadPool for ATen/TH multithreading.
 */
ThreadPool* get_threadpool();

/**
 * Returns the underlying pthreadpool instance used by the implementation of
 * ThreadPool returned by `get_threadpool()`. Only for use in external libraries
 * so as to unify threading across internal (i.e. ATen, etc.) and external (e.g.
 * NNPACK, QNNPACK, XNNPACK) use cases. The returned instance stays valid after
 * ThreadPool::_unsafe_reset_threadpool().
 */
pthreadpool_t get_pthreadpool();

} // namespace executorch::extension::threadpool

namespace torch::executorch::threadpool { // DEPRECATED
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces. Note that threadpool incorrectly used
// the namespace `torch::executorch` instead of `torch::executor`.
using ::executorch::extension::threadpool::get_pthreadpool; // DEPRECATED
using ::executorch::extension::threadpool::get_threadpool; // DEPRECATED
using ::executorch::extension::threadpool::ThreadPool; // DEPRECATED
} // namespace torch::executorch::threadpool
