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

#include <pthreadpool.h>

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

  size_t get_thread_count() const;

  /**
   * INTERNAL: Resets the threadpool by creating a new threadpool with requested
   * # of threads. This is not a thread safe call. When calling this method,
   * threads of the threadpool might be doing some work. Some other code may
   * also be holding on to the threadpool pointer, that is no longer valid. This
   * is a private API, which will later be replaced by something that allows
   * creating of threadpool with requested size and use such a threadpool with
   * backend delegates, custom ops or optimized lib.
   */
  [[deprecated("This API is experimental and may change without notice.")]]
  bool _unsafe_reset_threadpool(uint32_t num_threads);

  /**
   * Run, in parallel, function fn(task_id) over task_id in range [0, range).
   * This function is blocking.  All input is processed by the time it returns.
   * NoThreadPoolGuard (see threadpool_guard.h) can used to disable use of
   * multiple threads with the scope of the guard When NoThreadPoolGuard is not
   * used all calls to run method are serialized.
   */
  void run(const std::function<void(size_t)>& fn, size_t range);

 private:
  friend pthreadpool_t get_pthreadpool();

 private:
  // This mutex is used inside get_thread_count API but it is not really needed
  // since data members of ThreadPool objects are not really mutable.
  // TODO(kimishpatel): Figure out if we will allow set_num_threads API, in
  // which case this mutex will be useful. Otherwise remove it.
  mutable std::mutex mutex_;
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_;
};

/**
 * Returns the singleton instance of ThreadPool for ATen/TH multithreading.
 */
ThreadPool* get_threadpool();

/**
 * Returns the underlying pthreadpool instance used by the implementation of
 * ThreadPool returned by `get_threadpool()`. Only for use in external libraries
 * so as to unify threading across internal (i.e. ATen, etc.) and external (e.g.
 * NNPACK, QNNPACK, XNNPACK) use cases.
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
