#pragma once

#include <pthreadpool.h>

// @nolint PATTERNLINT Ok to use stdlib for this optional library
#include <functional>
// @nolint PATTERNLINT Ok to use stdlib for this optional library
#include <memory>
// @nolint PATTERNLINT Ok to use stdlib for this optional library
#include <mutex>

namespace torch {
namespace executorch {
namespace threadpool {

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
  // For now this is non-movable, but if we want to have clients
  // such as say torch::executorch::Executor, to be able to own
  // threadpool, then we will have to make this movable.
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  size_t get_thread_count() const;

  /*
   * Resets the threadpool by creating a new threadpool with requested # of
   * threads. This is not a thread safe call. When calling this method, threads
   * of the threadpool might be doing some work. Some other code may also be
   * holding on to the threadpool pointer, that is no longer valid. This is a
   * private API, which will later be replaced by something that allows creating
   * of threadpool with requested size and use such a threadpool with backend
   * delegates, custom ops or optimized lib.
   */
  bool _unsafe_reset_threadpool(uint32_t num_threads);

  // Run, in parallel, function fn(task_id) over task_id in range [0, range).
  // This function is blocking.  All input is processed by the time it returns.
  // NoThreadPoolGuard (see threadpool_guard.h) can used to disable
  // use of multiple threads with the scope of the guard
  // When NoThreadPoolGuard is not used all calls to run method are serialized.
  void run(const std::function<void(size_t)>& fn, size_t range);

 private:
  friend pthreadpool_t get_pthreadpool();

 private:
  // This mutex is used inside get_thread_count API but it is not
  // really needed. Since data members of ThreadPool objects are not
  // really mutable.
  // Figure out if we will allow set_num_threads API, in which mutex
  // will be useful. Otherwise remove it.
  // TODO(kimishpatel)
  mutable std::mutex mutex_;
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_;
};

// Return a singleton instance of ThreadPool for ATen/TH multithreading.
ThreadPool* get_threadpool();

// Exposes the underlying implementation of ThreadPool.
// Only for use in external libraries so as to unify threading across
// internal (i.e. ATen, etc.) and external (e.g. NNPACK, QNNPACK, XNNPACK)
// use cases.
pthreadpool_t get_pthreadpool();

} // namespace threadpool
} // namespace executorch
} // namespace torch
