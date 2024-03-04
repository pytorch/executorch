/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <tuple>

#include <executorch/backends/xnnpack/threadpool/threadpool.h>
#include <executorch/extension/parallel/thread_parallel.h>
#include <executorch/runtime/platform/assert.h>

namespace torch::executor {

using namespace torch::executorch::threadpool;

inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline std::tuple<int64_t, int64_t>
calc_num_tasks_and_chunk_size(int64_t begin, int64_t end, int64_t grain_size) {
  if ((end - begin) < grain_size) {
    return std::make_tuple(1, std::max((int64_t)0, end - begin));
  }
  // Choose number of tasks based on grain size and number of threads.
  int64_t chunk_size =
      divup((end - begin), get_threadpool()->get_thread_count());
  // Make sure each task is at least grain_size size.
  chunk_size = std::max(grain_size, chunk_size);
  int64_t num_tasks = divup((end - begin), chunk_size);
  return std::make_tuple(num_tasks, chunk_size);
}

void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const std::function<void(int64_t, int64_t)>& f) {
  ET_CHECK_MSG(begin >= 0 && end >= 0, "Begin and end should be non-negative");
  ET_CHECK_MSG(end >= begin, "end should be greater than or equal to begin");
  ET_CHECK_MSG(grain_size > 0, "grain_size should be positive");
  int64_t num_tasks = 0, chunk_size = 0;
  std::tie(num_tasks, chunk_size) =
      calc_num_tasks_and_chunk_size(begin, end, grain_size);

  auto task = [f, begin, end, chunk_size](size_t task_id) {
    int64_t local_start = begin + static_cast<int64_t>(task_id) * chunk_size;
    if (local_start < end) {
      int64_t local_end = std::min(end, (int64_t)(chunk_size + local_start));
      f(local_start, local_end);
    }
  };

  // Per protocol from threadpool (pthreadpool), when this returns, all tasks
  // are executed, so this is synchronous.
  get_threadpool()->run(task, num_tasks);
}

} // namespace torch::executor
