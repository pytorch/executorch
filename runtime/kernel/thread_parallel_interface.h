/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <functional>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace extension {
namespace internal {
template <typename Func>
inline bool parallel_for_no_threadpool(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const Func& f) {
  ET_CHECK_OR_RETURN_FALSE(
      begin >= 0 && end >= 0 && end >= begin,
      "begin = %" PRId64 ", end = %" PRId64,
      begin,
      end);
  ET_CHECK_OR_RETURN_FALSE(grain_size > 0, "grain_size = %" PRId64, grain_size);
  f(begin, end);
  return true;
}

// Match GRAIN_SIZE from PyTorch core.
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIterator.h#L78
constexpr int64_t GRAIN_SIZE = 32768;

} // namespace internal

#ifdef ET_USE_THREADPOOL
/**
 * A helper to run a function in parallel.
 *
 * begin, end: describe the extent of the workitems via first and last workitem
 * to be processed
 * grain_size: number of workitems processed by user callback which is
 * described below
 * f: user function applied in parallel to the chunks, signature:
 *   void f(int64_t begin, int64_t end)
 * Returns true if all work items are processed successfully, false otherwise
 *
 * Warning: parallel_for does NOT copy thread local states from the current
 * thread to the worker threads. Users need to protect the access to captured
 * data if they mutate them in f.
 */
bool parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const std::function<void(int64_t, int64_t)>& f);

int64_t get_thread_num();

void set_thread_num(int64_t thread_num);
#else // ET_USE_THREADPOOL
template <typename Func>
bool parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const Func& func) {
  return internal::parallel_for_no_threadpool(begin, end, grain_size, func);
}

inline int64_t get_thread_num() {
  return 0;
}

inline void set_thread_num(int64_t thread_num) {
  ET_DCHECK_MSG(false, "cannot set_thread_num without threading support!");
}
#endif // ET_USE_THREADPOOL

/**
 * Convenience version of parallel_for that sets the grain size to
 * internal::GRAIN_SIZE.
 */
template <typename Func>
bool parallel_for(const int64_t begin, const int64_t end, const Func& func) {
  return parallel_for(begin, end, internal::GRAIN_SIZE, func);
}
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::get_thread_num;
using ::executorch::extension::parallel_for;
using ::executorch::extension::set_thread_num;
} // namespace executor
} // namespace torch
