/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
// @nolint PATTERNLINT Ok to use stdlib for this optional library
#include <functional>

namespace executorch {
namespace extension {

/**
 * A helper to run function in parallel.
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
