/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cpuinfo.h>

namespace executorch::extension::cpuinfo {

/**
 * Returns the number of performance cores, or 0 if it is unknown, for example
 * on x86 or when every core is an efficiency core. On Linux other than
 * Android, the count is capped at get_num_allowed_cpus(), without checking
 * which cores the affinity mask allows.
 */
uint32_t get_num_performant_cores();

/**
 * Returns the number of CPUs in the calling thread's affinity mask on Linux,
 * or 0 if it is not available. taskset and cgroup cpusets shrink the mask,
 * CPU quotas do not. On Android the mask changes as an app moves between the
 * foreground and the background, so it is a poor size for a long-lived pool.
 */
uint32_t get_num_allowed_cpus();

} // namespace executorch::extension::cpuinfo

namespace torch::executorch::cpuinfo { // DEPRECATED
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces. Note that threadpool incorrectly used
// the namespace `torch::executorch` instead of `torch::executor`.
using ::executorch::extension::cpuinfo::get_num_performant_cores; // DEPRECATED
} // namespace torch::executorch::cpuinfo
