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
 * Returns the number of performance cores, or 0 when they cannot be told
 * apart, as on x86. On Linux other than Android, the count is limited to
 * get_num_allowed_cpus().
 */
uint32_t get_num_performant_cores();

/**
 * Returns the number of CPUs in the calling thread's affinity mask on Linux
 * other than Android, or 0 elsewhere. taskset and cgroup cpusets shrink the
 * mask, CPU quotas do not. Android is left out because it moves apps between
 * cpusets as they go to the background and back, and a thread pool sized in
 * the background would stay small. Other Linux systems rarely change the
 * cpuset of a running process.
 */
uint32_t get_num_allowed_cpus();

} // namespace executorch::extension::cpuinfo

namespace torch::executorch::cpuinfo { // DEPRECATED
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces. Note that threadpool incorrectly used
// the namespace `torch::executorch` instead of `torch::executor`.
using ::executorch::extension::cpuinfo::get_num_performant_cores; // DEPRECATED
} // namespace torch::executorch::cpuinfo
