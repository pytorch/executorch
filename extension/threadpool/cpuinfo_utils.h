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

uint32_t get_num_performant_cores();

} // namespace executorch::extension::cpuinfo

namespace torch::executorch::cpuinfo { // DEPRECATED
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces. Note that threadpool incorrectly used
// the namespace `torch::executorch` instead of `torch::executor`.
using ::executorch::extension::cpuinfo::get_num_performant_cores; // DEPRECATED
} // namespace torch::executorch::cpuinfo
