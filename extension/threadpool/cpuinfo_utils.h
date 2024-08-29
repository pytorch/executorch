/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cpuinfo.h>

namespace torch {
namespace executorch {
namespace cpuinfo {

uint32_t get_num_performant_cores();

} // namespace cpuinfo
} // namespace executorch
} // namespace torch
