/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <dispatch/dispatch.h>
#include <cstdint>

namespace executorch::backends::coreai::testing {

inline dispatch_time_t test_deadline(uint64_t seconds = 10) {
  return dispatch_time(DISPATCH_TIME_NOW, seconds * NSEC_PER_SEC);
}

} // namespace executorch::backends::coreai::testing
