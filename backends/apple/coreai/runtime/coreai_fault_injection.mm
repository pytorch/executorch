/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_fault_injection.h"

namespace executorch::backends::coreai {

#if defined(COREAI_ASSETS_TESTING) && COREAI_ASSETS_TESTING
namespace testing {
thread_local FaultCallback fault_callback = nullptr;
}

int fault_point(FaultPoint point) {
  return testing::fault_callback == nullptr ? 0
                                            : testing::fault_callback(point);
}
#endif

} // namespace executorch::backends::coreai
