/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cerrno>
#include <utility>
#include "coreai_file.h"

namespace executorch::backends::coreai {

enum class FaultPoint {
  Write,
  SyncFile,
  SyncDirectory,
  FullSync,
  Rename,
  Remove,
  CreateStagingDirectory,
  OpenStagingDirectory,
  BeforeCreateDirectory,
  BeforeCacheEntryLock,
  BeforeSDK,
  AfterSDK,
  BeforeEvict,
  AfterEvict,
};

#if defined(COREAI_ASSETS_TESTING) && COREAI_ASSETS_TESTING
int fault_point(FaultPoint point);
namespace testing {
using FaultCallback = int (*)(FaultPoint);
extern thread_local FaultCallback fault_callback;
} // namespace testing
#else
constexpr int fault_point(FaultPoint) {
  return 0;
}
#endif

template <typename F>
auto fault_aware_retry(FaultPoint point, F&& call) {
  return retry_eintr([&]() -> decltype(call()) {
    const int fault = fault_point(point);
    if (fault != 0) {
      errno = fault;
      return -1;
    }
    return call();
  });
}

} // namespace executorch::backends::coreai
