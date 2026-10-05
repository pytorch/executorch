/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdio>

int main() {
  executorch::runtime::runtime_init();
  const auto* backend = executorch::runtime::get_backend_class("CoreAIBackend");
  if (backend == nullptr) {
    std::fputs("CoreAIBackend was not retained by the linker\n", stderr);
    return 1;
  }
  if (!backend->is_available()) {
    std::fputs("CoreAIBackend requires OS 27 or newer\n", stderr);
    return 1;
  }
  std::puts("CoreAIBackend is registered and available");
  return 0;
}
