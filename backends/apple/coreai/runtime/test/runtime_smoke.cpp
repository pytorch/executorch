/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/apple/coreai/runtime/coreai_cache.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdio>

namespace {
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using RootClear = Error (*)(const char*);
using FileClear = Error (*)(const char*, const char*);
using LoaderClear = Error (*)(DataLoader&, const char*);

// Keep every overload in the link check without touching cache storage.
RootClear volatile clear_root = &executorch::backends::coreai::clear_cache;
FileClear volatile clear_file =
    &executorch::backends::coreai::clear_cache_for_pte;
LoaderClear volatile clear_loader =
    &executorch::backends::coreai::clear_cache_for_pte;
} // namespace

int main() {
  executorch::runtime::runtime_init();
  if (clear_root == nullptr || clear_file == nullptr ||
      clear_loader == nullptr) {
    std::fputs("Core AI cache APIs were not linked\n", stderr);
    return 1;
  }
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
