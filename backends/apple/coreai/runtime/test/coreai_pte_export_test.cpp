/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "coreai_pte.h"

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdio>
#include <cstring>

namespace {
using executorch::extension::FileDataLoader;
using executorch::runtime::DataLoader;

bool matches_export(const char* pte_path, const char* manifest_path) {
  auto loader = FileDataLoader::from(pte_path);
  auto manifest_loader = FileDataLoader::from(manifest_path);
  if (!loader.ok() || !manifest_loader.ok()) {
    return false;
  }
  auto result = executorch::backends::coreai::inspect_coreai_pte(loader.get());
  if (!result.ok() || result->processed.size() != 1) {
    return false;
  }
  auto size = manifest_loader->size();
  if (!size.ok() || size.get() == 0) {
    return false;
  }
  auto expected = manifest_loader->load(
      0,
      size.get(),
      DataLoader::SegmentInfo(DataLoader::SegmentInfo::Type::Program));
  return expected.ok() && result->processed[0].size() == size.get() &&
      std::memcmp(result->processed[0].data(), expected->data(), size.get()) ==
      0;
}
} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  if (argc != 3 || !matches_export(argv[1], argv[2])) {
    std::fputs(
        "PTE inspection did not reproduce the exported Core AI manifest\n",
        stderr);
    return 1;
  }
  std::puts("PTE inspection matches the exported Core AI manifest");
  return 0;
}
