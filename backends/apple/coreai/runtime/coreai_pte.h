/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/result.h>

#include <utility>
#include <vector>

namespace executorch::backends::coreai {

struct CoreAIPteData {
  // Borrowed inline data and segment descriptors must die before the program.
  runtime::FreeableBuffer program_storage;
  std::vector<runtime::FreeableBuffer> processed;

  explicit CoreAIPteData(runtime::FreeableBuffer&& storage)
      : program_storage(std::move(storage)) {}
  CoreAIPteData(CoreAIPteData&&) = default;
  CoreAIPteData(const CoreAIPteData&) = delete;
  CoreAIPteData& operator=(const CoreAIPteData&) = delete;
  CoreAIPteData& operator=(CoreAIPteData&&) = delete;
};

// The loader and its data must remain stable until the returned owner is
// released.
runtime::Result<CoreAIPteData> inspect_coreai_pte(runtime::DataLoader& loader);

} // namespace executorch::backends::coreai
