/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/op_sdpa_row_exact.h>

namespace torch {
namespace executor {
namespace native {

namespace {
thread_local bool sdpa_row_exact = false;
} // namespace

SdpaRowExactScope::SdpaRowExactScope(bool enabled) : previous_(sdpa_row_exact) {
  sdpa_row_exact = enabled;
}

SdpaRowExactScope::~SdpaRowExactScope() {
  sdpa_row_exact = previous_;
}

bool sdpa_row_exact_enabled() {
  return sdpa_row_exact;
}

} // namespace native
} // namespace executor
} // namespace torch
