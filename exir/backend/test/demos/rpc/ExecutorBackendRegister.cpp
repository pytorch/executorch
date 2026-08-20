/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/exir/backend/test/demos/rpc/ExecutorBackend.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>

namespace example {
namespace {
static ::executorch::runtime::Error register_success =
    register_executor_backend();
} // namespace
} // namespace example
