/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace torch {
namespace executor {

/**
 * BackendExecutionContext will be used to inject run time context.
 * The current plan is to add temp allocator and event tracer (for profiling) as
 * part of the runtime context.
 */
class BackendExecutionContext final {};

} // namespace executor
} // namespace torch
