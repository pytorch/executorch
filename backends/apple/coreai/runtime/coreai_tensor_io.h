/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <functional>
#import "ETCoreAIBridge.h"

namespace executorch::backends::coreai {

// Validates args, then runs the session select_session picks for them; the
// selector may assume every argument is a valid tensor.
runtime::Error execute_session(
    runtime::Span<runtime::EValue*> args,
    size_t input_count,
    size_t output_count,
    const std::function<runtime::Result<id<ETCoreAISession>>()>&
        select_session);

} // namespace executorch::backends::coreai
