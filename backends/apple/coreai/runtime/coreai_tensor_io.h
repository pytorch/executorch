/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#import "ETCoreAIBridge.h"

namespace executorch::backends::coreai {

runtime::Error execute_session(
    id<ETCoreAISession> session,
    runtime::Span<runtime::EValue*> args,
    size_t input_count,
    size_t output_count);

} // namespace executorch::backends::coreai
