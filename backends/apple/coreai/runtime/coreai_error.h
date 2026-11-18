/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#import <Foundation/Foundation.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::coreai {

// Binding and execution preserve bridge errors that callers can act on.
runtime::Error execution_error(NSError* error);
// Acquisition failures describe SDK/cache internals, not caller arguments.
runtime::Error acquisition_error(NSError* error);

} // namespace executorch::backends::coreai
