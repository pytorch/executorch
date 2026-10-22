/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ${generated_comment}
// Exposing an API for registering generated kernels at once.
#pragma once

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/profiler.h>

#if defined(_WIN32)
#if defined(EXECUTORCH_MANUAL_REGISTRATION_BUILD_SHARED)
#define EXECUTORCH_MANUAL_REGISTRATION_API __declspec(dllexport)
#elif defined(EXECUTORCH_MANUAL_REGISTRATION_USE_SHARED)
#define EXECUTORCH_MANUAL_REGISTRATION_API __declspec(dllimport)
#else
#define EXECUTORCH_MANUAL_REGISTRATION_API
#endif
#elif defined(__GNUC__)
#define EXECUTORCH_MANUAL_REGISTRATION_API __attribute__((visibility("default")))
#else
#define EXECUTORCH_MANUAL_REGISTRATION_API
#endif

namespace torch {
namespace executor {

/// Register the generated kernels in this library.
EXECUTORCH_MANUAL_REGISTRATION_API Error ${manual_registration_function_name}();

} // namespace executor
} // namespace torch

#undef EXECUTORCH_MANUAL_REGISTRATION_API
