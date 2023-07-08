// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * @file
 *
 * Common includes used by all kernel implementations.
 */

#pragma once

// This list should be very conservative since most kernel .cpp files will
// include these and depend on their transitive deps. Only add a header if 99%
// of kernels would have included it anyway.
#include <executorch/core/kernel_types/kernel_types.h> // IWYU pragma: export
#include <executorch/core/kernel_types/util/ScalarTypeUtil.h> // IWYU pragma: export
#include <executorch/core/kernel_types/util/tensor_util.h> // IWYU pragma: export
#include <executorch/runtime/kernel/kernel_runtime_context.h> // IWYU pragma: export
