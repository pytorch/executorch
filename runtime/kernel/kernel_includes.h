/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * Common includes used by all kernel implementations.
 */

#pragma once

// This list should be very conservative since most kernel .cpp files will
// include these and depend on their transitive deps. Only add a header if 99%
// of kernels would have included it anyway.
#include <executorch/runtime/core/exec_aten/exec_aten.h> // IWYU pragma: export
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h> // IWYU pragma: export
#include <executorch/runtime/core/exec_aten/util/tensor_util.h> // IWYU pragma: export
#include <executorch/runtime/kernel/kernel_runtime_context.h> // IWYU pragma: export
