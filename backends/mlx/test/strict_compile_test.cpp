/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Compile-only test to verify MLX delegate headers are clean under strict
 * warnings (-Wconversion, -Wsign-conversion, -Wshorten-64-to-32, -Werror).
 *
 * This file includes the delegate headers and instantiates key types to ensure
 * template code is also checked. It is never linked or executed — a successful
 * compilation is the test.
 */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/named_data_map.h>
#include <mlx/mlx.h>
#pragma clang diagnostic pop

// These are the headers we want to verify under strict warnings
#include "MLXExecutor.h"
#include "MLXInterpreter.h"
#include "MLXLoader.h"

// Instantiate key types to ensure template code is checked
namespace {
[[maybe_unused]] void force_instantiation() {
  using namespace executorch::backends::mlx;

  // Force safe_mul template instantiation
  (void)safe_mul<size_t>(0, 0, "test");

  // Force check_allocation_bounded instantiation
  ::mlx::core::Shape shape = {1, 2, 3};
  check_allocation_bounded(shape, ::mlx::core::float32, "test");
}
} // namespace
