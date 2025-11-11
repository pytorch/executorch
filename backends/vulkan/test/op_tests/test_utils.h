/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

/**
 * Convert at::ScalarType to executorch::ScalarType
 */
executorch::aten::ScalarType at_scalartype_to_et_scalartype(
    at::ScalarType dtype);

/**
 * Get the string name of a c10::ScalarType for better error messages
 */
std::string scalar_type_name(c10::ScalarType dtype);

/**
 * Convert c10::ScalarType to vkcompute::vkapi::ScalarType
 */
vkcompute::vkapi::ScalarType from_at_scalartype(c10::ScalarType at_scalartype);
