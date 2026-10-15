// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>

namespace ptn {

// Validate method structure and its data-backed bindings against package bytes.
// Throws std::runtime_error for structural defects or mismatched constants.
void validate_method_constants(const Method& method, const Package& package);

// Require every program-wide state FQN to have identical storage and tensor
// metadata across methods. Mutation use may vary.
void validate_program_state(const Program& program);

} // namespace ptn
