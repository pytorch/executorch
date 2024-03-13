/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#include <torch/library.h>
#include <ATen/Tensor.h>

${ops_headers}

namespace torch {
namespace executor {
namespace function {

${dispatch_anonymous_definitions}

// All out variants ops
${static_init_dispatch_registrations}

namespace ${dispatch_namespace} {

${dispatch_namespaced_definitions}

} // namespace ${dispatch_namespace}

} // namespace function
} // namespace executor
} // namespace torch
