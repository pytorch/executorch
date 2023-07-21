// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
// clang-format off
// Generated code for registering custom operators into the dispatcher.

#include <torch/library.h>
#include <ATen/Tensor.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
$ops_headers

namespace torch {
namespace executor {
namespace function {


${dispatch_anonymous_definitions}

// All out variants ops
${static_init_dispatch_registrations}

namespace ${dispatch_namespace}
{
  ${dispatch_namespaced_definitions}

} // namespace ${dispatch_namespace}

} // namespace function
} // namespace executor
} // namespace torch
