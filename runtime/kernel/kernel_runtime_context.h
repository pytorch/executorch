// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace torch {
namespace executor {

/**
 * Bucket type abstraction that contains many elements of runtime state that
 * a kernel author may want available, but would otherwise be unable to access.
 *
 * Forwarded along to all operators when running in lean mode. NOTE: Will not be
 * forwarded to operators if running in ATen mode as those operators do not
 * expect to receive a KernelRuntimeContext and would not use it.
 *
 * This includes things like setting an error state, a scratch allocator for
 * operators that need more then constant space, and a TensorResizer for dynamic
 * shape tensors allowing programs to be more flexible with Tensor shape.
 *
 * TODO(T147221312): Define this interface
 */
class KernelRuntimeContext {};

} // namespace executor
} // namespace torch

// TODO(T147221312): Remove these aliases once all code uses
// KernelRuntimeContext.
namespace exec_aten {
using RuntimeContext = torch::executor::KernelRuntimeContext;
} // namespace exec_aten
namespace torch::executor {
using RuntimeContext = torch::executor::KernelRuntimeContext;
} // namespace torch::executor
