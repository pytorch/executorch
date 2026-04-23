/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * CpuOps.cpp - CPU op implementations for the portable backend
 *
 * Uses generic dispatch_kernel() to call ExecuTorch portable kernels.
 * Adding a new op is a single line: CPU_DISPATCH_OP(op_name, "kernel_name");
 */

#include <executorch/backends/portable/runtime_v2/cpu/CpuOpRegistry.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace portable {

// Global registry instance
OperatorRegistry<CpuGraph>& cpu_op_registry() {
  static OperatorRegistry<CpuGraph> registry;
  return registry;
}

namespace {

/// Generic kernel dispatch - passes args + dummy return slot to kernel
void dispatch_kernel(
    CpuGraph& graph,
    const std::vector<ValueRef>& args,
    const char* kernel_name) {
  
  auto& ctx = graph.context();
  
  auto kernel = torch::executor::getOpsFn(
      kernel_name, runtime::ArrayRef<runtime::TensorMeta>());
  
  if (!kernel) {
    ET_LOG(Error, "CPU: kernel %s not found", kernel_name);
    ctx.fail(runtime::Error::NotSupported);
    return;
  }

  // Build stack from args + dummy return slot (codegen wrapper expects it)
  runtime::EValue dummy;
  std::vector<runtime::EValue*> stack;
  stack.reserve(args.size() + 1);
  
  for (size_t i = 0; i < args.size(); i++) {
    stack.push_back(graph.value_ptr(args[i]));
  }
  stack.push_back(&dummy);
  
  kernel(ctx, runtime::Span<runtime::EValue*>(stack.data(), stack.size()));
}

/// Dispatch for aten::copy_ (in-place buffer writeback). The IR emits 4
/// args [self, src, non_blocking, out_formal] — the in-place semantic is
/// "self gets src's bytes." We dispatch to the 3-arg in-place
/// "aten::copy_" kernel directly, ignoring the formal out arg.
///
/// Short-circuit for self-aliased writebacks: when the AOT memory planner
/// has aliased self and src to the same slot (true for buffer mutations
/// after our `alias_buffer_mutations_post_planning` pass), self.data_ptr
/// == src.data_ptr — the writeback is trivially satisfied by the kernel
/// that already wrote to the shared slot. Skip the (potentially expensive,
/// large for KV-cache) memcpy in that case.
void dispatch_copy_inplace(
    CpuGraph& graph,
    const std::vector<ValueRef>& args) {
  auto& ctx = graph.context();
  if (args.size() < 3) {
    ET_LOG(Error, "CPU: aten::copy_ expects at least 3 args, got %zu", args.size());
    ctx.fail(runtime::Error::InvalidArgument);
    return;
  }
  auto* self_ev = graph.value_ptr(args[0]);
  auto* src_ev  = graph.value_ptr(args[1]);
    if (self_ev->isTensor() && src_ev->isTensor()) {
      if (self_ev->toTensor().const_data_ptr() ==
          src_ev->toTensor().const_data_ptr()) {
        // Aliased: the bytes already match (kernel that produced src wrote
        // through to self's slot). Writeback is a no-op.
        ET_LOG(Debug,
               "CPU: aten::copy_ short-circuit (self == src @ %p) — no-op",
               self_ev->toTensor().const_data_ptr());
        return;
      }
    }
  auto kernel = torch::executor::getOpsFn(
      "aten::copy_", runtime::ArrayRef<runtime::TensorMeta>());
  if (!kernel) {
    ET_LOG(Error, "CPU: kernel aten::copy_ not found");
    ctx.fail(runtime::Error::NotSupported);
    return;
  }
  // 3-arg in-place schema: copy_(self, src, non_blocking) -> self.
  // Stack: [self, src, non_blocking, return_slot].
  runtime::EValue dummy;
  std::vector<runtime::EValue*> stack = {
      self_ev,                        // self
      src_ev,                         // src
      graph.value_ptr(args[2]),       // non_blocking
      &dummy,                         // return slot
  };
  kernel(ctx, runtime::Span<runtime::EValue*>(stack.data(), stack.size()));
}

}  // namespace

//===----------------------------------------------------------------------===//
// Op Registration - maps op names to kernel names
//===----------------------------------------------------------------------===//

#define CPU_DISPATCH_OP(op_name, kernel_name) \
  CPU_REGISTER_OP(op_name, [](CpuGraph& graph, const std::vector<ValueRef>& args) { \
    dispatch_kernel(graph, args, kernel_name); \
  })

REGISTER_CPU_OPERATORS {
  CPU_DISPATCH_OP(aten::add, "aten::add.out");
  CPU_DISPATCH_OP(aten::sub, "aten::sub.out");
  CPU_DISPATCH_OP(aten::mul, "aten::mul.out");
  CPU_DISPATCH_OP(aten::div, "aten::div.out");
  CPU_DISPATCH_OP(aten::permute_copy, "aten::permute_copy.out");
  CPU_DISPATCH_OP(aten::mm, "aten::mm.out");
  CPU_DISPATCH_OP(aten::clone, "aten::clone.out");
  // copy_ uses a custom dispatcher (drops the formal out arg, calls the
  // 3-arg in-place kernel). The IR emits 4 args but the in-place kernel
  // only takes 3; the formal out is just a memory-plan placeholder.
  CPU_REGISTER_OP(aten::copy_, dispatch_copy_inplace);
}

#undef CPU_DISPATCH_OP

}  // namespace portable
}  // namespace backends
}  // namespace executorch
