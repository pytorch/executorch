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

#include <executorch/backends/native/runtimes/cpu/CpuOpRegistry.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace portable {

// Global registry instance — returns the wrapper that adds default-handler
// fallback to the underlying OperatorRegistry<CpuGraph>.
CpuOpDispatcher& cpu_op_registry() {
  static CpuOpDispatcher registry;
  return registry;
}

namespace {

/// Convention for default dispatch: op full_name → ET kernel name.
///
/// Post-lowering, ET's IR carries op names already in `.out` form
/// (either name="aten::X.out" overload="" or name="aten::X" overload="out",
/// both producing full_name "aten::X.out"). The default resolver is the
/// identity — the op name IS the kernel name.
///
/// In-place ops (aten::X_) and special-case ops (aten::copy_, ...) are
/// expected to have explicit registrations and never reach this resolver.
std::string default_kernel_name_for(const std::string& op_name) {
  return op_name;
}

/// Generic kernel dispatch - passes args + dummy return slot to kernel
/// Dispatch a kernel.
///
/// IR args are passed positionally into the kernel's stack. When
/// `pass_return_as_kernel_out` is true, the IR's trailing return slot
/// is treated as the kernel's `out` arg (it's read by the kernel as
/// the destination buffer to write into) and a fresh dummy slot is
/// appended for the kernel's actual return slot. Used by `_` → `.out`
/// remap registrations: the in-place op's IR has one fewer schema
/// arg than the `.out` kernel (no `out`); we bridge by passing the
/// IR's trailing return slot as the kernel's `out`.
///
/// In-place correctness depends on the AOT memory planner having
/// aliased the IR's return slot to the mutated arg via Tensor(a!) —
/// the kernel writes into that storage as if it were a true `out`
/// buffer, which is the in-place semantic.
///
/// When the IR op IS the kernel (default dispatch), pass false (the
/// default): IR args go straight to the kernel's stack with no extra
/// slots — IR's trailing return matches kernel's trailing return by
/// position.
void dispatch_kernel(
    CpuGraph& graph,
    const std::vector<ValueRef>& args,
    const char* kernel_name,
    bool pass_return_as_kernel_out = false) {
  auto& ctx = graph.context();

  auto kernel = torch::executor::getOpsFn(
      kernel_name, runtime::ArrayRef<runtime::TensorMeta>());

  if (!kernel) {
    ET_LOG(Error, "CPU: kernel %s not found", kernel_name);
    ctx.fail(runtime::Error::NotSupported);
    return;
  }

  runtime::EValue dummy;
  std::vector<runtime::EValue*> stack;
  stack.reserve(args.size() + (pass_return_as_kernel_out ? 1 : 0));

  for (size_t i = 0; i < args.size(); i++) {
    stack.push_back(graph.value_ptr(args[i]));
  }
  if (pass_return_as_kernel_out) {
    // IR's trailing slot already filled the kernel's `out` position
    // by positional alignment. Append a fresh slot for the kernel's
    // actual return.
    stack.push_back(&dummy);
  }

  kernel(ctx, runtime::Span<runtime::EValue*>(stack.data(), stack.size()));
}

/// Default dispatcher used by CpuOpDispatcher when no explicit handler
/// is registered. Computes the ET kernel name via
/// `default_kernel_name_for` and delegates to `dispatch_kernel`.
void default_dispatch(
    CpuGraph& graph,
    const std::vector<ValueRef>& args,
    const std::string& op_name) {
  std::string kernel_name = default_kernel_name_for(op_name);
  dispatch_kernel(graph, args, kernel_name.c_str());
}

/// Dispatch for aten::copy_. Routes to ET's 3-arg in-place
/// `aten::copy_` kernel directly. The IR emits 4 args [self, src,
/// non_blocking, formal_out] (the .out-style trailing slot the emit
/// machinery requires); the kernel signature is
///   copy_(self, src, non_blocking) -> self
/// which expects a 4-element stack [args..., return_slot]. We use
/// `formal_out` as the return slot — it's an EValue already allocated
/// by the memory planner specifically for this purpose.
///
/// Distinct from the default `args + dummy` stack shape (which would
/// produce 5 elements and crash the 3-arg kernel), so this needs an
/// explicit handler instead of routing through the default.
void dispatch_copy_inplace(CpuGraph& graph, const std::vector<ValueRef>& args) {
  auto& ctx = graph.context();
  if (args.size() < 4) {
    ET_LOG(
        Error,
        "CPU: aten::copy_ expects 4 args [self, src, non_blocking, out], got %zu",
        args.size());
    ctx.fail(runtime::Error::InvalidArgument);
    return;
  }
  // Short-circuit when self and src share storage. The AOT memory
  // planner aliases buffer-mutation writebacks to their source's slot
  // (Tensor(a!) chain), which makes `aten::copy_(buf, buf)` a no-op
  // semantically but a full-tensor `memcpy(p, p, n)` in libc-land
  // (UB by C++ spec; libc doesn't short-circuit; full memory traffic
  // for nothing). Catch it here.
  auto* self_ev = graph.value_ptr(args[0]);
  auto* src_ev = graph.value_ptr(args[1]);
  if (self_ev && src_ev && self_ev->isTensor() && src_ev->isTensor()) {
    const auto& self_t = self_ev->toTensor();
    const auto& src_t = src_ev->toTensor();
    if (self_t.const_data_ptr() == src_t.const_data_ptr() &&
        self_t.nbytes() == src_t.nbytes()) {
      // Same storage — copy is a no-op. Skip the kernel call entirely.
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
  // 4-element stack: 3 kernel args + return slot (formal_out).
  std::vector<runtime::EValue*> stack = {
      graph.value_ptr(args[0]), // self
      graph.value_ptr(args[1]), // src
      graph.value_ptr(args[2]), // non_blocking
      graph.value_ptr(args[3]), // return slot (formal "out")
  };
  kernel(ctx, runtime::Span<runtime::EValue*>(stack.data(), stack.size()));
}

} // namespace

//===----------------------------------------------------------------------===//
// Op Registration - maps op names to kernel names
//===----------------------------------------------------------------------===//

#define CPU_DISPATCH_OP(op_name, kernel_name)                           \
  CPU_REGISTER_OP(                                                      \
      op_name, [](CpuGraph& graph, const std::vector<ValueRef>& args) { \
        dispatch_kernel(graph, args, kernel_name,                       \
                        /*pass_return_as_kernel_out=*/true);            \
      })

REGISTER_CPU_OPERATORS {
  // Default dispatcher: any op not explicitly registered below routes
  // through here. Convention: aten::X → aten::X.out kernel via ET's
  // portable kernel registry (which selective build trims as needed).
  // Out-variant ops (aten::add, aten::mm, aten::clone, ...) are NOT
  // listed here — they fall through to the default. Adding a new
  // out-variant op requires no code change.
  cpu_op_registry().set_default_handler(default_dispatch,
                                        default_kernel_name_for);

  // ===========================================================
  // In-place op dispatches
  // -----------------------------------------------------------
  // Routes `aten::X_` to `aten::X.out`. After AOT's
  // alias_buffer_mutations_post_planning pass, an in-place op's
  // instruction args end with the emitter's synthetic `out` value_id
  // (which emit's spec dedup makes equal to `self`). The `.out`
  // kernel signature is `(self, ..., out)`; the reinplaced IR
  // produces exactly that layout.
  //
  // CRITICAL: keys are the FULL "<op>.<overload>" identity. Registering
  // `aten::pow_` (base only) would silently catch every overload of
  // pow_ and route them ALL to one kernel — wrong for ops like pow_
  // that have schema-incompatible Scalar/Tensor variants. Each overload
  // gets its own explicit entry.
  //
  // Pointwise unary (no overload, single .out variant).
  CPU_DISPATCH_OP(aten::relu_, "aten::relu.out");
  CPU_DISPATCH_OP(aten::relu6_, "aten::relu6.out");
  CPU_DISPATCH_OP(aten::sigmoid_, "aten::sigmoid.out");
  CPU_DISPATCH_OP(aten::tanh_, "aten::tanh.out");
  CPU_DISPATCH_OP(aten::exp_, "aten::exp.out");
  CPU_DISPATCH_OP(aten::expm1_, "aten::expm1.out");
  CPU_DISPATCH_OP(aten::log_, "aten::log.out");
  CPU_DISPATCH_OP(aten::log1p_, "aten::log1p.out");
  CPU_DISPATCH_OP(aten::log2_, "aten::log2.out");
  CPU_DISPATCH_OP(aten::log10_, "aten::log10.out");
  CPU_DISPATCH_OP(aten::neg_, "aten::neg.out");
  CPU_DISPATCH_OP(aten::abs_, "aten::abs.out");
  CPU_DISPATCH_OP(aten::sqrt_, "aten::sqrt.out");
  CPU_DISPATCH_OP(aten::rsqrt_, "aten::rsqrt.out");
  CPU_DISPATCH_OP(aten::reciprocal_, "aten::reciprocal.out");
  CPU_DISPATCH_OP(aten::square_, "aten::square.out");
  CPU_DISPATCH_OP(aten::cos_, "aten::cos.out");
  CPU_DISPATCH_OP(aten::sin_, "aten::sin.out");
  CPU_DISPATCH_OP(aten::tan_, "aten::tan.out");
  CPU_DISPATCH_OP(aten::cosh_, "aten::cosh.out");
  CPU_DISPATCH_OP(aten::sinh_, "aten::sinh.out");
  CPU_DISPATCH_OP(aten::asin_, "aten::asin.out");
  CPU_DISPATCH_OP(aten::acos_, "aten::acos.out");
  CPU_DISPATCH_OP(aten::atan_, "aten::atan.out");
  CPU_DISPATCH_OP(aten::asinh_, "aten::asinh.out");
  CPU_DISPATCH_OP(aten::acosh_, "aten::acosh.out");
  CPU_DISPATCH_OP(aten::atanh_, "aten::atanh.out");
  CPU_DISPATCH_OP(aten::erf_, "aten::erf.out");
  CPU_DISPATCH_OP(aten::erfc_, "aten::erfc.out");
  CPU_DISPATCH_OP(aten::sign_, "aten::sign.out");
  CPU_DISPATCH_OP(aten::ceil_, "aten::ceil.out");
  CPU_DISPATCH_OP(aten::floor_, "aten::floor.out");
  CPU_DISPATCH_OP(aten::round_, "aten::round.out");
  CPU_DISPATCH_OP(aten::trunc_, "aten::trunc.out");
  CPU_DISPATCH_OP(aten::frac_, "aten::frac.out");
  CPU_DISPATCH_OP(aten::silu_, "aten::silu.out");
  CPU_DISPATCH_OP(aten::gelu_, "aten::gelu.out");
  CPU_DISPATCH_OP(aten::elu_, "aten::elu.out");
  CPU_DISPATCH_OP(aten::leaky_relu_, "aten::leaky_relu.out");
  CPU_DISPATCH_OP(aten::hardsigmoid_, "aten::hardsigmoid.out");
  CPU_DISPATCH_OP(aten::hardswish_, "aten::hardswish.out");
  CPU_DISPATCH_OP(aten::logical_not_, "aten::logical_not.out");
  CPU_DISPATCH_OP(aten::bitwise_not_, "aten::bitwise_not.out");

  // Pointwise binary in-place. The IR already carries `alpha` at args[2]
  // for add_/sub_ (Int Scalar valued 1), so plain dispatch — the
  // 4-arg IR matches the .out kernel's (self, other, alpha, out)
  // signature directly.
  cpu_op_registry().register_op(
      "aten::add_.Tensor",
      [](CpuGraph& g, const std::vector<ValueRef>& a) {
        dispatch_kernel(g, a, "aten::add.out", /*pass_return_as_kernel_out=*/true);
      });
  cpu_op_registry().register_op(
      "aten::sub_.Tensor",
      [](CpuGraph& g, const std::vector<ValueRef>& a) {
        dispatch_kernel(g, a, "aten::sub.out", /*pass_return_as_kernel_out=*/true);
      });
  cpu_op_registry().register_op(
      "aten::mul_.Tensor",
      [](CpuGraph& g, const std::vector<ValueRef>& a) {
        dispatch_kernel(g, a, "aten::mul.out", /*pass_return_as_kernel_out=*/true);
      });
  cpu_op_registry().register_op(
      "aten::div_.Tensor",
      [](CpuGraph& g, const std::vector<ValueRef>& a) {
        dispatch_kernel(g, a, "aten::div.out", /*pass_return_as_kernel_out=*/true);
      });
  cpu_op_registry().register_op(
      "aten::atan2_",
      [](CpuGraph& g, const std::vector<ValueRef>& a) {
        dispatch_kernel(g, a, "aten::atan2.out", /*pass_return_as_kernel_out=*/true);
      });
  cpu_op_registry().register_op(
      "aten::logical_and_",
      [](CpuGraph& g, const std::vector<ValueRef>& a) {
        dispatch_kernel(g, a, "aten::logical_and.out", /*pass_return_as_kernel_out=*/true);
      });
  cpu_op_registry().register_op(
      "aten::logical_or_",
      [](CpuGraph& g, const std::vector<ValueRef>& a) {
        dispatch_kernel(g, a, "aten::logical_or.out", /*pass_return_as_kernel_out=*/true);
      });
  cpu_op_registry().register_op(
      "aten::logical_xor_",
      [](CpuGraph& g, const std::vector<ValueRef>& a) {
        dispatch_kernel(g, a, "aten::logical_xor.out", /*pass_return_as_kernel_out=*/true);
      });

  // pow_.Scalar(Tensor self, Scalar exp): the in-place form `square_`
  // decomposes into. Its .out kernel is named `pow.Tensor_Scalar_out`
  // (different overload-name convention: `_.Scalar` ↔ `.Tensor_Scalar_out`).
  cpu_op_registry().register_op(
      "aten::pow_.Scalar",
      [](CpuGraph& g, const std::vector<ValueRef>& a) {
        dispatch_kernel(g, a, "aten::pow.Tensor_Scalar_out",
                        /*pass_return_as_kernel_out=*/true);
      });

  // Misc — unambiguous (single overload).
  CPU_DISPATCH_OP(aten::clamp_, "aten::clamp.out");
  CPU_DISPATCH_OP(aten::clamp_min_, "aten::clamp_min.out");
  CPU_DISPATCH_OP(aten::clamp_max_, "aten::clamp_max.out");
  CPU_DISPATCH_OP(aten::addcmul_, "aten::addcmul.out");
  CPU_DISPATCH_OP(aten::addcdiv_, "aten::addcdiv.out");
  // ===========================================================
  // "Seed-then-write" out-variants used as in-place via remap.
  // -----------------------------------------------------------
  // These kernels start with a `memcpy(out, in, in.nbytes())` to seed
  // `out` with `in`'s contents, then write the modified positions.
  // When called with `in == out` (which our `_` → `.out` remap
  // deliberately arranges via Tensor(a!) aliasing), this seed memcpy
  // is `memcpy(p, p, n)` — UB by the C++ spec AND a wasted full-tensor
  // copy on the in-place hot path.
  //
  // TODO(upstream-portable-kernels): add `if (in.data_ptr() !=
  // out.data_ptr())` guard around the seed memcpy in each of these
  // kernels in `kernels/portable/cpu/op_*.cpp`. Better: extract a
  // shared `seed_out_from_in(out, in)` helper that does the alias
  // check once, and have all "seed-then-write" `.out` kernels call it.
  // Affects: index_put.out, index_add.out, scatter_add.out,
  //          masked_scatter.out, scatter.*_out, index_fill.*_out, ...
  CPU_DISPATCH_OP(aten::index_put_, "aten::index_put.out");
  CPU_DISPATCH_OP(aten::index_add_, "aten::index_add.out");
  CPU_DISPATCH_OP(aten::scatter_add_, "aten::scatter_add.out");
  CPU_DISPATCH_OP(aten::masked_scatter_, "aten::masked_scatter.out");

  // hardtanh_ has only the (min_val, max_val) form — single overload.
  CPU_DISPATCH_OP(aten::hardtanh_, "aten::hardtanh.out");

  // ===========================================================
  // Functional `.out` ops that may appear in IR (NOT in-place forms).
  // Registered explicitly so the router's base-name `can_run` probe
  // succeeds. Default dispatch (pass_return_as_kernel_out=false via
  // default_dispatch path) is the right semantics — IR's args already
  // include the return slot.
  // -----------------------------------------------------------
  // _to_copy: dtype/device cast emitted by AOT for things like `x + 0`
  // when AOT inserts an implicit cast. Reaches our backend because it's
  // not an in-place op.
  cpu_op_registry().register_op(
      "aten::_to_copy.out",
      [](CpuGraph& g, const std::vector<ValueRef>& a) {
        // IR op IS the kernel (no remap). Args already include the
        // return slot per emit's convention; no extra dummy needed.
        dispatch_kernel(g, a, "aten::_to_copy.out",
                        /*pass_return_as_kernel_out=*/false);
      });
  cpu_op_registry().register_op(
      "aten::clone.out",
      [](CpuGraph& g, const std::vector<ValueRef>& a) {
        dispatch_kernel(g, a, "aten::clone.out",
                        /*pass_return_as_kernel_out=*/false);
      });

  // ===========================================================
  // Multi-overload in-place ops — DROPPED to avoid silent mis-dispatch.
  // -----------------------------------------------------------
  // The following have schema-incompatible overloads; safe routing
  // requires explicit per-overload registration, which has not been
  // verified against the AOT lowering's actual emit names.
  //
  // To re-enable, register each overload with its full name + verify
  // the kernel-arg shape matches the dispatch_kernel default stack:
  //
  //   aten::pow_           (.Scalar, .Tensor)
  //   aten::remainder_     (.Scalar, .Tensor)
  //   aten::fmod_          (.Scalar, .Tensor)
  //   aten::bitwise_and_   (.Scalar, .Tensor)
  //   aten::bitwise_or_    (.Scalar, .Tensor)
  //   aten::bitwise_xor_   (.Scalar, .Tensor)
  //   aten::masked_fill_   (.Scalar, .Tensor)
  //   aten::scatter_       (.src, .value, with optional reduce)
  //   aten::index_fill_    (.int_Scalar, .int_Tensor)
  //   aten::lerp_          (.Scalar, .Tensor)
  //   aten::fill_          (.Scalar, .Tensor)
  // ===========================================================

  // copy_ uses a custom dispatcher (drops the formal out arg, calls the
  // 3-arg in-place kernel). The IR emits 4 args but the in-place kernel
  // only takes 3; the formal out is just a memory-plan placeholder.
  CPU_REGISTER_OP(aten::copy_, dispatch_copy_inplace);
}

#undef CPU_DISPATCH_OP

} // namespace portable
} // namespace backends
} // namespace executorch
