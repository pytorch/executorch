/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AOTI op shims that route to MetalOpRegistry.
//
// Each AOTI extern "C" entry is a thin wrapper that:
//   - Looks up its op via lookupRegistryOp (cached in a function-local
//     static so the registry hit happens once per process).
//   - Builds EValues from the AOTI ABI args via OpInvocation<NIn, NOut>.
//   - (Optionally) allocates an output tensor via allocateMpsOutput.
//   - Calls inv.dispatch(op).
//
// To add a new fallback op:
//   1. Register the op in backends/metal/ops/registry/MetalOpRegistry.mm.
//   2. Add an extern "C" entry below following the pattern of one of the
//      existing ops.
//
// Contract: every op consumed here must be registered in MetalOpRegistry
// before the first AOTI shim invocation. Each shim's per-call-site
// `static MetalOp* op = lookupRegistryOp(...)` caches the result on
// first call; if the registry doesn't yet contain the op at that point,
// the slot caches nullptr and every subsequent call returns
// NotImplemented. MetalOpRegistry's static-init runs at lib load, so in
// practice this contract is always satisfied — but if a future change
// introduces lazy registration, that pattern would need to revisit the
// caching here.
//
// kMaxOpInputs / kMaxOpOutputs (defined below) cap the slot counts that
// OpInvocation<NIn, NOut> supports. Bumping them requires editing this
// file; both checks are static_assert'd so a too-large NIn/NOut fails
// at compile time with a clear message.

#import <Metal/Metal.h>

#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_dtype.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_tensor.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_types.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/runtime.h>
#include <executorch/backends/apple/metal/runtime/shims/v2/stack_tensor_view.h>
#include <executorch/backends/metal/ops/registry/MetalOp.h>
#include <executorch/backends/metal/ops/registry/MetalOpRegistry.h>
#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/log.h>

#include <array>
#include <cstdint>
#include <vector>

namespace executorch {
namespace backends {
namespace metal {

namespace {

using executorch::backends::metal_v2::MetalOp;
using executorch::backends::metal_v2::MetalOpRegistry;

// Returns the registered op or nullptr (with a one-shot ET_LOG) on miss.
// Designed to be cached at each call site via a function-local static —
// each unique missing op then logs exactly once per process.
//
// Returns nullptr (rather than aborting) so the AOTI caller receives
// Error::NotImplemented and can decide between a CPU fallback and a
// user-facing error.
MetalOp* lookupRegistryOp(const char* op_name, const char* aoti_symbol) {
  MetalOp* op = MetalOpRegistry::shared().get(op_name);
  if (!op) {
    ET_LOG(Error,
        "%s: op '%s' not in MetalOpRegistry — returning NotImplemented. "
        "Add registration in backends/metal/ops/registry/MetalOpRegistry.mm "
        "to enable this op.",
        aoti_symbol, op_name);
  }
  return op;
}

constexpr size_t kMaxOpInputs = 8;
constexpr size_t kMaxOpOutputs = 4;

AOTITorchError dispatchOp(
    MetalOp* op,
    ::executorch::runtime::EValue* const* in_ptrs, size_t in_n,
    ::executorch::runtime::EValue* const* out_ptrs, size_t out_n);

// Per-call invocation builder for op-registry dispatch. Owns stack-resident
// StackTensorViews / EValues / pointer arrays sized to NIn / NOut.
//
// Restriction: NIn > 0 AND NOut > 0. Ops with no inputs (e.g. aten::ones)
// or no outputs (in-place ops) are not currently representable; add a
// specialization or relax the assertion if needed.
template <size_t NIn, size_t NOut>
struct OpInvocation {
  static_assert(NIn > 0 && NOut > 0,
      "OpInvocation requires at least one input AND one output.");
  static_assert(NIn <= kMaxOpInputs, "Bump kMaxOpInputs");
  static_assert(NOut <= kMaxOpOutputs, "Bump kMaxOpOutputs");

  StackTensorView in_views[NIn];
  StackTensorView out_views[NOut];
  ::executorch::runtime::EValue in_evs[NIn];
  ::executorch::runtime::EValue out_evs[NOut];
  ::executorch::runtime::EValue* in_ptrs[NIn];
  ::executorch::runtime::EValue* out_ptrs[NOut];

  // `t` must outlive the dispatch.
  void setInputTensor(size_t i, Tensor* t) {
    in_evs[i] = ::executorch::runtime::EValue(in_views[i].makeView(*t));
    in_ptrs[i] = &in_evs[i];
  }
  void setInputOptionalTensor(size_t i, Tensor* t) {
    in_evs[i] = t ? ::executorch::runtime::EValue(in_views[i].makeView(*t))
                  : ::executorch::runtime::EValue();
    in_ptrs[i] = &in_evs[i];
  }
  template <typename T>
  void setInputScalar(size_t i, T v) {
    in_evs[i] = ::executorch::runtime::EValue(v);
    in_ptrs[i] = &in_evs[i];
  }
  void setInputNone(size_t i) {
    in_evs[i] = ::executorch::runtime::EValue();
    in_ptrs[i] = &in_evs[i];
  }
  void setOutputTensor(size_t i, Tensor* t) {
    out_evs[i] = ::executorch::runtime::EValue(out_views[i].makeView(*t));
    out_ptrs[i] = &out_evs[i];
  }

  AOTITorchError dispatch(MetalOp* op) {
    return dispatchOp(op, in_ptrs, NIn, out_ptrs, NOut);
  }
};

// Heap-alloc-free dispatch over pre-built EValue arrays.
AOTITorchError dispatchOp(
    MetalOp* op,
    executorch::runtime::EValue* const* in_ptrs, size_t in_n,
    executorch::runtime::EValue* const* out_ptrs, size_t out_n) {
  if (!op) {
    ET_LOG(Error, "dispatchOp: op is null");
    return Error::NotImplemented;
  }
  if (in_n > kMaxOpInputs || out_n > kMaxOpOutputs) {
    ET_LOG(Error,
        "dispatchOp: op '%s' exceeds max in=%zu/%zu out=%zu/%zu",
        op->name(), in_n, kMaxOpInputs, out_n, kMaxOpOutputs);
    return Error::InvalidArgument;
  }
  op->dispatch(
      getMetalStream(),
      ::executorch::runtime::Span<::executorch::runtime::EValue*>(
          const_cast<::executorch::runtime::EValue**>(in_ptrs), in_n),
      ::executorch::runtime::Span<::executorch::runtime::EValue*>(
          const_cast<::executorch::runtime::EValue**>(out_ptrs), out_n));
  return Error::Ok;
}

// Allocates a fresh MPS tensor + registers it (refcount = 1) atomically.
// The handle's storage is freed via aoti_torch_delete_tensor_object.
// Returns nullptr on failure (already ET_LOG'd).
AOTITensorHandle allocateMpsOutput(
    const std::vector<int64_t>& sizes, int32_t dtype) {
  size_t bytes = dtype_to_bytes(dtype);
  for (int64_t s : sizes) bytes *= static_cast<size_t>(s);

  void* ptr = nullptr;
  AOTITorchError err = aoti_torch_mps_malloc(&ptr, bytes);
  if (err != Error::Ok || !ptr) {
    ET_LOG(Error, "allocateMpsOutput: malloc(%zu) failed", bytes);
    return nullptr;
  }

  std::vector<int64_t> strides = compute_contiguous_strides(sizes);

  AOTITensorHandle handle = nullptr;
  err = aoti_torch_create_owned_tensor_from_blob_v2(
      ptr, static_cast<int64_t>(sizes.size()),
      const_cast<int64_t*>(sizes.data()), strides.data(),
      dtype, &handle);
  if (err != Error::Ok || !handle) {
    ET_LOG(Error, "allocateMpsOutput: create_owned_tensor_from_blob_v2 failed");
    aoti_torch_mps_free(ptr);
    return nullptr;
  }
  return handle;
}

}  // namespace

extern "C" {

// aten::mm

AOTITorchError aoti_torch_mps_mm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2) {
  if (!out || !self || !mat2) return Error::InvalidArgument;
  static MetalOp* op = lookupRegistryOp("aten::mm", "aoti_torch_mps_mm_out");
  if (!op) return Error::NotImplemented;

  OpInvocation<2, 1> inv;
  inv.setInputTensor(0, reinterpret_cast<Tensor*>(self));
  inv.setInputTensor(1, reinterpret_cast<Tensor*>(mat2));
  inv.setOutputTensor(0, reinterpret_cast<Tensor*>(out));
  return inv.dispatch(op);
}

// aten::bmm

AOTITorchError aoti_torch_mps_bmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2) {
  if (!out || !self || !mat2) return Error::InvalidArgument;
  static MetalOp* op = lookupRegistryOp("aten::bmm", "aoti_torch_mps_bmm_out");
  if (!op) return Error::NotImplemented;

  OpInvocation<2, 1> inv;
  inv.setInputTensor(0, reinterpret_cast<Tensor*>(self));
  inv.setInputTensor(1, reinterpret_cast<Tensor*>(mat2));
  inv.setOutputTensor(0, reinterpret_cast<Tensor*>(out));
  return inv.dispatch(op);
}

// torchao 4-bit quantized linear.
// Schema: (x, wq, ws, wz?, b?, group_size, nbit).

AOTITorchError aoti_torch_mps__linear_fp_act_4bit_weight(
    AOTITensorHandle A,
    AOTITensorHandle B,
    int64_t group_size,
    AOTITensorHandle S,
    AOTITensorHandle Z,
    AOTITensorHandle* ret) {
  if (!A || !B || !S || !ret) {
    ET_LOG(Error, "[v2 4bit] null required handles");
    return Error::InvalidArgument;
  }
  static MetalOp* op = lookupRegistryOp(
      "executorch_native::affine_quantized_linear.default",
      "aoti_torch_mps__linear_fp_act_4bit_weight");
  if (!op) return Error::NotImplemented;

  auto* a_t = reinterpret_cast<Tensor*>(A);
  auto* b_t = reinterpret_cast<Tensor*>(B);
  auto* s_t = reinterpret_cast<Tensor*>(S);
  auto* z_t = reinterpret_cast<Tensor*>(Z);

  if (a_t->dim() < 2 || b_t->dim() != 2) {
    ET_LOG(Error, "[v2 4bit] bad ranks A=%dD B=%dD",
        int(a_t->dim()), int(b_t->dim()));
    return Error::InvalidArgument;
  }

  // Output shape = A.shape[:-1] + [N], N = B.shape[0].
  std::vector<int64_t> out_sizes;
  out_sizes.reserve(a_t->dim());
  for (ssize_t i = 0; i < a_t->dim() - 1; ++i) out_sizes.push_back(a_t->sizes()[i]);
  out_sizes.push_back(b_t->sizes()[0]);

  AOTITensorHandle out_handle = allocateMpsOutput(
      out_sizes, static_cast<int32_t>(a_t->dtype()));
  if (!out_handle) return Error::MemoryAllocationFailed;

  OpInvocation<7, 1> inv;
  inv.setInputTensor(0, a_t);
  inv.setInputTensor(1, b_t);
  inv.setInputTensor(2, s_t);
  inv.setInputOptionalTensor(3, z_t);
  inv.setInputNone(4);                            // bias = None.
  inv.setInputScalar(5, group_size);
  inv.setInputScalar(6, static_cast<int64_t>(4)); // nbit = 4.
  inv.setOutputTensor(0, reinterpret_cast<Tensor*>(out_handle));

  AOTITorchError err = inv.dispatch(op);
  if (err != Error::Ok) {
    aoti_torch_delete_tensor_object(out_handle);
    return err;
  }

  *ret = out_handle;
  return Error::Ok;
}

// SDPA (scaled-dot-product attention).
// Schema: (Q, K, V, attn_mask?, dropout_p, is_causal, scale?).

AOTITorchError aoti_torch_mps__scaled_dot_product_attention_math_for_mps(
    AOTITensorHandle query,
    AOTITensorHandle key,
    AOTITensorHandle value,
    AOTITensorHandle* attn_mask,
    double dropout_p,
    int32_t is_causal,
    AOTITensorHandle* dropout_mask,
    double* scale,
    AOTITensorHandle* ret0,
    AOTITensorHandle* ret1) {
  if (!query || !key || !value || !ret0 || !ret1) {
    ET_LOG(Error, "[v2 sdpa] null required handles");
    return Error::InvalidArgument;
  }
  if (dropout_p != 0.0) {
    ET_LOG(Error, "[v2 sdpa] dropout_p != 0 not implemented (got %f)", dropout_p);
    return Error::NotImplemented;
  }
  if (dropout_mask && *dropout_mask) {
    ET_LOG(Error, "[v2 sdpa] dropout_mask not implemented");
    return Error::NotImplemented;
  }
  static MetalOp* op = lookupRegistryOp(
      "aten::scaled_dot_product_attention.default",
      "aoti_torch_mps__scaled_dot_product_attention_math_for_mps");
  if (!op) return Error::NotImplemented;

  auto* q_t = reinterpret_cast<Tensor*>(query);
  auto* k_t = reinterpret_cast<Tensor*>(key);
  auto* v_t = reinterpret_cast<Tensor*>(value);
  auto* mask_t = (attn_mask && *attn_mask) ? reinterpret_cast<Tensor*>(*attn_mask) : nullptr;

  // ret0 (SDPA output) shape = Q's shape, dtype = Q's dtype.
  std::vector<int64_t> out_sizes(q_t->sizes().begin(), q_t->sizes().end());
  const int32_t dtype = static_cast<int32_t>(q_t->dtype());
  AOTITensorHandle out_handle = allocateMpsOutput(out_sizes, dtype);
  if (!out_handle) return Error::MemoryAllocationFailed;

  OpInvocation<7, 1> inv;
  inv.setInputTensor(0, q_t);
  inv.setInputTensor(1, k_t);
  inv.setInputTensor(2, v_t);
  inv.setInputOptionalTensor(3, mask_t);
  inv.setInputScalar(4, dropout_p);
  inv.setInputScalar(5, is_causal != 0);
  if (scale) {
    inv.setInputScalar(6, *scale);
  } else {
    inv.setInputNone(6);
  }
  inv.setOutputTensor(0, reinterpret_cast<Tensor*>(out_handle));

  AOTITorchError err = inv.dispatch(op);
  if (err != Error::Ok) {
    aoti_torch_delete_tensor_object(out_handle);
    return err;
  }

  // ret1 (attention weights). Shape = [B, Hq, qLen, kvLen] for the
  // standard 4-D SDPA layout. SDPAOp doesn't fill these — we allocate
  // backing storage at the right shape so callers that introspect
  // ret1.sizes() / numel() see correct values.
  std::vector<int64_t> aw_sizes;
  aw_sizes.reserve(4);
  if (q_t->dim() == 4 && k_t->dim() == 4) {
    aw_sizes.push_back(q_t->sizes()[0]);  // B
    aw_sizes.push_back(q_t->sizes()[1]);  // Hq
    aw_sizes.push_back(q_t->sizes()[2]);  // qLen
    aw_sizes.push_back(k_t->sizes()[2]);  // kvLen
  } else {
    // Non-standard rank — best-effort fallback. Element count is NOT
    // the true attn-weight count; callers should not introspect numel.
    aw_sizes.assign(q_t->sizes().begin(), q_t->sizes().end());
    ET_LOG(Error, "[v2 sdpa] Q rank %d / K rank %d — using Q's shape for ret1",
        int(q_t->dim()), int(k_t->dim()));
  }
  AOTITensorHandle aw_handle = allocateMpsOutput(aw_sizes, dtype);
  if (!aw_handle) {
    aoti_torch_delete_tensor_object(out_handle);
    return Error::MemoryAllocationFailed;
  }

  *ret0 = out_handle;
  *ret1 = aw_handle;
  return Error::Ok;
}

// convolution.
// ConvOp does not yet exist in MetalOpRegistry. Once it lands, fill in
// the EValue marshalling below — schema: (input, weight, bias?, stride[],
// padding[], dilation[], transposed, output_padding[], groups).

AOTITorchError aoti_torch_mps_convolution(
    AOTITensorHandle input,
    AOTITensorHandle weight,
    AOTITensorHandle* /*bias*/,
    const int64_t* /*stride*/, int64_t /*stride_len_*/,
    const int64_t* /*padding*/, int64_t /*padding_len_*/,
    const int64_t* /*dilation*/, int64_t /*dilation_len_*/,
    int32_t /*transposed*/,
    const int64_t* /*output_padding*/, int64_t /*output_padding_len_*/,
    int64_t /*groups*/,
    AOTITensorHandle* ret0) {
  if (!input || !weight || !ret0) return Error::InvalidArgument;

  static MetalOp* op = lookupRegistryOp(
      "aten::convolution", "aoti_torch_mps_convolution");
  if (!op) return Error::NotImplemented;

  ET_LOG(Error,
      "aoti_torch_mps_convolution: ConvOp is registered but the EValue "
      "marshalling has not been wired up. Add the schema (input, weight, "
      "bias?, stride[], padding[], dilation[], transposed, "
      "output_padding[], groups) to this shim and dispatch via OpInvocation.");
  return Error::NotImplemented;
}

}  // extern "C"

}  // namespace metal
}  // namespace backends
}  // namespace executorch
