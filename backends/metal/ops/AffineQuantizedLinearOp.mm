/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "AffineQuantizedLinearOp.h"

#include <executorch/backends/metal/ops/registry/MetalOpRegistry.h>
#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/backends/metal/core/MetalDeviceInfo.h>
#include <executorch/backends/metal/ops/registry/OpUtils.h>
#include <executorch/backends/metal/ops/AffineQuantizedLinearMlxJit.h>
#include <executorch/backends/metal/ops/MatMulCommon.h>
#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/platform/log.h>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::Error;

//===----------------------------------------------------------------------===//
// computeOutputShape — out shape == x.sizes()[:-1] + [N], where
// N = wq.size(0).
//===----------------------------------------------------------------------===//

std::vector<SizesType> AffineQuantizedLinearOp::computeOutputShape(
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs) const {
  if (inputs.size() < 2 || !inputs[0]->isTensor() || !inputs[1]->isTensor()) {
    return {};
  }
  const auto& x  = inputs[0]->toTensor();
  const auto& wq = inputs[1]->toTensor();
  std::vector<SizesType> out;
  out.reserve(x.dim());
  for (ssize_t i = 0; i < x.dim() - 1; ++i) {
    out.push_back(static_cast<SizesType>(x.sizes()[i]));
  }
  out.push_back(static_cast<SizesType>(wq.sizes()[0]));
  return out;
}

//===----------------------------------------------------------------------===//
// dispatch — parse 7 inputs, validate, route across MLX's tile families.
//===----------------------------------------------------------------------===//

namespace {

// Port of MLX's get_qmv_batch_limit (mlx/backend/metal/quantized.cpp:84-126).
// Returns the M threshold above which we should use a matrix-matrix kernel
// (qmm_t / qmm_t_splitk / qmm_t_nax) instead of the vector kernel (qmv*).
// Below the limit, qmv handles M >= 1 via its M grid dim — this is generally
// faster for skinny shapes.
// Tabular by (architecture_gen, architecture_size, K, N). We treat:
//   - Apple9+ (gen 13+) as the "modern" branch
//   - non-Apple9 as the older branch
//   - 'd' (Mac large/desktop) vs default ('p'/'g'/'m' — Phone, Mac base, etc.)
// MLX uses the device's `architecture_gen` and `architecture_size`. We
// approximate via MTLGPUFamilyApple9 (≈ gen 13+) and the existing
// DeviceTier enum (MacUltra ≈ 'd' size).
inline int getQmvBatchLimit(int K, int N, MetalStream* stream) {
  const bool nax = MetalDeviceInfo::isNaxAvailable();
  const auto tier = MetalDeviceInfo::tier();
  const bool large = (tier == DeviceTier::MacUltra);  // 'd' (desktop) ≈ Ultra

  if (nax) {
    // Apple9+ (gen 13+): MLX returns 32/18/12 for 'd', 18/12/8 for default.
    if (large) {
      if (K <= 2048 && N <= 2048) return 32;
      if (K <= 4096 && N <= 4096) return 18;
      return 12;
    }
    if (K <= 2048 && N <= 2048) return 18;
    if (K <= 4096 && N <= 4096) return 12;
    return 8;
  }
  // Older arch: MLX returns 32/18/12 for 'd', 14/10/6 for default.
  if (large) {
    if (K <= 2048 && N <= 2048) return 32;
    if (K <= 4096 && N <= 4096) return 18;
    return 12;
  }
  if (K <= 2048 && N <= 2048) return 14;
  if (K <= 4096 && N <= 4096) return 10;
  return 6;
}

// MLX's qmm() routes to qmm_nax when:
//   metal::is_nax_available() && transpose && (K % 64 == 0) &&
//   (env::enable_tf32() || x.dtype() != float32)
// We always have transpose=true for our op.
// We ALSO require `M >= 64` (= BM tile size) — empirically, NAX is slower
// than plain qmm_t for M < 64 because the BM=64 tile wastes 100*(1-M/64)%
// of compute on padding (e.g., seq=16 → 75% waste, measured 1.6× slower
// than qmm_t). At M >= 64 NAX wins by 5-10% on prefill (Llama-7B bench).
inline bool shouldUseQmmNax(MetalStream* stream, int M, int K,
                             executorch::aten::ScalarType dtype) {
  return MetalDeviceInfo::isNaxAvailable() && (M >= 64) && (K % 64 == 0) &&
         (tf32Enabled() || dtype != executorch::aten::ScalarType::Float);
}

}  // anonymous namespace

void AffineQuantizedLinearOp::dispatch(
    MetalStream* stream,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) {
  if (inputs.size() < 7) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: expected 7 inputs (x, wq, ws, wz?, "
           "b?, group_size, nbit); got %zu", inputs.size());
    return;
  }
  if (outputs.size() < 1) {
    ET_LOG(Error, "AffineQuantizedLinearOp: expected 1 output");
    return;
  }

  // ---- Parse inputs ----
  auto& x  = inputs[0]->toTensor();
  auto& wq = inputs[1]->toTensor();
  auto& ws = inputs[2]->toTensor();
  auto& y  = outputs[0]->toTensor();

  // wz: optional
  const executorch::aten::Tensor* wz = nullptr;
  if (inputs[3] && inputs[3]->isTensor()) {
    wz = &inputs[3]->toTensor();
  }
  // b (linear bias): optional
  const executorch::aten::Tensor* b = nullptr;
  if (inputs[4] && inputs[4]->isTensor()) {
    b = &inputs[4]->toTensor();
  }
  // group_size, nbit (Scalars)
  int64_t group_size = 0, nbit = 0;
  if (inputs[5]) {
    auto& s = *inputs[5];
    if (s.isInt())         group_size = s.toInt();
    else if (s.isDouble()) group_size = static_cast<int64_t>(s.toDouble());
  }
  if (inputs[6]) {
    auto& s = *inputs[6];
    if (s.isInt())         nbit = s.toInt();
    else if (s.isDouble()) nbit = static_cast<int64_t>(s.toDouble());
  }

  // ---- Validate ----
  const auto dtype = x.scalar_type();
  if (!supports(dtype)) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: unsupported activation dtype %d "
           "(only fp32/fp16/bf16)", int(dtype));
    return;
  }
  if (wq.scalar_type() != executorch::aten::ScalarType::Byte) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: wq must be uint8 (Byte), got dtype %d",
           int(wq.scalar_type()));
    return;
  }
  if (ws.scalar_type() != dtype) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: ws dtype %d must match x dtype %d",
           int(ws.scalar_type()), int(dtype));
    return;
  }
  if (wz && wz->scalar_type() != dtype) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: wz dtype %d must match x dtype %d",
           int(wz->scalar_type()), int(dtype));
    return;
  }
  if (b && b->scalar_type() != dtype) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: b dtype %d must match x dtype %d",
           int(b->scalar_type()), int(dtype));
    return;
  }
  if (nbit != 4 && nbit != 8) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: nbit must be 4 or 8 (initial v0); got %lld",
           (long long)nbit);
    return;  // Error::NotImplemented at op surface
  }
  if (group_size != 32 && group_size != 64 && group_size != 128 &&
      group_size != 256 && group_size != 512 && group_size != 1024) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: group_size must be one of "
           "{32,64,128,256,512,1024}; got %lld", (long long)group_size);
    return;
  }
  if (wq.dim() != 2) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: wq must be 2-D [N, K*nbit/8]; got dim=%d",
           int(wq.dim()));
    return;
  }
  if (x.dim() < 2) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: x must be at least 2-D [..., K]; got dim=%d",
           int(x.dim()));
    return;
  }

  // ---- Derive shapes ----
  const int N = static_cast<int>(wq.sizes()[0]);
  const int K = static_cast<int>(x.sizes()[x.dim() - 1]);
  // M = product of all but last x dim. For [..., K] input, this collapses
  // batch + sequence dims into a single M for the matmul.
  int M = 1;
  for (ssize_t i = 0; i < x.dim() - 1; ++i) {
    M *= static_cast<int>(x.sizes()[i]);
  }

  // Shape consistency.
  const int wq_inner = static_cast<int>(wq.sizes()[1]);
  const int expected_wq_inner = (K * static_cast<int>(nbit) + 7) / 8;
  if (wq_inner != expected_wq_inner) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: wq.size(1)=%d != K*nbit/8=%d (K=%d, nbit=%lld)",
           wq_inner, expected_wq_inner, K, (long long)nbit);
    return;
  }
  if (K % static_cast<int>(group_size) != 0) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: K=%d must be divisible by group_size=%lld",
           K, (long long)group_size);
    return;
  }
  const int n_groups = K / static_cast<int>(group_size);
  if (ws.dim() != 2 || ws.sizes()[0] != N || ws.sizes()[1] != n_groups) {
    ET_LOG(Error,
           "AffineQuantizedLinearOp: ws shape must be [N=%d, K/gs=%d]; got dim=%d",
           N, n_groups, int(ws.dim()));
    return;
  }
  if (wz) {
    if (wz->dim() != 2 || wz->sizes()[0] != N || wz->sizes()[1] != n_groups) {
      ET_LOG(Error,
             "AffineQuantizedLinearOp: wz shape must be [N=%d, K/gs=%d]; got dim=%d",
             N, n_groups, int(wz->dim()));
      return;
    }
  }
  if (b) {
    if (b->dim() != 1 || b->sizes()[0] != N) {
      ET_LOG(Error,
             "AffineQuantizedLinearOp: b shape must be [N=%d]; got dim=%d",
             N, int(b->dim()));
      return;
    }
  }

  // ---- Materialize biases tensor ----
  // The JIT dispatch helpers always need a real biases tensor at slot 2.
  // If the user passed `wz=None` (symmetric quant), allocate a zeros
  // scratch buffer and wrap it as a Tensor view so the helpers can take
  // it as `const Tensor&`. This keeps the JIT layer dtype/shape-agnostic
  // — translation from "no zero-points" to "zero-filled biases" lives
  // here at the registered op level, NOT inside JIT helpers.
  void* zeros_scratch = nullptr;
  size_t zeros_bytes = 0;
  std::vector<executorch::aten::SizesType> zeros_sizes_storage;
  std::vector<executorch::aten::StridesType> zeros_strides_storage;
  std::vector<executorch::runtime::etensor::TensorImpl::DimOrderType>
      zeros_dim_order_storage;
  alignas(executorch::runtime::etensor::TensorImpl)
      std::byte zeros_impl_storage[sizeof(executorch::runtime::etensor::TensorImpl)];
  bool zeros_constructed = false;

  // Synthesized biases tensor when needed.
  executorch::aten::Tensor wz_synth =
      *reinterpret_cast<executorch::aten::Tensor*>(zeros_impl_storage);

  if (wz == nullptr) {
    size_t element_size = 4;
    switch (dtype) {
      case executorch::aten::ScalarType::Half:
      case executorch::aten::ScalarType::BFloat16: element_size = 2; break;
      default: element_size = 4; break;
    }
    zeros_bytes = size_t(N) * size_t(n_groups) * element_size;
    zeros_scratch = stream->allocator().alloc(zeros_bytes);
    ET_CHECK_MSG(zeros_scratch != nullptr,
                 "AffineQuantizedLinearOp: zeros-scratch alloc(%zu) failed",
                 zeros_bytes);
    std::memset(zeros_scratch, 0, zeros_bytes);

    // Build a non-owning Tensor view over the scratch buffer.
    zeros_sizes_storage = {static_cast<executorch::aten::SizesType>(N),
                           static_cast<executorch::aten::SizesType>(n_groups)};
    zeros_strides_storage = {static_cast<executorch::aten::StridesType>(n_groups),
                             1};
    zeros_dim_order_storage = {0, 1};
    auto* impl = new (zeros_impl_storage)
        executorch::runtime::etensor::TensorImpl(
            dtype, /*dim=*/2,
            zeros_sizes_storage.data(),
            zeros_scratch,
            zeros_dim_order_storage.data(),
            zeros_strides_storage.data(),
            executorch::runtime::TensorShapeDynamism::STATIC);
    wz_synth = executorch::aten::Tensor(impl);
    zeros_constructed = true;
  }

  // Reference handed to JIT helpers — either user's wz or our synthesized
  // zeros view. After this point the helpers see a non-nullable Tensor&.
  const executorch::aten::Tensor& wz_for_jit =
      (wz != nullptr) ? *wz : wz_synth;

  // RAII to free the scratch + run TensorImpl dtor on early return / scope exit.
  struct ZerosScratchGuard {
    MetalStream* stream;
    void* ptr;
    void* impl_storage;
    bool constructed;
    ~ZerosScratchGuard() {
      if (constructed) {
        reinterpret_cast<executorch::runtime::etensor::TensorImpl*>(
            impl_storage)->~TensorImpl();
      }
      if (ptr) stream->allocator().free(ptr);
    }
  };
  ZerosScratchGuard zeros_guard{stream, zeros_scratch, zeros_impl_storage,
                                zeros_constructed};

  // ---- Route ----
  // Reshape y to [M, N] equivalent — y already has shape [..., N] from
  // the AOTI shim allocation. The dispatch helpers don't care about the
  // batch dims since they treat the buffer as [M*N] flat.
  // 4-way routing (mirrors MLX QuantizedMatmul::eval_gpu, transpose=true,
  // B=1 — our op signature has no batch dim):
  //   M < vector_limit                           → qmv* (qmv_quad / qmv_fast / qmv)
  //   M >= vector_limit, NAX eligible            → qmm_t_nax  (Apple9+, K%64==0)
  //   M >= vector_limit, splitk eligible (no NAX)→ qmm_t_splitk (+ accum)
  //   M >= vector_limit, otherwise               → qmm_t  (existing default)
  // NAX-before-splitk ordering note: MLX upstream tries splitk first and
  // falls through to NAX inside qmm(). On Apple9 we measured NAX winning
  // over splitk for prefill (seq=128/512: ~+10%); splitk only helped when
  // M was tiny (~16) AND NAX was also unavailable. So we invert MLX's
  // order: NAX wins when both are eligible. Splitk remains as a fallback
  // for non-NAX hardware (Apple7/8) and shapes where K % 64 != 0.
  const int vector_limit = getQmvBatchLimit(N, K, stream);
  if (M < vector_limit) {
    aql_helpers::dispatchAffineQmvViaMlxJit(
        stream, x, wq, ws, wz_for_jit, y,
        M, N, K,
        static_cast<int>(group_size), static_cast<int>(nbit),
        dtype);
  } else if (shouldUseQmmNax(stream, M, K, dtype)) {
    aql_helpers::dispatchAffineQmmTNaxViaMlxJit(
        stream, x, wq, ws, wz_for_jit, y,
        M, N, K,
        static_cast<int>(group_size), static_cast<int>(nbit),
        dtype);
  } else {
    // No NAX (or K not divisible by 64). Try splitk first; if it
    // degenerates to split_k <= 1, fall back to plain qmm_t.
    bool dispatched = aql_helpers::dispatchAffineQmmTSplitKViaMlxJit(
        stream, x, wq, ws, wz_for_jit, y,
        M, N, K,
        static_cast<int>(group_size), static_cast<int>(nbit),
        dtype);
    if (!dispatched) {
      aql_helpers::dispatchAffineQmmTViaMlxJit(
          stream, x, wq, ws, wz_for_jit, y,
          M, N, K,
          static_cast<int>(group_size), static_cast<int>(nbit),
          dtype);
    }
  }

  // ---- Post-add linear bias ----
  if (b != nullptr) {
    // Use the registered AddOp through the registry to compute y += b
    // (broadcast over the M dim). AddOp's binary kernel handles
    // broadcasting via its standard PyTorch semantics.
    static MetalOp* addOp = MetalOpRegistry::shared().get("aten::add.Tensor");
    if (addOp == nullptr) {
      // Fallback: try alternate registration name. If not present, log
      // and skip (test will catch the bias-handling regression).
      addOp = MetalOpRegistry::shared().get("aten::add");
    }
    if (addOp != nullptr) {
      // Build EValues for in-place add: y = y + b
      runtime::EValue addInEVs[2];
      runtime::EValue* addInPtrs[2];
      addInEVs[0] = runtime::EValue(y);
      addInEVs[1] = runtime::EValue(*b);
      addInPtrs[0] = &addInEVs[0];
      addInPtrs[1] = &addInEVs[1];
      runtime::EValue addOutEV(y);
      runtime::EValue* addOutPtr = &addOutEV;
      addOp->dispatch(
          stream,
          ::executorch::runtime::Span<::executorch::runtime::EValue*>(addInPtrs, 2),
          ::executorch::runtime::Span<::executorch::runtime::EValue*>(&addOutPtr, 1));
    } else {
      ET_LOG(Error,
             "AffineQuantizedLinearOp: bias post-add requested but "
             "aten::add op not in MetalOpRegistry");
    }
  }
}

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
