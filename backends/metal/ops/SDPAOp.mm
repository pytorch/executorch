/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "SDPAOp.h"

#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/backends/metal/core/MetalDeviceInfo.h>
#include <executorch/backends/metal/ops/registry/OpUtils.h>
#include <executorch/backends/metal/ops/MatMulCommon.h>  // tf32Enabled()
#include <executorch/backends/metal/ops/SdpaMlxJit.h>
#include <executorch/runtime/platform/log.h>

#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::Error;

namespace {

// Architecture-suffix detection — mirrors MetalStream::getDefaultFlushInterval
// and matches what MLX uses (d.get_architecture().back()) to choose
// vector-2pass `blocks` heuristic.
//   's' → Apple7-Apple8 (M-family Max/Pro)
//   'd' → Apple9+ Ultra
//   'p' → iPhone (mobile A-series)
//   'g' → base/pro
char getArchSuffix(MetalStream* stream) {
  char suffix = 'g';
  if (@available(macOS 13.0, iOS 16.0, *)) {
    auto* device = stream->device();
    if (device) {
      id arch = [device architecture];
      if (arch) {
        NSString* name = [arch name];
        if (name && [name length] > 0) {
          suffix = [name characterAtIndex:[name length] - 1];
        }
      }
    }
  }
  return suffix;
}

// NOTE: `isNaxAvailable(MetalStream*)` is now defined in
// `ops/MatMulCommon.h` (shared with the matmul + affine-quantized-linear
// dispatch paths). It is reachable here because SDPAOp.mm transitively
// pulls in MatMulCommon via SdpaMlxJit.h's includes.
}  // namespace

//===----------------------------------------------------------------------===//
// computeOutputShape — out shape == [B, Hq, qL, V_dim].
// Note: MLX's SDPA returns ONE tensor (the attention output). The PyTorch
// MPS-backend op `_scaled_dot_product_attention_math_for_mps` returns two
// (output + attention weights), but the high-level
// `aten::scaled_dot_product_attention.default` we're registering returns
// one. The AOTI shim layer adapts between the two ABIs.
//===----------------------------------------------------------------------===//

std::vector<SizesType> SDPAOp::computeOutputShape(
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs) const {
  if (inputs.size() < 3 || !inputs[0]->isTensor() || !inputs[2]->isTensor()) {
    return {};
  }
  const auto& q = inputs[0]->toTensor();
  const auto& v = inputs[2]->toTensor();
  return {static_cast<SizesType>(q.size(0)),
          static_cast<SizesType>(q.size(1)),
          static_cast<SizesType>(q.size(2)),
          static_cast<SizesType>(v.size(3))};
}

//===----------------------------------------------------------------------===//
// dispatch — parse 7 inputs (3 tensors + 1 optional mask + 3 scalars),
// route to the right MLX kernel family.
//===----------------------------------------------------------------------===//

void SDPAOp::dispatch(
    MetalStream* stream,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) {
  if (inputs.size() < 3) {
    ET_LOG(Error, "SDPAOp: expected at least 3 tensor inputs (q, k, v)");
    return;
  }
  if (outputs.size() < 1) {
    ET_LOG(Error, "SDPAOp: expected at least 1 output tensor");
    return;
  }

  auto& Q = inputs[0]->toTensor();
  auto& K = inputs[1]->toTensor();
  auto& V = inputs[2]->toTensor();
  auto& O = outputs[0]->toTensor();

  // ---- Validate shapes & dtype ----
  if (Q.dim() != 4 || K.dim() != 4 || V.dim() != 4 || O.dim() != 4) {
    ET_LOG(Error,
           "SDPAOp: all tensors must be 4-D [B, H, L, D]; got "
           "Q=%dD K=%dD V=%dD O=%dD",
           int(Q.dim()), int(K.dim()), int(V.dim()), int(O.dim()));
    return;
  }
  const auto dtype = Q.scalar_type();
  if (!supports(dtype)) {
    ET_LOG(Error, "SDPAOp: unsupported dtype %d (only fp32/fp16/bf16)", int(dtype));
    return;
  }
  if (K.scalar_type() != dtype || V.scalar_type() != dtype ||
      O.scalar_type() != dtype) {
    ET_LOG(Error, "SDPAOp: Q/K/V/O dtype mismatch");
    return;
  }
  if (Q.strides()[3] != 1 || K.strides()[3] != 1 || V.strides()[3] != 1) {
    ET_LOG(Error,
           "SDPAOp: innermost (D) dim must be contiguous; got strides "
           "Q=%d K=%d V=%d",
           int(Q.strides()[3]), int(K.strides()[3]), int(V.strides()[3]));
    return;
  }

  const int B = static_cast<int>(Q.sizes()[0]);
  const int Hq = static_cast<int>(Q.sizes()[1]);
  const int Hkv = static_cast<int>(K.sizes()[1]);
  const int qL = static_cast<int>(Q.sizes()[2]);
  const int kL = static_cast<int>(K.sizes()[2]);
  const int D = static_cast<int>(Q.sizes()[3]);
  const int Vdim = static_cast<int>(V.sizes()[3]);

  if (Hkv == 0 || Hq % Hkv != 0) {
    ET_LOG(Error, "SDPAOp: bad GQA shape Hq=%d Hkv=%d", Hq, Hkv);
    return;
  }
  const int gqa_factor = Hq / Hkv;

  // ---- Parse optional inputs ----
  // inputs[3] = attn_mask (optional Tensor)
  // inputs[4] = dropout_p (double)
  // inputs[5] = is_causal (bool)
  // inputs[6] = scale (optional double)
  const executorch::aten::Tensor* mask = nullptr;
  if (inputs.size() >= 4 && inputs[3] && inputs[3]->isTensor()) {
    mask = &inputs[3]->toTensor();
  }
  double dropout_p = 0.0;
  if (inputs.size() >= 5 && inputs[4]) {
    auto& s = *inputs[4];
    if (s.isDouble())   dropout_p = s.toDouble();
    else if (s.isInt()) dropout_p = static_cast<double>(s.toInt());
  }
  if (dropout_p != 0.0) {
    ET_LOG(Error, "SDPAOp: dropout_p must be 0 (eval-only); got %f", dropout_p);
    return;
  }
  bool is_causal = false;
  if (inputs.size() >= 6 && inputs[5]) {
    auto& s = *inputs[5];
    if (s.isBool())     is_causal = s.toBool();
    else if (s.isInt()) is_causal = (s.toInt() != 0);
  }
  // scale: PyTorch default is 1/sqrt(D).
  float scale = 1.0f / std::sqrt(static_cast<float>(D));
  if (inputs.size() >= 7 && inputs[6]) {
    auto& s = *inputs[6];
    if (s.isDouble())   scale = static_cast<float>(s.toDouble());
    else if (s.isInt()) scale = static_cast<float>(s.toInt());
  }

  // ---- Routing decision tree (mirrors MLX
  // scaled_dot_product_attention.cpp:643-786 + use_fallback at 588-637) ----
  // Core split: qL ≤ 8 → vector path; else steel path.
  // Vector head-dim allow-list: {64, 96, 128, 256}.
  // Steel head-dim allow-list:  {64, 80, 128}.
  const bool sdpa_vector_supported_head_dim =
      (D == Vdim) &&
      (D == 64 || D == 96 || D == 128 || D == 256);
  const bool sdpa_full_supported_head_dim =
      (D == Vdim) &&
      (D == 64 || D == 80 || D == 128);

  // do_causal: only meaningful when qL > 1 (single-token decode never
  // masks — there's nothing to mask out). MLX upstream applies the same
  // rule (scaled_dot_product_attention.cpp:743).
  const bool do_causal = is_causal && (qL > 1);

  if (qL <= 8) {
    if (!sdpa_vector_supported_head_dim) {
      ET_CHECK_MSG(false,
          "SDPAOp: vector path requires D == V == {64|96|128|256}; got D=%d V=%d",
          D, Vdim);
      return;
    }
    if (qL > kL) {
      ET_CHECK_MSG(false,
          "SDPAOp: vector path requires qL <= kL; got qL=%d kL=%d", qL, kL);
      return;
    }
    if (qL * gqa_factor > 32) {
      ET_CHECK_MSG(false,
          "SDPAOp: vector path requires qL*gqa <= 32; got qL=%d gqa=%d",
          qL, gqa_factor);
      return;
    }

    // Choose single-pass vs 2-pass per MLX heuristic
    // (scaled_dot_product_attention.cpp:740-750):
    //   2-pass when (devc in {'d','s'} && kL >= 1024)
    //          OR   (Hkv < Hq && kL >= 4096)  [GQA + long kL]
    const char devc = getArchSuffix(stream);
    const bool use_2pass =
        ((devc == 'd' || devc == 's') && kL >= 1024) ||
        (Hkv < Hq && kL >= 4096);

    if (use_2pass) {
      sdpa_mlx_jit::dispatchSdpaVector2PassViaMlxJit(
          stream, Q, K, V, mask, O, scale, do_causal, dtype, devc);
    } else {
      sdpa_mlx_jit::dispatchSdpaVectorViaMlxJit(
          stream, Q, K, V, mask, O, scale, do_causal, dtype);
    }
  } else {
    // Steel path (qL > 8, prefill).
    if (!sdpa_full_supported_head_dim) {
      ET_CHECK_MSG(false,
          "SDPAOp: steel path requires D == V == {64|80|128}; got D=%d V=%d",
          D, Vdim);
      return;
    }

    // NAX vs non-NAX selection (scaled_dot_product_attention.cpp:177-179):
    //   nax if is_nax_available() && D != 80 && (dtype != fp32 || tf32_enabled())
    // CAVEAT: MLX's `env::enable_tf32()` (and our compile-time
    // `EXECUTORCH_METAL_TF32_DEFAULT_ENABLED`) is purely a ROUTING flag
    // — it decides whether fp32 dispatches to the NAX kernel. It does
    // NOT activate any hardware-level TF32 mode (Metal has no such knob;
    // the NAX kernel is the same code path used for bf16/fp16). So with
    // TF32 ON, fp32 SDPA dispatches to the NAX kernel directly.
    // Empirically, the steel_attention_nax kernel produces incorrect
    // output for fp32 inputs on D=128+causal shapes (sdpa_steel_causal
    // test sees max_diff=8.0 vs 0.0 for the SIMD-MMA path). bf16 NAX
    // works correctly for the same shape. Matmul-fp32-NAX is also
    // correct. The bug appears specific to steel_attention_nax with
    // T=float; root cause not yet investigated (could be MLX 0.31.2
    // kernel issue or Apple9 NAX behavior with fp32 cooperative-tensor
    // matmul). Until investigated, gate fp32 SDPA off the NAX path —
    // bf16/fp16 SDPA still use NAX. fp32 matmul (separate dispatch) is
    // unaffected and continues to use NAX with TF32.
    const bool useNax = MetalDeviceInfo::isNaxAvailable() && (D != 80) &&
        (dtype != executorch::aten::ScalarType::Float);

    sdpa_mlx_jit::dispatchSteelAttentionViaMlxJit(
        stream, Q, K, V, mask, O, scale, do_causal, dtype, useNax);
  }
}

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
