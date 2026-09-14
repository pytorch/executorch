/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "MLXExecutor.h" // Tensor, StreamOrDevice

namespace executorch {
namespace backends {
namespace mlx {

// The K/V window to attend over + how to mask it. `kind` mirrors MLX's mask
// forms: no mask, its fused "causal", or an explicit tensor for anything MLX
// cannot express -- a sliding window, and later tree/speculative patterns.
//
// One attention's worth. A layout holding a private window per sequence builds
// one of these per sequence and attends each in turn.
struct AttendSpec {
  Tensor K;
  Tensor V;
  enum class Mask { None, Causal, Explicit } kind;
  std::optional<Tensor> mask; // Explicit only
};

// Apply `spec` to `q`: match the stored K/V to the query dtype, translate the
// mask kind into what MLX takes, and run SDPA.
inline Tensor
attend(const AttendSpec& spec, const Tensor& q, float scale, StreamOrDevice s) {
  // No-op when equal; the storage precision may differ from the compute dtype.
  Tensor K = spec.K.dtype() == q.dtype()
      ? spec.K
      : ::mlx::core::astype(spec.K, q.dtype(), s);
  Tensor V = spec.V.dtype() == q.dtype()
      ? spec.V
      : ::mlx::core::astype(spec.V, q.dtype(), s);
  // MLX takes the mask as a mode string plus an optional tensor. None and
  // Explicit both map to "" and are told apart only by spec.mask, so an
  // Explicit with no mask would silently attend unmasked.
  std::string mask_mode;
  switch (spec.kind) {
    case AttendSpec::Mask::None:
      break;
    case AttendSpec::Mask::Causal:
      mask_mode = "causal";
      break;
    case AttendSpec::Mask::Explicit:
      if (!spec.mask) {
        throw std::runtime_error(
            "attend: Explicit mask kind with no mask tensor");
      }
      break;
  }
  return ::mlx::core::fast::scaled_dot_product_attention(
      q, K, V, scale, mask_mode, spec.mask, std::nullopt, false, s);
}

// Tensor-typed op face of the off-graph KV cache, kept separate from the
// neutral Cache (which is tensor-free) so a cache can expose both without a
// diamond. ExecutionState holds one; nothing assigns it yet -- the registry
// that owns the cache and hands this pointer to the executor lands in a
// follow-up, until which exec_update_and_attend is unreachable.
class MLXCache {
 public:
  // Named here, not in cache.h: a backend face is tensor-typed and the
  // neutral header cannot know about it.
  static constexpr const char* kFaceName = "mlx.MLXCache";

  virtual ~MLXCache() = default;

  // Write this step's K/V for `layer` at `positions`, one host int per query
  // token, then attend `q` over it. q/k/v are BHSD.
  //
  // A layout holding one window answers with a single SDPA; one holding a
  // private window per sequence answers with several and rejoins them. The
  // layout makes that choice, so it owns the call.
  virtual Tensor attend(
      int layer,
      const std::vector<int32_t>& positions,
      const Tensor& q,
      const Tensor& k,
      const Tensor& v,
      float scale,
      StreamOrDevice s) = 0;

  // The stream for work the cache starts between steps, such as copying cells
  // to fork a sequence. Bound at init, and by the last of several delegates
  // that share the cache.
  virtual void bind_controller_stream(::mlx::core::Stream /*s*/) {}
};

} // namespace mlx
} // namespace backends
} // namespace executorch
