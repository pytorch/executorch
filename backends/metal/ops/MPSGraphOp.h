/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/metal/core/MpsInterop.h>  // for ET_METAL_USE_MPSGRAPH

#if ET_METAL_USE_MPSGRAPH

#include <executorch/backends/metal/ops/registry/MetalOp.h>
#include <executorch/backends/metal/core/MetalStream.h>

#import <Metal/Metal.h>
#import <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// MPSGraphOp - Base class for ops that delegate compute to MPSGraph.
// Subclass contract:
//   - Override `buildGraph()` to construct an MPSGraph for the given input
//     shapes/dtypes. Return the graph + its input/output placeholder tensors.
//   - Override `cacheKey()` if the default (shape+dtype-based) hashing is
//     insufficient (e.g. when constants or attributes affect the graph).
// Base-class behavior (in dispatch()):
//   1. Build a cache key from inputs/outputs.
//   2. On miss: call buildGraph(), cache the entry.
//   3. End any active compute encoder on the stream.
//   4. Wrap our MTLCommandBuffer as MPSCommandBuffer.
//   5. Wrap each MTLBuffer as MPSGraphTensorData.
//   6. encodeToCommandBuffer.
//   7. Subsequent ops open a new encoder on the same cmd buffer.
// Constraints:
//   - Under MTL4: uses a dedicated singleton legacy queue + MTLSharedEvent
//     for cross-queue sync to MetalStream's MTL4 cb. (MPSGraph wraps
//     id<MTLCommandBuffer> legacy; not directly interoperable with the
//     MTL4 cb, so we go via a legacy cb + event.)
//===----------------------------------------------------------------------===//

class MPSGraphOp : public MetalOp {
 public:
  MPSGraphOp() = default;
  ~MPSGraphOp() override;

  // Final dispatch: cache lookup + build-on-miss + encode.
  // Subclasses extend behavior via cacheKey() / buildGraph().
  void dispatch(
      MetalStream* stream,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) override final;

  // MPSGraph ops don't ship MSL kernel source.
  const char* kernelSource() const override {
    return "";
  }

 protected:
  struct CachedGraph {
    MPSGraph* graph = nil; // strong (ARC manages — all consumers ARC)
    std::vector<MPSGraphTensor*> inputPlaceholders; // ARC-managed
    std::vector<MPSGraphTensor*> outputPlaceholders; // ARC-managed
  };

  // Build the MPSGraph for the concrete input shapes. Called once per cache
  // miss. Subclass returns the placeholders (in input order) + outputs.
  virtual CachedGraph buildGraph(
      ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) = 0;

  // Compute a human-readable cache key. Used only for log messages on cache
  // miss — the actual cache is keyed on a packed binary ShapeKey (below) for
  // perf. Subclasses can override to add op-specific signature info.
  virtual std::string cacheKey(::executorch::runtime::Span<::executorch::runtime::EValue*> inputs, ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs);

 private:
  // Binary cache key: packs (dtype, dim, sizes...) for each input then
  // each output into a small stack-resident array. Comparable + hashable
  // without any heap allocation, so the hot dispatch path is alloc-free.
  struct ShapeKey {
    static constexpr size_t kMaxPacked = 64;
    std::array<uint64_t, kMaxPacked> data{};
    size_t len = 0;
    bool operator==(const ShapeKey& o) const {
      return len == o.len &&
          std::memcmp(data.data(), o.data.data(), len * sizeof(uint64_t)) == 0;
    }
  };
  struct ShapeKeyHash {
    size_t operator()(const ShapeKey& k) const noexcept {
      // FNV-1a 64-bit
      uint64_t h = 0xcbf29ce484222325ULL;
      for (size_t i = 0; i < k.len; ++i) {
        h ^= k.data[i];
        h *= 0x100000001b3ULL;
      }
      return static_cast<size_t>(h);
    }
  };
  static ShapeKey packShapes(::executorch::runtime::Span<::executorch::runtime::EValue*> inputs, ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs);

  std::unordered_map<ShapeKey, CachedGraph, ShapeKeyHash> cache_;
  // Memo of the last looked-up entry. Most inference loops call dispatch
  // repeatedly with the same shapes, so this avoids even the hash + map
  // lookup on the hot path.
  ShapeKey last_key_;
  CachedGraph* last_entry_ = nullptr;
};

//===----------------------------------------------------------------------===//
// MPSGraphMatMulOp - aten::mm via
// MPSGraph.matrixMultiplicationWithPrimaryTensor
// Inputs:  A [M, K], B [K, N]
// Output:  C [M, N] = A @ B
// Useful for cases where MPSGraph's tile/algorithm selection beats our hand
// kernel (large matmul, mixed dtypes, etc). Per-shape graph cache keeps
// per-call overhead at the encode step (~50-100µs) after warm-up.
//===----------------------------------------------------------------------===//

class MPSGraphMatMulOp : public MPSGraphOp {
 public:
  const char* name() const override {
    return "aten::mm";
  }
  bool supports(ScalarType dtype) const override {
    return dtype == ScalarType::Float || dtype == ScalarType::Half ||
        dtype == ScalarType::BFloat16;
  }
  // Output shape for mm(A[M,K], B[K,N]) is [M, N]. Required so MPSGraphOp's
  // resizeOutput sets the right shape on the output tensor — without this,
  // the base resizeOutput falls back to inputs[0]'s shape ([M, K]) and the
  // last (N - K) elements of the output are never written.
  std::vector<runtime::etensor::Tensor::SizesType> computeOutputShape(
      ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs) const override {
    if (inputs.size() < 2 || !inputs[0]->isTensor() || !inputs[1]->isTensor()) {
      return {};
    }
    auto& A = inputs[0]->toTensor();
    auto& B = inputs[1]->toTensor();
    if (A.dim() < 2 || B.dim() < 2)
      return {};
    using SizesType = runtime::etensor::Tensor::SizesType;
    return {
        static_cast<SizesType>(A.size(A.dim() - 2)),
        static_cast<SizesType>(B.size(B.dim() - 1))};
  }

 protected:
  CachedGraph buildGraph(::executorch::runtime::Span<::executorch::runtime::EValue*> inputs, ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) override;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch

#endif // ET_METAL_USE_MPSGRAPH
