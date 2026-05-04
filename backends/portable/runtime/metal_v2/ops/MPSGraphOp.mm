/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MPSGraphOp.h"

// MPSGraphOp.h's gate fully removes the class declaration when
// ET_METAL_USE_MPSGRAPH=0; wrap the .mm body in the same gate so the TU
// compiles to an empty object file in that build mode.
#if ET_METAL_USE_MPSGRAPH

#include <executorch/runtime/platform/log.h>

#include <atomic>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal_v2 {

using executorch::runtime::EValue;
using executorch::runtime::etensor::ScalarType;
using executorch::runtime::etensor::Tensor;

namespace {

// Map ExecuTorch ScalarType to MPSDataType.
MPSDataType toMPSDataType(ScalarType t) {
  switch (t) {
    case ScalarType::Float:    return MPSDataTypeFloat32;
    case ScalarType::Half:     return MPSDataTypeFloat16;
    case ScalarType::BFloat16: return MPSDataTypeBFloat16;
    case ScalarType::Int:      return MPSDataTypeInt32;
    case ScalarType::Long:     return MPSDataTypeInt64;
    case ScalarType::Short:    return MPSDataTypeInt16;
    case ScalarType::Char:     return MPSDataTypeInt8;
    case ScalarType::Byte:     return MPSDataTypeUInt8;
    case ScalarType::Bool:     return MPSDataTypeBool;
    default:
      ET_LOG(Error, "MPSGraphOp: unsupported dtype %d", static_cast<int>(t));
      return MPSDataTypeFloat32;
  }
}

// NSArray<NSNumber*> from a Tensor's sizes.
NSArray<NSNumber*>* nsShape(const Tensor& t) {
  NSMutableArray<NSNumber*>* a =
      [NSMutableArray arrayWithCapacity:t.dim()];
  for (ssize_t i = 0; i < t.dim(); ++i) {
    [a addObject:@(t.size(i))];
  }
  return a;
}

// Wrap an ExecuTorch tensor as MPSGraphTensorData (no copy, no retain on data).
MPSGraphTensorData* makeTensorData(MetalStream* stream, const Tensor& t) {
  // R8.3: use bufferAndOffsetForPtr so subregion tensors (AOTI workspace
  // views) bind at the correct byte offset within the parent MTLBuffer.
  auto bo = stream->bufferAndOffsetForPtr(t.mutable_data_ptr(), t.nbytes());
  if (bo.offset == 0) {
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:bo.mtl
                                                    shape:nsShape(t)
                                                 dataType:toMPSDataType(t.scalar_type())];
  }
  // Non-zero offset path: construct a TensorData rooted at the offset.
  // MPSGraphTensorData has an MTLBuffer+offset descriptor variant on
  // macOS 14+; the simpler portable form is to use rowBytes/init that
  // takes a (buffer, descriptor) pair where the descriptor includes the
  // offset via the buffer's contents+offset wrapping. We use the
  // explicit (buffer, shape, dataType, rowBytes) initializer with no
  // direct offset — fall back to wrapping a sub-region MTLBuffer.
  // (Apple's MPSGraphTensorData has no public offset-aware init that
  // takes shape+dataType; create a sub-allocated alias via
  // newBufferWithBytesNoCopy on the offset host ptr.)
  void* sub_ptr = static_cast<char*>([bo.mtl contents]) + bo.offset;
  id<MTLDevice> dev = stream->device();
  id<MTLBuffer> sub = [dev newBufferWithBytesNoCopy:sub_ptr
                                              length:t.nbytes()
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
  auto* td = [[MPSGraphTensorData alloc] initWithMTLBuffer:sub
                                                     shape:nsShape(t)
                                                  dataType:toMPSDataType(t.scalar_type())];
  [sub release];  // td retains the buffer
  return td;
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// MPSGraphOp base
//===----------------------------------------------------------------------===//

MPSGraphOp::~MPSGraphOp() {
  for (auto& kv : cache_) {
    if (kv.second.graph) [kv.second.graph release];
  }
  cache_.clear();
}

std::string MPSGraphOp::cacheKey(
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) {
  std::ostringstream oss;
  auto append = [&oss](const Tensor& t) {
    oss << static_cast<int>(t.scalar_type()) << ':';
    for (ssize_t i = 0; i < t.dim(); ++i) {
      oss << t.size(i) << (i + 1 < t.dim() ? 'x' : ';');
    }
  };
  for (auto* e : inputs) append(e->toTensor());
  oss << "->";
  for (auto* e : outputs) append(e->toTensor());
  return oss.str();
}

MPSGraphOp::ShapeKey MPSGraphOp::packShapes(
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) {
  ShapeKey k;
  size_t n = 0;
  auto pack = [&](const Tensor& t) {
    // dtype + dim + sizes; bail if we'd overflow (caller falls back to slow
    // path, which is correct but allocates).
    size_t need = 2 + static_cast<size_t>(t.dim());
    if (n + need > ShapeKey::kMaxPacked) {
      n = ShapeKey::kMaxPacked + 1;  // sentinel: overflow
      return;
    }
    k.data[n++] = static_cast<uint64_t>(static_cast<int>(t.scalar_type()));
    k.data[n++] = static_cast<uint64_t>(t.dim());
    for (ssize_t i = 0; i < t.dim(); ++i) {
      k.data[n++] = static_cast<uint64_t>(t.size(i));
    }
  };
  for (auto* e : inputs) pack(e->toTensor());
  // Marker between inputs and outputs so [a,b] / [c] is distinct from [a] / [b,c].
  if (n < ShapeKey::kMaxPacked) k.data[n++] = ~0ULL;
  for (auto* e : outputs) pack(e->toTensor());
  k.len = std::min(n, ShapeKey::kMaxPacked);
  return k;
}

void MPSGraphOp::dispatch(
    MetalStream* stream,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) {
  // Resize output(s) using the standard helper (subclass may also override
  // computeOutputShape if non-trivial).
  for (auto* outE : outputs) {
    auto err = resizeOutput(inputs, outE);
    if (err != runtime::Error::Ok) {
      ET_LOG(Error, "MPSGraphOp(%s): output resize failed", name());
      return;
    }
  }

  // 1. Cache lookup. Two-level: a single-slot memo for the common case
  //    where the same shapes appear across consecutive calls (steady-state
  //    inference), and a hash-keyed map for the slow path.
  ShapeKey key = packShapes(inputs, outputs);
  CachedGraph* g_ptr = nullptr;
  if (last_entry_ && key == last_key_) {
    g_ptr = last_entry_;
  } else {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      auto entry = buildGraph(inputs, outputs);
      if (!entry.graph) {
        ET_LOG(Error, "MPSGraphOp(%s): buildGraph returned null", name());
        return;
      }
      [entry.graph retain];
      it = cache_.emplace(key, std::move(entry)).first;
      ET_LOG(
          Info,
          "MPSGraphOp(%s): built graph for key=%s",
          name(),
          cacheKey(inputs, outputs).c_str());
    }
    last_key_ = key;
    last_entry_ = &it->second;
    g_ptr = &it->second;
  }
  const CachedGraph& g = *g_ptr;

  if (g.inputPlaceholders.size() != inputs.size() ||
      g.outputPlaceholders.size() != outputs.size()) {
    ET_LOG(Error,
           "MPSGraphOp(%s): cached graph arity mismatch (%zu/%zu vs %zu/%zu)",
           name(), g.inputPlaceholders.size(), g.outputPlaceholders.size(),
           inputs.size(), outputs.size());
    return;
  }

  auto* mstream = static_cast<MetalStream*>(stream);
  if (!mstream) {
    ET_LOG(Error, "MPSGraphOp(%s): stream is not MetalStream", name());
    return;
  }

  @autoreleasepool {
    // Wrap inputs/outputs as MPSGraphTensorData (shared by both paths).
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        [NSMutableDictionary dictionaryWithCapacity:inputs.size()];
    for (size_t i = 0; i < inputs.size(); ++i) {
      feeds[g.inputPlaceholders[i]] =
          makeTensorData(mstream, inputs[i]->toTensor());
    }
    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [NSMutableDictionary dictionaryWithCapacity:outputs.size()];
    for (size_t i = 0; i < outputs.size(); ++i) {
      results[g.outputPlaceholders[i]] =
          makeTensorData(mstream, outputs[i]->toTensor());
    }

    // MetalStream handles all the cross-queue sync, encoder close,
    // commit-and-continue adopt-back. We just encode our MPSGraph.
    mstream->encodeWithLegacyCommandBuffer([&](MPSCommandBuffer* mpsCB) {
      [g.graph encodeToCommandBuffer:mpsCB
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:results
                  executionDescriptor:nil];
    });
  }
}

//===----------------------------------------------------------------------===//
// MPSGraphMatMulOp - aten::mm
//===----------------------------------------------------------------------===//

MPSGraphOp::CachedGraph MPSGraphMatMulOp::buildGraph(
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) {
  const Tensor& A = inputs[0]->toTensor();
  const Tensor& B = inputs[1]->toTensor();
  const Tensor& C = outputs[0]->toTensor();

  MPSGraph* graph = [MPSGraph new];
  MPSDataType dt = toMPSDataType(C.scalar_type());

  MPSGraphTensor* aPh = [graph placeholderWithShape:nsShape(A)
                                            dataType:toMPSDataType(A.scalar_type())
                                                name:@"A"];
  MPSGraphTensor* bPh = [graph placeholderWithShape:nsShape(B)
                                            dataType:toMPSDataType(B.scalar_type())
                                                name:@"B"];
  MPSGraphTensor* cTensor =
      [graph matrixMultiplicationWithPrimaryTensor:aPh
                                   secondaryTensor:bPh
                                              name:@"C"];
  // If output dtype differs from inputs (e.g. fp16 inputs producing fp32),
  // cast. Most matmuls keep the dtype.
  if (toMPSDataType(A.scalar_type()) != dt) {
    cTensor = [graph castTensor:cTensor toType:dt name:@"C_cast"];
  }

  return CachedGraph{
      .graph = graph,
      .inputPlaceholders = {aPh, bPh},
      .outputPlaceholders = {cTensor},
  };
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch

#endif // ET_METAL_USE_MPSGRAPH
