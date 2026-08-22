//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#pragma once

#import <Foundation/Foundation.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <executorch/backends/apple/mps/runtime/MPSDevice.h>
#include <executorch/backends/apple/mps/schema_generated.h>

namespace executorch {
namespace backends {
namespace mps {
namespace delegate {

#define INF std::numeric_limits<float>::infinity()

MPSDataType getMPSScalarType(executorch::aten::ScalarType scalar_type);
executorch::aten::ScalarType getScalarType(MPSDataType mpsDataType);
MPSGraphTensor *castMPSTensor(MPSGraph *mpsGraph, MPSGraphTensor *tensor, executorch::aten::ScalarType toType);
MPSGraphTensor *castMPSTensor(MPSGraph *mpsGraph, MPSGraphTensor *tensor, MPSDataType toType);
std::vector<int64_t> getMPSShapeVec(const MPSShape *shape);

template <typename T = size_t> std::vector<T> flatbufferDimsToVector(const flatbuffers::Vector<int32_t> *dims) {
  std::vector<T> dimsData;
  dimsData.reserve(dims->size());
  for (auto dim : *dims) {
    dimsData.push_back(static_cast<T>(dim));
  }
  return dimsData;
}

id<MTLBuffer> getMTLBufferStorage(const executorch::aten::Tensor &tensor);
void *pageAlignedBlockPtr(const void *ptr, NSUInteger size, NSUInteger *alignedBlockSize);

MPSGraphTensor *permuteTensor(MPSGraph *graph, MPSGraphTensor *inputTensor, NSArray *permuteOrder);

} // namespace delegate
} // namespace mps
} // namespace backends
} // namespace executorch
