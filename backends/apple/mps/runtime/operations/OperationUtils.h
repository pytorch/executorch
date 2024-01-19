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

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

#define INF std::numeric_limits<float>::infinity()

MPSDataType getMPSScalarType(exec_aten::ScalarType scalar_type);
exec_aten::ScalarType getScalarType(MPSDataType mpsDataType);
MPSGraphTensor *castMPSTensor(MPSGraph *mpsGraph, MPSGraphTensor *tensor, exec_aten::ScalarType toType);
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

static inline id<MTLBuffer> getMTLBufferStorage(const Tensor &tensor) {
#if TARGET_OS_SIMULATOR
  // Simulator crashes in newBufferWithBytesNoCopy, so we're making a copy of
  // the data.
  uint8_t *data = tensor.mutable_data_ptr<uint8_t>();
  return [MPSDevice::getInstance()->device() newBufferWithBytes:data length:tensor.nbytes() options:0];
#else
  uint8_t *data = tensor.mutable_data_ptr<uint8_t>();
  return [MPSDevice::getInstance()->device() newBufferWithBytesNoCopy:data
                                                               length:tensor.nbytes()
                                                              options:0
                                                          deallocator:nil];
#endif // TARGET_OS_SIMULATOR
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
