//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#pragma once

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#if !EXIR_MPS_DELEGATE
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#else
#include <executorch/backends/apple/mps/runtime/MPSDevice.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#endif
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace mps {

#if EXIR_MPS_DELEGATE
using torch::executor::mps::delegate::MPSDevice;
#endif

#if EXIR_MPS_DELEGATE
using namespace exec_aten;
#else
using namespace torch;
#endif

MPSDataType getMPSDataType(ScalarType scalar_type);
MPSDataType getMPSScalarType(ScalarType scalar_type);
ScalarType getScalarType(MPSDataType mpsDataType);
MPSGraphTensor*
castMPSTensor(MPSGraph* mpsGraph, MPSGraphTensor* tensor, ScalarType toType);
MPSGraphTensor*
castMPSTensor(MPSGraph* mpsGraph, MPSGraphTensor* tensor, MPSDataType toType);

// The MPSShape could vary based on memory format
MPSShape* getMPSShape(
    const Tensor& t,
    MemoryFormat memory_format = MemoryFormat::Contiguous);
MPSShape* getMPSShape(
    const IntArrayRef& sizes,
    MemoryFormat memory_format = MemoryFormat::Contiguous);
std::vector<int64_t> getMPSShapeVec(const MPSShape* shape);

static inline id<MTLBuffer> getMTLBufferStorage(const Tensor& tensor) {
#if EXIR_MPS_DELEGATE
#if TARGET_OS_SIMULATOR
  // Simulator crashes in newBufferWithBytesNoCopy, so we're making a copy of
  // the data.
  uint8_t* data = tensor.mutable_data_ptr<uint8_t>();
  return [MPSDevice::getInstance()->device() newBufferWithBytes:data
                                                         length:tensor.nbytes()
                                                        options:0];
#else
  uint8_t* data = tensor.mutable_data_ptr<uint8_t>();
  return [MPSDevice::getInstance()->device()
      newBufferWithBytesNoCopy:data
                        length:tensor.nbytes()
                       options:0
                   deallocator:nil];
#endif // TARGET_OS_SIMULATOR
#else
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
#endif // EXIR_MPS_DELEGATE
}

class Placeholder {
 public:
  Placeholder()
      : _placeholder(nullptr), _value(nullptr), _tensor(Tensor(nullptr)) {}
  Placeholder(MPSGraphTensor* mpsGraphTensor)
      : _placeholder(mpsGraphTensor),
        _value(nullptr),
        _tensor(Tensor(nullptr)) {}
  Placeholder(
      MPSGraphTensor* mpsGraphTensor,
      const Tensor& self,
      MPSShape* mpsShape = nullptr,
      MPSDataType dataType = MPSDataTypeInvalid);
  MPSGraphTensor* getMPSGraphTensor() {
    return _placeholder;
  }
  MPSGraphTensorData* getMPSGraphTensorData() {
    return _value;
  }
  bool isIntermediate() {
    return _value == nullptr;
  }

 private:
  MPSGraphTensor* _placeholder;
  MPSGraphTensorData* _value;
  Tensor _tensor;
};

} // namespace mps
