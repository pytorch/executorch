//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

namespace mps {
using namespace torch;

PyMPSGraphTensor*
MPSGraphModule::constantWithScalar(MPSDataType dataType, const IntArrayRef& sizes, double scalar) {
  TORCH_CHECK(!sizes.empty(), "No sizes passed to create a constant with scalar");
  if (sizes.back() == 0) {
    // Cannot create a zero-sized dimension through mpsGraph
    return nil;
  }
  return [mpsGraph constantWithScalar:scalar
                                shape:getMPSShape(sizes)
                             dataType:dataType];
}

PyMPSGraphTensor*
MPSGraphModule::full(IntArrayRef size, double scalar, MPSDataType dataType) {
  if (size.back() == 0) {
    // Cannot create a zero-sized dimension through mpsGraph
    return nil;
  }
  return [mpsGraph constantWithScalar:scalar
                                shape:getMPSShape(size)
                             dataType:dataType];
}

PyMPSGraphTensor*
MPSGraphModule::full_like(MPSGraphTensor* inputTensor, double scalar) {
  return [mpsGraph constantWithScalar:scalar
                                shape:inputTensor.shape
                             dataType:inputTensor.dataType];
}

PyMPSGraphTensor*
MPSGraphModule::constant(double scalar, MPSDataType dataType) {
  return [mpsGraph constantWithScalar:scalar
                             dataType:dataType];
}

PyMPSGraphTensor*
MPSGraphModule::constantTensor(Tensor constant_tensor, MPSDataType dataType) {
  NSData* dataBuffer = [[NSData alloc] initWithBytes:constant_tensor.data_ptr()
                            length:constant_tensor.nbytes()];
  return [mpsGraph constantWithData:dataBuffer
                              shape:getMPSShape(constant_tensor)
                           dataType:dataType];
}
}//namespace mps
