//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

namespace mps {
using namespace torch;

PyMPSGraphTensor*
MPSGraphModule::arange(Scalar start, Scalar end, Scalar step, MPSDataType dataType, const int numEle) {
    auto shapeTensor = [mpsGraph constantWithData:[NSData dataWithBytes:&numEle length:sizeof(int32_t)]
                                            shape:@[ @1 ]
                                         dataType:MPSDataTypeInt32];
    auto startScalar = start.isFloatingPoint() ? start.to<float>() : start.to<int>();
    auto stepScalar = step.isFloatingPoint() ? step.to<float>() : step.to<int>();
    auto coordsTensor = [mpsGraph coordinateAlongAxis:0 withShapeTensor:shapeTensor name:nil];
    coordsTensor = [mpsGraph castTensor:coordsTensor toType:dataType name:@"coords"];

    auto startTensor = [mpsGraph constantWithScalar:startScalar
                             dataType:dataType];
    auto multiplyTensor = [mpsGraph constantWithScalar:stepScalar
                             dataType:dataType];
    auto scaledCoords = [mpsGraph multiplicationWithPrimaryTensor:coordsTensor
                                                secondaryTensor:multiplyTensor
                                                            name:nil];
    auto outputTensor = [mpsGraph additionWithPrimaryTensor:scaledCoords secondaryTensor:startTensor name:nil];
    return outputTensor;
}
} // namespace mps
