//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

namespace mps {
using namespace torch;

PyMPSGraphTensor*
MPSGraphModule::clamp(MPSGraphTensor* inputTensor, float min, float max, bool use_min, bool use_max) {
  if(use_min && use_max) {
    MPSGraphTensor* minTensor = [mpsGraph constantWithScalar:min
                                                       shape:inputTensor.shape
                                                    dataType:inputTensor.dataType];
    MPSGraphTensor* maxTensor = [mpsGraph constantWithScalar:max
                                                       shape:inputTensor.shape
                                                    dataType:inputTensor.dataType];
    return [mpsGraph clampWithTensor:inputTensor
                      minValueTensor:minTensor
                      maxValueTensor:maxTensor
                                name:@"clamp"];
  } else if(use_min && !use_max) {
    MPSGraphTensor* minTensor = [mpsGraph constantWithScalar:min
                                                       shape:inputTensor.shape
                                                    dataType:inputTensor.dataType];
    return [mpsGraph maximumWithPrimaryTensor:inputTensor
                                                secondaryTensor:minTensor
                                                          name:nil];
  } else if(!use_min && use_max) {
    MPSGraphTensor* maxTensor = [mpsGraph constantWithScalar:max
                                                    shape:inputTensor.shape
                                                dataType:inputTensor.dataType];
    return [mpsGraph minimumWithPrimaryTensor:inputTensor
                                                    secondaryTensor:maxTensor
                                                               name:nil];
  }

  //For the case that neither min nor max is given? Nothing in the documentation forbids this.
  return inputTensor;
}

PyMPSGraphTensor* MPSGraphModule::where(MPSGraphTensor* condition,
    MPSGraphTensor* input, MPSGraphTensor* other) {
  if ([condition dataType] != MPSDataTypeBool) {
    condition = [mpsGraph castTensor:condition
                              toType:MPSDataTypeBool
                              name:@"condition"];
  }
  MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:condition
                                                     truePredicateTensor:input
                                                    falsePredicateTensor:other
                                                                    name:nil];
  return outputTensor;
}

}//namespace mps
