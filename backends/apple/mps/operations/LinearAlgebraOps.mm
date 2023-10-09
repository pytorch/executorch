//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

namespace mps {
using namespace torch;

PyMPSGraphTensor*
MPSGraphModule::addmm(MPSGraphTensor* biasTensor,
                      MPSGraphTensor* inputTensor,
                      MPSGraphTensor* weightTensor,
                      const float beta,
                      const float alpha) {

  MPSGraphTensor* betaTensor = [mpsGraph constantWithScalar:beta
                                              dataType:inputTensor.dataType];
  MPSGraphTensor* alphaTensor = [mpsGraph constantWithScalar:alpha
                                              dataType:inputTensor.dataType];

  if(inputTensor.shape == weightTensor.shape) {
    weightTensor = [mpsGraph transposeTensor:weightTensor
                                      dimension:0
                                      withDimension:1
                                      name:@"addmm/transposedWeightTensor"];
  }

  MPSGraphTensor* multiplyTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:inputTensor
                                                                   secondaryTensor:weightTensor
                                                                   name:@"addmm/matmul"];
  MPSGraphTensor* alphaTimesMultiply = [mpsGraph multiplicationWithPrimaryTensor:multiplyTensor
                                                              secondaryTensor:alphaTensor
                                                              name:@"addmm/alpha*matmul"];
  MPSGraphTensor* betaBiasTensor = biasTensor;
  if(beta!=0.0) {
    betaBiasTensor = [mpsGraph multiplicationWithPrimaryTensor:biasTensor
                                                  secondaryTensor:betaTensor
                                                  name:@"addmm/beta*bias"];
  }
  MPSGraphTensor* outputTensor = [mpsGraph additionWithPrimaryTensor:alphaTimesMultiply
                                                        secondaryTensor:betaBiasTensor
                                                        name:@"addmm/beta*bias*alpha*matmul"];

  return outputTensor;
}
}//namespace
