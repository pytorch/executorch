//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

namespace mps {
using namespace torch;

PyMPSGraphTensor*
MPSGraphModule::relu(MPSGraphTensor* inputTensor) {
  return [mpsGraph reLUWithTensor:inputTensor
                             name:@"relu"];

}

PyMPSGraphTensor*
MPSGraphModule::leaky_relu(MPSGraphTensor* inputTensor, float negative_slope) {
  return [mpsGraph leakyReLUWithTensor:inputTensor
                                 alpha:negative_slope
                                  name:@"leaky_relu"];

}

MPSGraphTensor* tanh(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  auto dataType = [inputTensor dataType];
  constexpr float kBeta =  M_SQRT2 * M_2_SQRTPI * 0.5;
  constexpr float kKappa = 0.044715f;
  MPSGraphTensor *betaf = [mpsGraph constantWithScalar: kBeta
                                                 shape: @[@1]
                                              dataType: dataType];
  MPSGraphTensor *kappaf = [mpsGraph constantWithScalar: kKappa
                                                  shape: @[@1]
                                               dataType: dataType];
  MPSGraphTensor *onef = [mpsGraph constantWithScalar: 1.0f
                                                shape: @[@1]
                                            dataType: dataType];
  MPSGraphTensor *halff = [mpsGraph constantWithScalar: 0.5f
                                                  shape: @[@1]
                                              dataType: dataType];
  MPSGraphTensor *erfTensor = [mpsGraph multiplicationWithPrimaryTensor: inputTensor
                                                        secondaryTensor: inputTensor
                                                                  name : nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor: erfTensor
                                        secondaryTensor: inputTensor
                                                  name : nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor: erfTensor
                                        secondaryTensor: kappaf
                                                  name : nil];
  erfTensor = [mpsGraph additionWithPrimaryTensor: erfTensor
                                  secondaryTensor: inputTensor
                                            name : nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor: erfTensor
                                        secondaryTensor: betaf
                                                  name : nil];
  erfTensor = [mpsGraph tanhWithTensor: erfTensor
                                 name : nil];
  erfTensor = [mpsGraph additionWithPrimaryTensor: erfTensor
                                  secondaryTensor: onef
                                            name : nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor: erfTensor
                                        secondaryTensor: halff
                                                  name : nil];
  return erfTensor;
}

MPSGraphTensor* normcdf(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  auto dataType = [inputTensor dataType];
  const float SQRT1_2 = 0.707106781186547524400844362104849039f;
  MPSGraphTensor *sqrt1_2 = [mpsGraph constantWithScalar: SQRT1_2
                                                      shape: @[@1]
                                                   dataType: dataType];
  MPSGraphTensor *onef = [mpsGraph constantWithScalar: 1.0f
                                                shape: @[@1]
                                            dataType: dataType];
  MPSGraphTensor *halff = [mpsGraph constantWithScalar: 0.5f
                                                  shape: @[@1]
                                              dataType: dataType];

  MPSGraphTensor *erfTensor = [mpsGraph multiplicationWithPrimaryTensor: inputTensor
                                                        secondaryTensor: sqrt1_2
                                                                name : nil];
  erfTensor = [mpsGraph erfWithTensor: erfTensor name : nil];
  erfTensor = [mpsGraph additionWithPrimaryTensor: erfTensor
                                    secondaryTensor: onef
                                                name : nil];
  erfTensor = [mpsGraph multiplicationWithPrimaryTensor: erfTensor
                                      secondaryTensor: halff
                                                  name : nil];

  return  erfTensor;
}

PyMPSGraphTensor*
MPSGraphModule::gelu(MPSGraphTensor* inputTensor,
                      const std::string &approximation=nil) {
  MPSGraphTensor* result;
  if (approximation == "tanh") {
    result = tanh(mpsGraph, inputTensor);
  } else {
    result = normcdf(mpsGraph, inputTensor);
  }
  return [mpsGraph multiplicationWithPrimaryTensor:result
                                    secondaryTensor:inputTensor
                                              name:nil];
}

PyMPSGraphTensor*
MPSGraphModule::softmax(MPSGraphTensor* inputTensor, const int dim, const bool half_to_float) {
  TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on MPS");
  return [mpsGraph softMaxWithTensor:inputTensor
                                axis:dim
                                name:@"softmax"];
}

PyMPSGraphTensor*
MPSGraphModule::log_softmax(MPSGraphTensor* inputTensor, const int dim, const bool half_to_float) {
  TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on MPS");
  MPSGraphTensor* softmaxTensor = [mpsGraph softMaxWithTensor:inputTensor
                                                   axis:dim
                                                   name:@"softmax"];
  return [mpsGraph logarithmWithTensor:softmaxTensor
                                name:@"log_softmax"];
}

PyMPSGraphTensor*
MPSGraphModule::hardTanh(MPSGraphTensor* inputTensor,
                         float min_value,
                         float max_value) {
  MPSDataType inputType = [inputTensor dataType];
  MPSShape* inputShape = [inputTensor shape];
  MPSGraphTensor* minTensor = [mpsGraph constantWithScalar:min_value shape:inputShape dataType:inputType];
  MPSGraphTensor* maxTensor = [mpsGraph constantWithScalar:max_value shape:inputShape dataType:inputType];
  MPSGraphTensor* lessThanMinPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                   secondaryTensor:minTensor
                                                                              name:@"LessThanPredicate"];
  MPSGraphTensor* greaterThanMaxPredicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                                      secondaryTensor:maxTensor
                                                                                 name:@"MoreThanPredicate"];

  MPSGraphTensor* temp = [mpsGraph selectWithPredicateTensor:lessThanMinPredicateTensor
                                              truePredicateTensor:minTensor
                                              falsePredicateTensor:inputTensor
                                              name:@"minOutput"];
  MPSGraphTensor* result = [mpsGraph selectWithPredicateTensor:greaterThanMaxPredicateTensor
                                              truePredicateTensor:maxTensor
                                              falsePredicateTensor:temp
                                              name:@"hardTanh"];
  return result;
}

PyMPSGraphTensor*
MPSGraphModule::glu(MPSGraphTensor* inputTensor, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, inputTensor.shape.count);
  auto splitTensors = [mpsGraph splitTensor:inputTensor
                                  numSplits:2
                                       axis:wrap_dim
                                       name:nil];
  return [mpsGraph multiplicationWithPrimaryTensor:splitTensors[0]
                                   secondaryTensor:[mpsGraph sigmoidWithTensor:splitTensors[1] name:nil]
                                              name:nil];
}

}//namespace mps
