//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

namespace mps {
using namespace torch;

std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*, PyMPSGraphTensor*>
MPSGraphModule::batchNorm(MPSGraphTensor* inputTensor,
                           MPSGraphTensor* meanTensor,
                           MPSGraphTensor* varTensor,
                           MPSGraphTensor* weightTensor,
                           MPSGraphTensor* biasTensor,
                           float momentum,
                           float epsilon) {


  //Shapes are NCHW so the input parameters to normalization are 1xCx1x1
  NSMutableArray<NSNumber*>* newShape = [NSMutableArray array];
  [newShape addObject:[NSNumber numberWithInt:1]];
  [newShape addObject:inputTensor.shape[1]];
  for(int i = 2; i<[inputTensor.shape count]; ++i) {
    [newShape addObject:[NSNumber numberWithInt:1]];
  }
  //No need for momentum since we are not training for now since not training?
  MPSGraphTensor* reshapedMeanTensor = [mpsGraph reshapeTensor:meanTensor
                             withShape:newShape
                                  name:nil];
  MPSGraphTensor* reshapedVarTensor = [mpsGraph reshapeTensor:varTensor
                             withShape:newShape
                                  name:nil];
  MPSGraphTensor* reshapedWeightTensor = [mpsGraph reshapeTensor:weightTensor
                               withShape:newShape
                                    name:nil];
  MPSGraphTensor* reshapedBiasTensor = [mpsGraph reshapeTensor:biasTensor
                               withShape:newShape
                                    name:nil];

  MPSGraphTensor* result = [mpsGraph normalizationWithTensor:inputTensor
                                meanTensor:reshapedMeanTensor
                            varianceTensor:reshapedVarTensor
                               gammaTensor:reshapedWeightTensor
                                betaTensor:reshapedBiasTensor
                                   epsilon:epsilon
                                      name:@"batch_norm"];
  MPSGraphTensor* saveVarTensor = [mpsGraph identityWithTensor:varTensor name:nil];
  MPSGraphTensor* saveMeanTensor = [mpsGraph identityWithTensor:meanTensor name:nil];

  //For now just return meanTensor and varTensor assuming this isn't training
  auto out_tuple =  std::make_tuple<PyMPSGraphTensor*>(result, saveMeanTensor, saveVarTensor);
  return out_tuple;
}

//Normalizes over the last ndim=normalized_shape.size() dimensions scaling
//with weight and bias tensors if they are non-nil
std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*, PyMPSGraphTensor*>
MPSGraphModule::layerNorm(MPSGraphTensor* inputTensor,
                          IntArrayRef normalized_shape,
                          MPSGraphTensor* weightTensor,
                          MPSGraphTensor* biasTensor,
                          float eps) {

  const int input_ndim = [inputTensor.shape count];
  const int normalized_shape_ndim = normalized_shape.size();
  const int ndim_to_normalize = input_ndim-normalized_shape_ndim;

  NSMutableArray<NSNumber*>* axesArray = [NSMutableArray arrayWithCapacity:normalized_shape_ndim];
  for (const auto idx : c10::irange(ndim_to_normalize, input_ndim)) {
    [axesArray addObject:[NSNumber numberWithInt:idx]];
  }

  MPSGraphTensor* meanTensor = [mpsGraph meanOfTensor:inputTensor
                                                 axes:axesArray
                                                 name:@"LayerNorm/MeanTensor"];

  MPSGraphTensor* varianceTensor = [mpsGraph varianceOfTensor:inputTensor
                                                    meanTensor:meanTensor
                                                          axes:axesArray
                                                          name:@"LayerNorm/varianceTensor"];
  MPSGraphTensor* normalizedTensor = [mpsGraph normalizationWithTensor:inputTensor
                                                              meanTensor:meanTensor
                                                              varianceTensor:varianceTensor
                                                              gammaTensor:weightTensor
                                                              betaTensor:biasTensor
                                                              epsilon:eps
                                                              name:@"LayerNorm/resultTensor"];

  return std::make_tuple<PyMPSGraphTensor*>(normalizedTensor, meanTensor, varianceTensor);
}
}//namespace mps
