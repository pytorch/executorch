
//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

Error
MPSGraphBuilder::mpsBatchNormOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSBatchNorm();
  ET_LOG(
    Debug, "%s: (%d, %d, %d, %d, %d) -> (%d, %d, %d)",
    __FUNCTION__,
    graphNode->input_id(),
    graphNode->mean_id(),
    graphNode->var_id(),
    graphNode->weight_id(),
    graphNode->bias_id(),
    graphNode->output1_id(),
    graphNode->output2_id(),
    graphNode->output3_id()
  );

  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input_id());
  MPSGraphTensor* meanTensor = getMPSGraphTensor(graphNode->mean_id());
  MPSGraphTensor* varTensor = getMPSGraphTensor(graphNode->var_id());
  MPSGraphTensor* weightTensor = getMPSGraphTensor(graphNode->weight_id());
  MPSGraphTensor* biasTensor = getMPSGraphTensor(graphNode->bias_id());
  float epsilon = graphNode->epsilon();

  // Shapes are NCHW so the input parameters to normalization are 1xCx1x1
  NSMutableArray<NSNumber*>* newShape = [NSMutableArray array];
  [newShape addObject:[NSNumber numberWithInt:1]];
  [newShape addObject:inputTensor.shape[1]];
  for(int i = 2; i<[inputTensor.shape count]; ++i) {
    [newShape addObject:[NSNumber numberWithInt:1]];
  }
  // No need for momentum since we are not training for now
  // TODO: Check if momentum is needed
  MPSGraphTensor* reshapedMeanTensor = [_mpsGraph reshapeTensor:meanTensor
                             withShape:newShape
                                  name:nil];
  MPSGraphTensor* reshapedVarTensor = [_mpsGraph reshapeTensor:varTensor
                             withShape:newShape
                                  name:nil];
  MPSGraphTensor* reshapedWeightTensor = [_mpsGraph reshapeTensor:weightTensor
                               withShape:newShape
                                    name:nil];
  MPSGraphTensor* reshapedBiasTensor = [_mpsGraph reshapeTensor:biasTensor
                               withShape:newShape
                                    name:nil];

  _idToMPSGraphTensor[graphNode->output1_id()] = [_mpsGraph normalizationWithTensor:inputTensor
                                meanTensor:reshapedMeanTensor
                            varianceTensor:reshapedVarTensor
                               gammaTensor:reshapedWeightTensor
                                betaTensor:reshapedBiasTensor
                                   epsilon:epsilon
                                      name:@"batch_norm"];

  //For now just return meanTensor and varTensor assuming this isn't training

  // saveVarTensor
  _idToMPSGraphTensor[graphNode->output2_id()] = [_mpsGraph identityWithTensor:varTensor name:nil];
  // saveMeanTensor
  _idToMPSGraphTensor[graphNode->output2_id()] =  [_mpsGraph identityWithTensor:meanTensor name:nil];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsLayerNormOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSLayerNorm();
  ET_LOG(
    Debug, "%s: (%d, %d, %d) -> (%d, %d, %d)",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->weight_id(),
    graphNode->bias_id(),
    graphNode->output1_id(),
    graphNode->output2_id(),
    graphNode->output3_id()
  );

  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());
  MPSGraphTensor* weightTensor = getMPSGraphTensor(graphNode->weight_id());
  MPSGraphTensor* biasTensor = getMPSGraphTensor(graphNode->bias_id());
  const int input_ndim = [inputTensor.shape count];
  const int normalized_shape_ndim = graphNode->normalized_shape()->size();
  const int ndim_to_normalize = input_ndim - normalized_shape_ndim;

  NSMutableArray<NSNumber*>* axesArray = [NSMutableArray arrayWithCapacity:normalized_shape_ndim];
  for (int32_t idx = ndim_to_normalize; idx < input_ndim; idx++)  {
    [axesArray addObject:[NSNumber numberWithInt:idx]];
  }

  MPSGraphTensor* meanTensor = [_mpsGraph meanOfTensor:inputTensor
                                                 axes:axesArray
                                                 name:@"LayerNorm/MeanTensor"];

  MPSGraphTensor* varianceTensor = [_mpsGraph varianceOfTensor:inputTensor
                                                    meanTensor:meanTensor
                                                          axes:axesArray
                                                          name:@"LayerNorm/varianceTensor"];
  MPSGraphTensor* normalizedTensor = [_mpsGraph normalizationWithTensor:inputTensor
                                                              meanTensor:meanTensor
                                                              varianceTensor:varianceTensor
                                                              gammaTensor:weightTensor
                                                              betaTensor:biasTensor
                                                              epsilon:graphNode->eps()
                                                              name:@"LayerNorm/resultTensor"];

  _idToMPSGraphTensor[graphNode->output1_id()] = normalizedTensor;
  _idToMPSGraphTensor[graphNode->output2_id()] = meanTensor;
  _idToMPSGraphTensor[graphNode->output3_id()] = varianceTensor;

  return Error::Ok;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
