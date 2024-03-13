
//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>
#include <iostream>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

Error
MPSGraphBuilder::mpsMatMulOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSMatMul();
  ET_LOG(
    Debug, "%s: (%d, %d) -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->input2_id(),
    graphNode->output_id()
  );

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph matrixMultiplicationWithPrimaryTensor:getMPSGraphTensor(graphNode->input1_id())
                                     secondaryTensor:getMPSGraphTensor(graphNode->input2_id())
                                                name:nil];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsAddmmOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSAddmm();
  ET_LOG(
    Debug, "%s: (%d, %d, %d) -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->input2_id(),
    graphNode->input3_id(),
    graphNode->output_id()
  );

  MPSGraphTensor* biasTensor = getMPSGraphTensor(graphNode->input1_id());
  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input2_id());
  MPSGraphTensor* weightTensor = getMPSGraphTensor(graphNode->input3_id());
  float beta = graphNode->beta();
  float alpha = graphNode->alpha();

  MPSGraphTensor* multiplyTensor = [_mpsGraph matrixMultiplicationWithPrimaryTensor:inputTensor
                                                                    secondaryTensor:weightTensor
                                                                               name:@"addmm/matmul"];
  MPSGraphTensor* alphaTimesMultiply = multiplyTensor;
  if (alpha != 1.0) {
    // assert
    MPSGraphTensor* alphaTensor = [_mpsGraph constantWithScalar:alpha
                                                       dataType:inputTensor.dataType];

    alphaTimesMultiply = [_mpsGraph multiplicationWithPrimaryTensor:multiplyTensor
                                                    secondaryTensor:alphaTensor
                                                              name:@"addmm/alpha*matmul"];
  }

  MPSGraphTensor* betaBiasTensor = biasTensor;
  if (beta != 1.0) {
    MPSGraphTensor* betaTensor = [_mpsGraph constantWithScalar:beta
                                                      dataType:inputTensor.dataType];

    betaBiasTensor = [_mpsGraph multiplicationWithPrimaryTensor:biasTensor
                                                  secondaryTensor:betaTensor
                                                  name:@"addmm/beta*bias"];
  }

  _idToMPSGraphTensor[graphNode->output_id()] = [_mpsGraph additionWithPrimaryTensor:alphaTimesMultiply
                                                                    secondaryTensor:betaBiasTensor
                                                                               name:@"addmm/beta*bias*alpha*matmul"];

  return Error::Ok;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
