
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
MPSGraphBuilder::mpsHardTanhOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSHardTanh();

  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  float minValue = graphNode->min_value();
  float maxValue = graphNode->max_value();
  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());

  MPSDataType inputType = [inputTensor dataType];
  MPSShape* inputShape = [inputTensor shape];
  MPSGraphTensor* minTensor = [_mpsGraph constantWithScalar:minValue shape:inputShape dataType:inputType];
  MPSGraphTensor* maxTensor = [_mpsGraph constantWithScalar:maxValue shape:inputShape dataType:inputType];
  MPSGraphTensor* lessThanMinPredicateTensor = [_mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                   secondaryTensor:minTensor
                                                                              name:@"LessThanPredicate"];
  MPSGraphTensor* greaterThanMaxPredicateTensor = [_mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                                      secondaryTensor:maxTensor
                                                                                 name:@"MoreThanPredicate"];

  MPSGraphTensor* temp = [_mpsGraph selectWithPredicateTensor:lessThanMinPredicateTensor
                                              truePredicateTensor:minTensor
                                              falsePredicateTensor:inputTensor
                                              name:@"minOutput"];

  _idToMPSGraphTensor[graphNode->output_id()] = [_mpsGraph selectWithPredicateTensor:greaterThanMaxPredicateTensor
                                                                truePredicateTensor:maxTensor
                                                                falsePredicateTensor:temp
                                                                                name:@"hardTanh"];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsReLUOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSReLU();

  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph reLUWithTensor:getMPSGraphTensor(graphNode->input1_id())
                          name:@"relu"];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsGELUOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSGELU();
  std::string approximation = graphNode->approximate()->str();
  Error status = Error::Ok;

  ET_LOG(
    Debug, "%s: %d (%s) -> %d",
    __FUNCTION__, graphNode->input1_id(), approximation.c_str(), graphNode->output_id()
  );

  if (approximation == "tanh") {
    status = mpsTanhOp(nodePtr);
  } else {
    status = mpsNormCdfOp(nodePtr);
  }

  ET_CHECK_OR_RETURN_ERROR(
    status == Error::Ok,
    Internal,
    "[ERROR] Couldn't add GELU node to MPSGraph");
    _idToMPSGraphTensor[graphNode->output_id()] =
      [_mpsGraph multiplicationWithPrimaryTensor:_idToMPSGraphTensor[graphNode->output_id()]
                                 secondaryTensor:getMPSGraphTensor(graphNode->input1_id())
                                           name:nil];

  return status;
}

Error
MPSGraphBuilder::mpsLeakyReLUOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSLeakyReLU();

  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph leakyReLUWithTensor:getMPSGraphTensor(graphNode->input1_id())
                             alpha:graphNode->negative_slope()
                              name:@"leaky_relu"];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsSoftmaxOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSSoftmax();

  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  ET_CHECK_MSG(!graphNode->half_to_float(), "softmax with half to float conversion is not supported on MPS");

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph softMaxWithTensor:getMPSGraphTensor(graphNode->input1_id())
                            axis:graphNode->dim()
                            name:@"softmax"];
  return Error::Ok;
}

Error
MPSGraphBuilder::mpsLogSoftmaxOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSLogSoftmax();

  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  ET_CHECK_MSG(!graphNode->half_to_float(), "softmax with half to float conversion is not supported on MPS");

  MPSGraphTensor* softmaxTensor = [_mpsGraph softMaxWithTensor:getMPSGraphTensor(graphNode->input1_id())
                                                         axis:graphNode->dim()
                                                         name:@"softmax"];
  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph logarithmWithTensor:softmaxTensor
                              name:@"log_softmax"];

  return Error::Ok;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
