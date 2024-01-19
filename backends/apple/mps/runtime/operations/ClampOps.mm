
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
MPSGraphBuilder::mpsClampOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSClamp();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->output_id()
  );

  std::pair<float, float> minMaxValues = getMinMaxValues(nodePtr);
  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());
  bool useMin = minMaxValues.first != -INF;
  bool useMax = minMaxValues.second != INF;

  if (useMin && useMax) {
    // Both min and max values are set
    MPSGraphTensor* minTensor = [_mpsGraph constantWithScalar:minMaxValues.first
                                                       shape:inputTensor.shape
                                                    dataType:inputTensor.dataType];
    MPSGraphTensor* maxTensor = [_mpsGraph constantWithScalar:minMaxValues.second
                                                       shape:inputTensor.shape
                                                    dataType:inputTensor.dataType];

    _idToMPSGraphTensor[graphNode->output_id()] = [_mpsGraph clampWithTensor:inputTensor
                                                              minValueTensor:minTensor
                                                              maxValueTensor:maxTensor
                                                                        name:@"clamp"];
  } else if (useMin && !useMax) {
    // Only min is set
    MPSGraphTensor* minTensor = [_mpsGraph constantWithScalar:minMaxValues.first
                                                       shape:inputTensor.shape
                                                    dataType:inputTensor.dataType];
    _idToMPSGraphTensor[graphNode->output_id()] = [_mpsGraph maximumWithPrimaryTensor:inputTensor
                                                                      secondaryTensor:minTensor
                                                                                 name:nil];
  } else if (!useMin && useMax) {
    // Only max is set
    MPSGraphTensor* maxTensor = [_mpsGraph constantWithScalar:minMaxValues.second
                                                    shape:inputTensor.shape
                                                dataType:inputTensor.dataType];
    _idToMPSGraphTensor[graphNode->output_id()] = [_mpsGraph minimumWithPrimaryTensor:inputTensor
                                                                     secondaryTensor:maxTensor
                                                                                name:nil];
  }
  return Error::Ok;
}

Error
MPSGraphBuilder::mpsWhereOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSWhere();
  ET_LOG(
    Debug, "%s: (%d, %d, %d) -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->input2_id(),
    graphNode->input3_id(),
    graphNode->output_id()
  );

  MPSGraphTensor* condition = getMPSGraphTensor(graphNode->input1_id());
  MPSGraphTensor* input = getMPSGraphTensor(graphNode->input2_id());
  MPSGraphTensor* other = getMPSGraphTensor(graphNode->input3_id());

  if ([condition dataType] != MPSDataTypeBool) {
    condition = [_mpsGraph castTensor:condition
                               toType:MPSDataTypeBool
                                 name:@"condition"];
  }
  _idToMPSGraphTensor[graphNode->output_id()]  = [_mpsGraph selectWithPredicateTensor:condition
                                                     truePredicateTensor:input
                                                    falsePredicateTensor:other
                                                                    name:nil];
  return Error::Ok;
}


} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
