
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
MPSGraphBuilder::mpsConstantOp(int32_t id) {
  _idToMPSGraphTensor[id] = [_mpsGraph constantWithData:getConstantData(id)
                                                  shape:getMPSShape(id)
                                               dataType:getMPSDataType(id)];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsFullOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSFull();
  ET_LOG(
    Debug, "%s: - -> %d",
    __FUNCTION__, graphNode->output_id()
  );

  if (numel(graphNode->shape()) == 0) {
    _idToMPSGraphTensor[graphNode->output_id()] = nil;
  } else {
    _idToMPSGraphTensor[graphNode->output_id()] =
      [_mpsGraph constantWithScalar:graphNode->fill_value()
                              shape:getMPSShape(graphNode->shape())
                           dataType:getMPSDataType(graphNode->dtype())];
  }

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsFullLikeOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSFullLike();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph constantWithScalar:graphNode->fill_value()
                            shape:getMPSGraphTensor(graphNode->input1_id()).shape
                         dataType:getMPSDataType(graphNode->dtype())];

  return Error::Ok;
}


} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
