
//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {


MPSGraphTensor* indexSelect(
  MPSGraphTensor* inputTensor,
  int64_t dim,
  MPSGraphTensor* indexTensor,
  MPSGraph* mpsGraph) {

  MPSGraphTensor* castIndexTensor = indexTensor;
  if(castIndexTensor.dataType != MPSDataTypeInt32) {
    castIndexTensor = [mpsGraph castTensor:indexTensor
                                     toType:MPSDataTypeInt32
                                       name:nil];
  }

  return  [mpsGraph gatherWithUpdatesTensor:inputTensor
                              indicesTensor:castIndexTensor
                                       axis:dim
                            batchDimensions:0
                                       name:nil];
}

Error
MPSGraphBuilder::mpsIndexSelectOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSIndexSelect();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->output_id()
  );

  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());
  MPSGraphTensor* indexTensor = getMPSGraphTensor(graphNode->index_id());
  MPSGraphTensor* castIndexTensor = indexTensor;
  if(castIndexTensor.dataType != MPSDataTypeInt32) {
    castIndexTensor = [_mpsGraph castTensor:indexTensor
                                     toType:MPSDataTypeInt32
                                       name:nil];
  }

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph gatherWithUpdatesTensor:inputTensor
                         indicesTensor:castIndexTensor
                                  axis:graphNode->dim()
                       batchDimensions:0
                                  name:nil];
  return Error::Ok;
}

Error
MPSGraphBuilder::mpsEmbeddingOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSEmbedding();
  ET_LOG(
    Debug, "%s: (%d, %d) -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->input2_id(),
    graphNode->output_id()
  );


  MPSGraphTensor* weightTensor = getMPSGraphTensor(graphNode->input1_id());
  MPSGraphTensor* indicesTensor = getMPSGraphTensor(graphNode->input2_id());
  int padding_idx = graphNode->padding_idx();

  if (padding_idx != -1) {
    MPSGraphTensor* constantTensor = [_mpsGraph constantWithScalar:padding_idx
                                                             shape:@[@1]
                                                          dataType:indicesTensor.dataType];

    MPSGraphTensor* notEqualTensor = [_mpsGraph notEqualWithPrimaryTensor:indicesTensor
                                                          secondaryTensor:constantTensor
                                                                     name:nil];
    MPSGraphTensor* condition = [_mpsGraph expandDimsOfTensor:notEqualTensor
                                                  axis:-1
                                                  name:@"unsqueeze"];
    MPSGraphTensor* valTensor = indexSelect(weightTensor, 0, indicesTensor, _mpsGraph);
    MPSGraphTensor* zeroTensor = [_mpsGraph constantWithScalar:0
                                                         shape:valTensor.shape
                                                      dataType:valTensor.dataType];
    _idToMPSGraphTensor[graphNode->output_id()] =
      [_mpsGraph selectWithPredicateTensor:condition
                       truePredicateTensor:valTensor
                      falsePredicateTensor:zeroTensor
                                       name:nil];
  } else {
    _idToMPSGraphTensor[graphNode->output_id()] = indexSelect(
      getMPSGraphTensor(graphNode->input1_id()),
      0,
      getMPSGraphTensor(graphNode->input2_id()),
      _mpsGraph
    );
  }

  return Error::Ok;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
