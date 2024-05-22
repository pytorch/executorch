
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
                                       name:@"castTensor"];
  }

  return  [mpsGraph gatherWithUpdatesTensor:inputTensor
                              indicesTensor:castIndexTensor
                                       axis:dim
                            batchDimensions:0
                                       name:@"indexSelect"];
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
                                       name:@"castTensor"];
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

Error
MPSGraphBuilder::mpsIndexTensorOp(NodePtr nodePtr) {
  Error err = Error::Ok;
  auto graphNode = nodePtr->mpsnode_union_as_MPSIndexTensor();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  if (_metal_kernel) {
    err = MPSDevice::getInstance()->compilePSO(LibraryType::INDEXING_KERNELS, "index_select");
    ET_CHECK_MSG(false, "Metal kernel path not yet implemented\n");
  } else {
    int validIndices = 0;
    int numIndices = graphNode->indices_id()->size();
    int axis = -1;
    int indexId = -1;
    for (int i = 0; i < numIndices; i++) {
      int32_t index_id = graphNode->indices_id()->Get(i);
      if (index_id == -1) {
        continue;
      }
      validIndices++;
      axis = i;
      indexId = index_id;
    }
    ET_LOG(Debug, "index.Tensor with %d indices (axis = %d)", validIndices, axis);
    ET_CHECK(validIndices > 0);

    if (validIndices == 1) {
      MPSGraphTensor* updatesTensor = getMPSGraphTensor(graphNode->input1_id());
      MPSGraphTensor* indexTensor = getMPSGraphTensor(indexId);
      _idToMPSGraphTensor[graphNode->output_id()] =
        [_mpsGraph gatherWithUpdatesTensor:updatesTensor indicesTensor:indexTensor axis:axis batchDimensions:0 name:nil];
    } else {
      ET_CHECK_MSG(false, "Not yet implemented");
    }
  }

  return err;
}

Error
MPSGraphBuilder::mpsIndexPutOp(NodePtr nodePtr) {
  Error err = Error::Ok;
  auto graphNode = nodePtr->mpsnode_union_as_MPSIndexPut();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  if (_metal_kernel) {
    err = MPSDevice::getInstance()->compilePSO(LibraryType::INDEXING_KERNELS, "index_put");
    ET_CHECK_MSG(false, "Metal kernel path not yet implemented\n");
  } else {
    int validIndices = 0;
    int numIndices = graphNode->indices_id()->size();
    int axis = -1;
    int indexId = -1;
    for (int i = 0; i < numIndices; i++) {
      int32_t index_id = graphNode->indices_id()->Get(i);
      if (index_id == -1) {
        continue;
      }
      validIndices++;
      axis = i;
      indexId = index_id;
    }
    ET_LOG(Debug, "index_put with %d indices (axis = %d)", validIndices, axis);
    ET_CHECK(validIndices > 0);

    if (validIndices == 1) {
      MPSGraphTensor* dataTensor = getMPSGraphTensor(graphNode->input1_id());
      MPSGraphTensor* updatesTensor = getMPSGraphTensor(graphNode->values_id());
      MPSGraphTensor* indicesTensor = getMPSGraphTensor(indexId);
      if (graphNode->values_shape()->size() != 0) {
        updatesTensor = [_mpsGraph broadcastTensor:updatesTensor
                                       toShape:getMPSShape(graphNode->values_shape())
                                            name:nil];
      }

      _idToMPSGraphTensor[graphNode->output_id()] =
        [_mpsGraph scatterWithDataTensor:dataTensor
                           updatesTensor:updatesTensor
                           indicesTensor:indicesTensor
                                    axis:axis
                                    mode:MPSGraphScatterModeSet
                                  name:nil];
    } else {
      ET_CHECK_MSG(false, "Not yet implemented");
    }
  }

  return err;
}

Error
MPSGraphBuilder::mpsScatterOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSScatter();
  ET_LOG(
    Debug, "%s %d: %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  int64_t dim = graphNode->dim();
  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());
  MPSGraphTensor* indicesTensor = getMPSGraphTensor(graphNode->idx_id());
  MPSGraphTensor* updatesTensor = getMPSGraphTensor(graphNode->src_id());

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph scatterAlongAxis:dim
                 withDataTensor:inputTensor
                  updatesTensor:updatesTensor
                  indicesTensor:indicesTensor
                           mode:MPSGraphScatterModeSet
                           name:nil];
  return Error::Ok;
}


} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
