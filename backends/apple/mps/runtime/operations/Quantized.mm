//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>
#include <executorch/backends/apple/mps/runtime/operations/OperationUtils.h>
#include <iostream>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {
Error
MPSGraphBuilder::mpsInt8PackedMMOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSInt8PackedMM();
  ET_LOG(
    Debug, "%s: (%d, %d, %d) -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->input2_id(),
    graphNode->input3_id(),
    graphNode->output_id()
  );

  MPSGraphTensor* ATensor = getMPSGraphTensor(graphNode->input1_id());
  MPSGraphTensor* BTensor = getMPSGraphTensor(graphNode->input2_id());
  MPSGraphTensor* scalesTensor = getMPSGraphTensor(graphNode->input3_id());

  auto castB = castMPSTensor(_mpsGraph, BTensor, ATensor.dataType);
  auto transposedB = [_mpsGraph transposeTensor:castB dimension:-1 withDimension:-2 name:@"int8packedmm/transposed"];
  auto mmTensor = [_mpsGraph matrixMultiplicationWithPrimaryTensor:ATensor
                                                                    secondaryTensor:transposedB
                                                                               name:@"int8packedmm/transposed*matmul"];

  _idToMPSGraphTensor[graphNode->output_id()] = [_mpsGraph multiplicationWithPrimaryTensor:mmTensor
                                                               secondaryTensor:scalesTensor
                                                                          name:@"int8packedmm/transposed*matmul*scales"];

  return Error::Ok;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
