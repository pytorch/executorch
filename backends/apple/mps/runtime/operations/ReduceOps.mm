
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
MPSGraphBuilder::mpsMeanOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSMean();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->output_id()
  );

  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());

  //MPSGraph wants negative axes to be converted to positive
  const int inputDims = [inputTensor.shape count];

  NSMutableArray<NSNumber*>* dimArray = [NSMutableArray array];
  for(int64_t i = 0; i < graphNode->num_dims(); i++) {
    int32_t dim = graphNode->dims()->Get(i);
    if (dim < 0) {
      dim = inputDims + dim;
    }
    [dimArray addObject:[NSNumber numberWithInt:dim]];
  }

  // Reverting back to get the ordering back to slowest axis first as MPSGraph expects
  dimArray = [[[dimArray reverseObjectEnumerator] allObjects] mutableCopy];

  MPSGraphTensor* meanTensor = [_mpsGraph meanOfTensor:inputTensor
                                                  axes:dimArray
                                                  name:@"Mean"];
  if (!graphNode->keep_dims()) {
    meanTensor = [_mpsGraph squeezeTensor:meanTensor
                                     axes:dimArray
                                     name:@"Mean/squeezed"];
  }

  _idToMPSGraphTensor[graphNode->output_id()] = meanTensor;
  return Error::Ok;
}


} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
