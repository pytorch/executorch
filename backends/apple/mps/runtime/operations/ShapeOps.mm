
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
MPSGraphBuilder::mpsPermuteOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSPermute();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->output_id()
  );

  NSMutableArray<NSNumber*>* permutation = [NSMutableArray array];
  for(int64_t i = 0; i < graphNode->num_dims(); i++) {
    [permutation addObject:[NSNumber numberWithInteger:graphNode->perm()->Get(i)]];
  }
  MPSGraphTensor* outputTensor = permuteTensor(
    _mpsGraph, getMPSGraphTensor(graphNode->input1_id()), permutation
  );
  _idToMPSGraphTensor[graphNode->output_id()] = outputTensor;

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsViewOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSView();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph reshapeTensor:getMPSGraphTensor(graphNode->input1_id())
                  withShape:getMPSShape(graphNode->shape())
                       name:@"view_copy"];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsExpandOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSExpand();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());

  // In torch, -1 is passed for dimensions which are to stay the same size
  for (int32_t i = 0; i < inputTensor.shape.count; i++) {
    int expandDimVal = graphNode->shape()->Get(i);
    if (expandDimVal == -1) {
      [shape addObject:inputTensor.shape[i]];
    } else {
      [shape addObject:[NSNumber numberWithInteger:expandDimVal]];
    }
  }

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph broadcastTensor:inputTensor
                       toShape:shape
                          name:@"expand_copy"];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsCatOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSCat();
  ET_LOG(
    Debug, "%s: %d",
    __FUNCTION__, graphNode->output_id()
  );

  NSMutableArray<MPSGraphTensor*>* inputTensors = [NSMutableArray arrayWithCapacity:graphNode->input_ids()->size()];;
  for (auto id : *graphNode->input_ids()) {
    MPSGraphTensor* catTensor = getMPSGraphTensor(id);
    if (catTensor != nil)
      [inputTensors addObject:catTensor];
  }
  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph concatTensors:inputTensors
                   dimension:graphNode->dim()
                        name:@"cat"];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsSqueezeOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSSqueeze();
  ET_LOG(
    Debug, "%s: %d",
    __FUNCTION__, graphNode->output_id()
  );

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph squeezeTensor:getMPSGraphTensor(graphNode->input1_id())
                        axes:getMPSShape(graphNode->dims())
                        name:@"squeeze"];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsUnsqueezeOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSUnsqueeze();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph expandDimsOfTensor:getMPSGraphTensor(graphNode->input1_id())
                             axis:graphNode->dim()
                             name:@"unsqueeze"];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsSelectOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSSelect();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  MPSGraphTensor* slicedTensor = [_mpsGraph sliceTensor:getMPSGraphTensor(graphNode->input1_id())
                                              dimension:graphNode->dim()
                                                  start:graphNode->index()
                                                 length:1
                                                   name:@"slice"];
  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph squeezeTensor:slicedTensor
                        axis:graphNode->dim()
                        name:@"slice/squeezed"];

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsPixelShuffleOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSPixelShuffle();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());
  const int ndims = inputTensor.shape.count;
  MPSGraphTensor* outputTensor = nil;
  int32_t upscaleFactor = graphNode->upscale_factor();

  ET_CHECK_OR_RETURN_ERROR(
    ndims >= 3, Internal,  "pixel_shuffle requires tensor with at least 3 dimensions.");
  if (upscaleFactor == 1) {
    // TODO: move this to AOT
    outputTensor = inputTensor;
  } else {
    ET_CHECK_OR_RETURN_ERROR(
      inputTensor.shape[ndims - 3].intValue % (upscaleFactor * upscaleFactor) == 0,
      Internal,
      "pixel_shuffle channels must be divisible by upscale factor squared.");

    outputTensor = [_mpsGraph depthToSpace2DTensor:inputTensor
                                         widthAxis:ndims - 1
                                        heightAxis:ndims - 2
                                         depthAxis:ndims - 3
                                         blockSize:upscaleFactor
                              usePixelShuffleOrder:true
                                             name:@"pixel_shuffle"];
  }

  _idToMPSGraphTensor[graphNode->output_id()] = outputTensor;
  return Error::Ok;
}

Error
MPSGraphBuilder::mpsSliceOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSSlice();
  ET_LOG(
    Debug, "%s %d: %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );

  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());
  int64_t dim = graphNode->dim();

  // Define input arrays as required by MPSGraph API
  NSMutableArray<NSNumber*>* start_arr = [NSMutableArray arrayWithCapacity: inputTensor.shape.count];
  NSMutableArray<NSNumber*>* end_arr = [NSMutableArray arrayWithCapacity: inputTensor.shape.count];
  NSMutableArray<NSNumber*>* step_arr = [NSMutableArray arrayWithCapacity: inputTensor.shape.count];
  // Step needs to be set to one for all other dims
  for (int i = 0; i < inputTensor.shape.count; i++) {
    step_arr[i] = @1;
    end_arr[i] = inputTensor.shape[i];
    start_arr[i] = @0;
  }

  start_arr[dim] = [NSNumber numberWithInteger:graphNode->start()];
  end_arr[dim] = [NSNumber numberWithInteger:graphNode->end()];
  step_arr[dim] = [NSNumber numberWithInteger:graphNode->step()];

  _idToMPSGraphTensor[graphNode->output_id()] =
    [_mpsGraph sliceTensor:inputTensor
                   starts:start_arr
                     ends:end_arr
                  strides:step_arr
                     name:@"strided_slice"];
  return Error::Ok;
}

Error
MPSGraphBuilder::mpsSplitWithSizesOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSSplitWithSizes();
  ET_LOG(
    Debug, "%s: %d -> len(output)=%d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_ids()->size()
  );

  std::vector<MPSGraphTensor*> splitResults;
  NSArray<MPSGraphTensor*>* mpsGraphResults;

  mpsGraphResults = [_mpsGraph splitTensor:getMPSGraphTensor(graphNode->input1_id())
                                splitSizes:getMPSShape(graphNode->split_sizes())
                                      axis:graphNode->dim()
                                      name:@"split_size"];

  int crtIdx = 0;
  for (auto outId : *graphNode->output_ids()) {
    _idToMPSGraphTensor[outId] = mpsGraphResults[crtIdx++];
  }

  return Error::Ok;
}

Error
MPSGraphBuilder::mpsCastOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSCast();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__, graphNode->input1_id(), graphNode->output_id()
  );


  _idToMPSGraphTensor[graphNode->output_id()] = castMPSTensor(
    _mpsGraph, getMPSGraphTensor(graphNode->input1_id()), getMPSDataType(graphNode->dtype()));

  return Error::Ok;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
