
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
MPSGraphBuilder::mpsDepthwiseConv2DOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSDepthwiseConv2D();
  ET_LOG(
    Debug, "%s: (%d, %d, %d) -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->input2_id(),
    graphNode->input3_id(),
    graphNode->output_id()
  );

  bool isConv1D = ([getMPSShape(graphNode->input2_id()) count] == 3);
  ET_CHECK(!isConv1D);

  MPSGraphDepthwiseConvolution3DOpDescriptor* depthWiseConv3dDescriptor =
    [[MPSGraphDepthwiseConvolution3DOpDescriptor new] autorelease];

  depthWiseConv3dDescriptor.strides =
      @[ @1, [[NSNumber alloc] initWithInteger:graphNode->stride_y()], [[NSNumber alloc] initWithInteger:graphNode->stride_x()] ];

  depthWiseConv3dDescriptor.dilationRates =
      @[ @1, [[NSNumber alloc] initWithInteger:graphNode->dilation_y()], [[NSNumber alloc] initWithInteger:graphNode->dilation_x()] ];

  depthWiseConv3dDescriptor.paddingStyle = MPSGraphPaddingStyleExplicit;
  depthWiseConv3dDescriptor.paddingValues = @[
    @0,
    @0,
    [[NSNumber alloc] initWithInteger:graphNode->padding_top()],
    [[NSNumber alloc] initWithInteger:graphNode->padding_bottom()],
    [[NSNumber alloc] initWithInteger:graphNode->padding_left()],
    [[NSNumber alloc] initWithInteger:graphNode->padding_right()]
  ];
  depthWiseConv3dDescriptor.channelDimensionIndex = -3LL;
  MPSGraphTensor* weightTransposeTensor = [_mpsGraph transposeTensor:getMPSGraphTensor(graphNode->input2_id())
                                                          dimension:-3
                                                      withDimension:-4
                                                                name:nil];
  MPSGraphTensor* depthwiseConvTensor = [_mpsGraph depthwiseConvolution3DWithSourceTensor:getMPSGraphTensor(graphNode->input1_id())
                                                                            weightsTensor:weightTransposeTensor
                                                                               descriptor:depthWiseConv3dDescriptor
                                                                                    name:nil];
  // Bias is optional
  if (graphNode->input3_id() != -1) {
    //Need to add correct dimension to bias to avoid broadcasting issues
    MPSGraphTensor* biasTensor = getMPSGraphTensor(graphNode->input3_id());
    biasTensor = [_mpsGraph expandDimsOfTensor:biasTensor
                                      axes:@[@0, @2, @3]
                                      name:nil];
    depthwiseConvTensor = [_mpsGraph additionWithPrimaryTensor:depthwiseConvTensor
                                                secondaryTensor:biasTensor
                                                            name:@"depthwiseConv2DWithBiasAdd"];
  }

  _idToMPSGraphTensor[graphNode->output_id()] = depthwiseConvTensor;
  return Error::Ok;
}

Error
MPSGraphBuilder::mpsConv2DOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSConv2D();
  ET_LOG(
    Debug, "%s: (%d, %d, %d) -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->input2_id(),
    graphNode->input3_id(),
    graphNode->output_id()
  );

  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());
  MPSGraphTensor* weightTensor = getMPSGraphTensor(graphNode->input2_id());

  bool isConv1D = ([weightTensor.shape count] == 3);
  if (isConv1D) {
    inputTensor = [_mpsGraph expandDimsOfTensor:inputTensor
                                            axis:2
                                            name:@"unsqueezeInput"];
    weightTensor = [_mpsGraph expandDimsOfTensor:weightTensor
                                              axis:2
                                              name:@"unsqueezeWeight"];
  }

  MPSGraphConvolution2DOpDescriptor* desc =
    [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:graphNode->stride_x()
                                                     strideInY:graphNode->stride_y()
                                            dilationRateInX:graphNode->dilation_x()
                                            dilationRateInY:graphNode->dilation_y()
                                                     groups:graphNode->groups()
                                                paddingLeft:graphNode->padding_left()
                                               paddingRight:graphNode->padding_right()
                                                 paddingTop:graphNode->padding_top()
                                              paddingBottom:graphNode->padding_bottom()
                                               paddingStyle:MPSGraphPaddingStyleExplicit
                                                 dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                              weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
    // Convert weights from OIHW to HWIO.
    MPSGraphTensor* weightTransposeTensor = permuteTensor(_mpsGraph, weightTensor, @[@2, @3, @1, @0]);
    MPSGraphTensor* conv2DTensor = [_mpsGraph convolution2DWithSourceTensor:inputTensor
                                                             weightsTensor:weightTransposeTensor
                                                                descriptor:desc
                                                                      name:@"conv2D"];

    // Bias is optional
    if (graphNode->input3_id() != -1) {
      // Need to add correct dimension to bias to avoid broadcasting issues
      MPSGraphTensor* biasTensor = getMPSGraphTensor(graphNode->input3_id());
      biasTensor = [_mpsGraph expandDimsOfTensor:biasTensor
                                           axes:@[@0,@2,@3]
                                           name:nil];
        conv2DTensor = [_mpsGraph additionWithPrimaryTensor:conv2DTensor
                                           secondaryTensor:biasTensor
                                                      name:@"conv2DWithBiasAdd"];
    }

  if (isConv1D) {
    conv2DTensor = [_mpsGraph squeezeTensor:conv2DTensor
                                       axis:2
                                       name:@"squeeze"];
  }

  _idToMPSGraphTensor[graphNode->output_id()] = conv2DTensor;
  return Error::Ok;
}


} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
