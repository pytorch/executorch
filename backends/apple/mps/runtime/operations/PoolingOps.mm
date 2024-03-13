
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
MPSGraphBuilder::mpsMaxPool2DWithIndicesOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSMaxPool2DWithIndices();
  ET_LOG(
    Debug, "%s: %d -> (%d, %d)",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->output1_id(),
    graphNode->output2_id()
  );

  MPSGraphPooling2DOpDescriptor* desc =
    [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:graphNode->kernel_width()
                                                kernelHeight:graphNode->kernel_height()
                                                   strideInX:graphNode->stride_width()
                                                   strideInY:graphNode->stride_height()
                                             dilationRateInX:graphNode->dilation_width()
                                             dilationRateInY:graphNode->dilation_height()
                                                 paddingLeft:graphNode->padding_left()
                                                paddingRight:graphNode->padding_right()
                                                  paddingTop:graphNode->padding_top()
                                               paddingBottom:graphNode->padding_bottom()
                                                paddingStyle:MPSGraphPaddingStyleExplicit
                                                  dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
  desc.ceilMode = graphNode->ceil_mode();
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunguarded-availability-new"
  desc.returnIndicesMode = MPSGraphPoolingReturnIndicesGlobalFlatten2D;
  desc.returnIndicesDataType = MPSDataTypeInt32;
#pragma clang diagnostic pop

  NSArray<MPSGraphTensor*>* outputs =
    [_mpsGraph maxPooling2DReturnIndicesWithSourceTensor:getMPSGraphTensor(graphNode->input1_id())
                                              descriptor:desc
                                                    name:@"MaxPool2DWithIndices"];


  _idToMPSGraphTensor[graphNode->output1_id()] = outputs[0];
  _idToMPSGraphTensor[graphNode->output2_id()] = outputs[1];
  return Error::Ok;
}

Error
MPSGraphBuilder::mpsAvgPool2DOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSAvgPool2D();
  ET_LOG(
    Debug, "%s: %d -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->output1_id()
  );

  MPSGraphPooling2DOpDescriptor* desc =
    [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:graphNode->kernel_width()
                                                kernelHeight:graphNode->kernel_height()
                                                   strideInX:graphNode->stride_width()
                                                   strideInY:graphNode->stride_height()
                                             dilationRateInX:graphNode->dilation_width()
                                             dilationRateInY:graphNode->dilation_height()
                                                 paddingLeft:graphNode->padding_left()
                                                paddingRight:graphNode->padding_right()
                                                  paddingTop:graphNode->padding_top()
                                               paddingBottom:graphNode->padding_bottom()
                                                paddingStyle:MPSGraphPaddingStyleExplicit
                                                  dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
  const bool useDivisor = graphNode->divisor_override() != 0;

  // If overriding divisor, zeroPads must be included to the average for correct behavior
  desc.includeZeroPadToAverage = useDivisor ? true : graphNode->count_include_pad();

  MPSGraphTensor* avgPoolTensor = [_mpsGraph avgPooling2DWithSourceTensor:getMPSGraphTensor(graphNode->input1_id())
                                                               descriptor:desc
                                                                     name:@"AvgPool2DTensor"];
  if (useDivisor) {
    // Here we rescale the average due to MPSGraph not supporting custom divisor directly
    const float divisor = float(graphNode->kernel_height() * graphNode->kernel_width()) / (float)graphNode->divisor_override();
    MPSGraphTensor* constantTensor = [_mpsGraph constantWithScalar:divisor
                                                             shape:@[@1]
                                                          dataType:MPSDataTypeFloat32];
    avgPoolTensor = [_mpsGraph multiplicationWithPrimaryTensor:avgPoolTensor
                                               secondaryTensor:constantTensor
                                                          name:@"AvgPool2DTensor/divisor_override"];

  }

  _idToMPSGraphTensor[graphNode->output1_id()] = avgPoolTensor;

  return Error::Ok;
}



} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
