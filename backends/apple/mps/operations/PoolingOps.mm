//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"
#include <ATen/native/mps/OperationUtils.h>

namespace mps {
using namespace torch;

std::tuple<PyMPSGraphTensor*, PyMPSGraphTensor*>
MPSGraphModule::maxPool2DWithIndices(MPSGraphTensor* inputTensor,
                                     IntArrayRef kernel_size,
                                     IntArrayRef stride,
                                     IntArrayRef padding,
                                     IntArrayRef dilation,
                                     bool ceil_mode) {

  int padH = padding[0];
  int padW = padding.size() == 1 ? padH : padding[1];
  const int kH = kernel_size[0];
  const int kW = kernel_size.size() == 1 ? kH : kernel_size[1];
  const int dH = stride.empty() ? kH : stride[0];
  const int dW = stride.empty() ? kW : stride.size() == 1 ? dH : stride[1];
  const int dilationH = dilation[0];
  const int dilationW = dilation.size() == 1 ? dilationH : dilation[1];

  MPSGraphPooling2DOpDescriptor* desc = [MPSGraphPooling2DOpDescriptor
      descriptorWithKernelWidth:kW
                   kernelHeight:kH
                      strideInX:dW
                      strideInY:dH
                dilationRateInX:dilationW
                dilationRateInY:dilationH
                    paddingLeft:padW
                   paddingRight:ceil_mode ? padW * dW : padW
                     paddingTop:padH
                  paddingBottom:ceil_mode ? padH * dH : padH
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
  desc.ceilMode = (padW == 0 && padH == 0) ? ceil_mode : false;
  desc.returnIndicesMode = MPSGraphPoolingReturnIndicesGlobalFlatten2D;
  desc.returnIndicesDataType = MPSDataTypeInt32;

  NSArray<MPSGraphTensor*>* outputs = [mpsGraph maxPooling2DReturnIndicesWithSourceTensor:inputTensor
                                                                 descriptor:desc
                                                                       name:@"MaxPool2DWithIndices"];


  return std::make_tuple(outputs[0], outputs[1]);
}


PyMPSGraphTensor*
MPSGraphModule::avgPool2D(MPSGraphTensor* inputTensor,
                                     IntArrayRef kernel_size,
                                     IntArrayRef stride,
                                     IntArrayRef padding,
                                     bool ceil_mode,
                                     bool count_include_pad,
                                     c10::optional<int> divisor_override) {
  int padH = padding[0];
  int padW = padding.size() == 1 ? padH : padding[1];
  const int kH = kernel_size[0];
  const int kW = kernel_size.size() == 1 ? kH : kernel_size[1];
  const int dH = stride.empty() ? kH : stride[0];
  const int dW = stride.empty() ? kW : stride.size() == 1 ? dH : stride[1];
  const int dilationH = 1;
  const int dilationW = 1;

  MPSGraphPooling2DOpDescriptor* desc = [MPSGraphPooling2DOpDescriptor
      descriptorWithKernelWidth:kW
                   kernelHeight:kH
                      strideInX:dW
                      strideInY:dH
                dilationRateInX:dilationW
                dilationRateInY:dilationH
                    paddingLeft:padW
                   paddingRight:ceil_mode ? padW * dW : padW
                     paddingTop:padH
                  paddingBottom:ceil_mode ? padH * dH : padH
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:MPSGraphTensorNamedDataLayoutNCHW];

  const bool use_divisor = divisor_override.has_value() && divisor_override.value() != 0;

  //if overriding divisor, zeroPads must be included to the average for correct behavior
  desc.includeZeroPadToAverage = use_divisor ? true : count_include_pad;

  MPSGraphTensor* avgPoolTensor = [mpsGraph avgPooling2DWithSourceTensor:inputTensor
                                                              descriptor:desc
                                                                    name:@"AvgPool2DTensor"];
  if(use_divisor) {
    //here we rescale the average due to MPSGraph not supporting custom divisor directly
    const float divisor = float(kH * kW) / (float)divisor_override.value();
    MPSGraphTensor* constantTensor = [mpsGraph constantWithScalar:divisor
                                                            shape:@[@1]
                                                         dataType:MPSDataTypeFloat32];
    avgPoolTensor = [mpsGraph multiplicationWithPrimaryTensor:avgPoolTensor
                                              secondaryTensor:constantTensor
                                                         name:@"AvgPool2DTensor/divisor_override"];

  }

  return avgPoolTensor;

}
} //namespace
