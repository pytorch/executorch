//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "utils/MPSGraphInterface.h"

namespace mps {
using namespace torch;

PyMPSGraphTensor*
MPSGraphModule::conv2D(MPSGraphTensor* primaryTensor, MPSGraphTensor* secondaryTensor,
                       MPSGraphTensor* biasTensor, IntArrayRef stride,
                       IntArrayRef padding, IntArrayRef dilation, bool transpose,
                       IntArrayRef outputPadding, int64_t groups, bool is_depthwise) {

  if(is_depthwise){
    MPSGraphDepthwiseConvolution2DOpDescriptor* desc = [MPSGraphDepthwiseConvolution2DOpDescriptor
                                    descriptorWithStrideInX:stride[0]
                                                  strideInY:stride[1]
                                            dilationRateInX:dilation[0]
                                            dilationRateInY:dilation[1]
                                                paddingLeft:padding[1]
                                               paddingRight:padding[1]
                                                 paddingTop:padding[0]
                                              paddingBottom:padding[0]
                                               paddingStyle:MPSGraphPaddingStyleExplicit
                                                 dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                              weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    MPSGraphTensor* depthwiseConv2DTensor = [mpsGraph depthwiseConvolution2DWithSourceTensor:primaryTensor
                                                                               weightsTensor:secondaryTensor
                                                                                  descriptor:desc
                                                                                        name:@"depthwiseConv2D"];
    //Can be a nullptr
    if(biasTensor){
        //Need to add correct dimension to bias to avoid broadcasting issues
        biasTensor = [mpsGraph expandDimsOfTensor:biasTensor
                                          axes:@[@0, @2, @3]
                                          name:nil];
        depthwiseConv2DTensor = [mpsGraph additionWithPrimaryTensor:depthwiseConv2DTensor
                                                    secondaryTensor:biasTensor
                                                               name:@"depthwiseConv2DWithBiasAdd"];
    }

    return depthwiseConv2DTensor;
  } else {
    MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
                                    descriptorWithStrideInX:stride[0]
                                                  strideInY:stride[1]
                                            dilationRateInX:dilation[0]
                                            dilationRateInY:dilation[1]
                                                     groups:groups
                                                paddingLeft:padding[1]
                                               paddingRight:padding[1]
                                                 paddingTop:padding[0]
                                              paddingBottom:padding[0]
                                               paddingStyle:MPSGraphPaddingStyleExplicit
                                                 dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                              weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

    MPSGraphTensor* conv2DTensor = [mpsGraph convolution2DWithSourceTensor:primaryTensor
                                                             weightsTensor:secondaryTensor
                                                                descriptor:desc
                                                                      name:@"conv2D"];

    //Can be a nullptr
    if(biasTensor){
        //Need to add correct dimension to bias to avoid broadcasting issues
        biasTensor = [mpsGraph expandDimsOfTensor:biasTensor
                                              axes:@[@0,@2,@3]
                                              name:nil];
        conv2DTensor = [mpsGraph additionWithPrimaryTensor:conv2DTensor
                                           secondaryTensor:biasTensor
                                                      name:@"conv2DWithBiasAdd"];
    }
    return conv2DTensor;
  }
}
} //namespace mps
