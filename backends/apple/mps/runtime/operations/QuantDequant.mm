//
//  Copyright (c) 2024 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

Error
MPSGraphBuilder::mpsDequantizePerChannelGroupOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSDequantizePerChannelGroup();
  ET_LOG(
    Debug, "%s: (%d, %d, %d) -> %d",
    __FUNCTION__,
    graphNode->input1_id(),
    graphNode->scales_id(),
    graphNode->zero_points_id(),
    graphNode->output_id()
  );

  ET_CHECK_OR_RETURN_ERROR(
    is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS),
    NotImplemented,
    "[ERROR] Operation %s is supported starting with macOS 15.0+ | iOS 18.0 + | iPadOS 18+ | tvOS 18+ | visionOS 2.0+ !",
    mpsgraph::EnumNameMPSNodeUnion(nodePtr->mpsnode_union_type()));

  MPSGraphTensor* inputTensor = getMPSGraphTensor(graphNode->input1_id());
  MPSGraphTensor* scalesTensor = getMPSGraphTensor(graphNode->scales_id());
  if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
    MPSGraphTensor *zpTensor = [_mpsGraph constantWithScalar:0
                                                  dataType:MPSDataTypeInt4];
    MPSGraphTensor *wDqTensor = [_mpsGraph dequantizeTensor:inputTensor
                                                scaleTensor:scalesTensor
                                            zeroPointTensor:zpTensor
                                                  dataType:MPSDataTypeFloat16
                                                      name:nil];
    _idToMPSGraphTensor[graphNode->output_id()] = wDqTensor;
  } else {
    _idToMPSGraphTensor[graphNode->output_id()] = nil;
  }

  return Error::Ok;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
