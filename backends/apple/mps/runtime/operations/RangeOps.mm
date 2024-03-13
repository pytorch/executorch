
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
MPSGraphBuilder::mpsArangeOp(NodePtr nodePtr) {
  auto graphNode = nodePtr->mpsnode_union_as_MPSArange();
  ET_LOG(
    Debug, "%s: () -> %d",
    __FUNCTION__,
    graphNode->output_id()
  );

  auto start = graphNode->start();
  auto end = graphNode->end();
  auto step = graphNode->step();
  MPSDataType dataType = getMPSDataType(graphNode->dtype());

  int32_t size_d = std::ceil(static_cast<double>(end - start) / step);
  auto shapeTensor = [_mpsGraph constantWithData:[NSData dataWithBytes:&size_d length:sizeof(int32_t)]
                                        shape:@[ @1 ]
                                      dataType:MPSDataTypeInt32];
  auto startScalar = start;
  auto stepScalar = step;
  auto coordsTensor = [_mpsGraph coordinateAlongAxis:0 withShapeTensor:shapeTensor name:nil];
  coordsTensor = [_mpsGraph castTensor:coordsTensor toType:dataType name:@"coords"];

  auto startTensor = [_mpsGraph constantWithScalar:startScalar
                            dataType:dataType];
  auto multiplyTensor = [_mpsGraph constantWithScalar:stepScalar
                            dataType:dataType];
  auto scaledCoords = [_mpsGraph multiplicationWithPrimaryTensor:coordsTensor
                                              secondaryTensor:multiplyTensor
                                                          name:nil];
  _idToMPSGraphTensor[graphNode->output_id()] = [_mpsGraph additionWithPrimaryTensor:scaledCoords secondaryTensor:startTensor name:nil];

  return Error::Ok;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
