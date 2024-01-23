//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>
#include <executorch/backends/apple/mps/runtime/MPSDevice.h>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

MPSGraphBuilder::MPSGraphBuilder(const void* buffer_pointer) : _buffer_pointer(buffer_pointer) {
  _mpsGraph = [MPSGraph new];
  _feeds = [NSMutableDictionary dictionary];
  _targetTensors = [NSMutableArray new];

  _mpsGraphExecutable = nil;
}

Error
MPSGraphBuilder::compileModel() {
  Error err = Error::Ok;

  ET_CHECK(_buffer_pointer != nullptr);
  ET_CHECK_OR_RETURN_ERROR(
    mpsgraph::MPSGraphBufferHasIdentifier(_buffer_pointer),
    DelegateInvalidCompatibility,
    "MPS Delegate Serialization Format version identifier '%.4s' != expected '%.4s'",
    flatbuffers::GetBufferIdentifier(_buffer_pointer),
    mpsgraph::MPSGraphIdentifier());

  _flatBufferGraph = mpsgraph::GetMPSGraph(_buffer_pointer);
  _idToMPSGraphTensor.resize(_flatBufferGraph->mps_values()->size(), nullptr);

  // Add the placeholder nodes to the graph.
  for (auto in_id : *_flatBufferGraph->input_ids()) {
    err = mpsGraphRankedPlaceholder(in_id);
    if (err != Error::Ok) {
      return err;
    }
  }

  // Parse all the serialized constant values and add them to MPSGraph.
  for (auto constant_id : *_flatBufferGraph->constant_ids()) {
    err = mpsConstantOp(constant_id);
    if (err != Error::Ok) {
      return err;
    }
  }

  // Create the corresponding MPSGraph ops of the serialized nodes from the FlatBuffer.
  for (auto node : *_flatBufferGraph->mps_nodes()) {
    err = addNodeToMPSGraph(node);
    if (err != Error::Ok) {
      return err;
    }
  }

  // Add the output nodes to the MPSGraphExecutable.
  for (auto out_id : *_flatBufferGraph->output_ids()) {
    ET_CHECK_OR_RETURN_ERROR(
      _idToMPSGraphTensor[out_id] != nil,
      InvalidState,
      "Failed to deserialize the model");

    [_targetTensors addObject: _idToMPSGraphTensor[out_id]];
  }

  return err;
}

Error
MPSGraphBuilder::mpsGraphRankedPlaceholder(int32_t id) {
  ET_LOG(Debug, "%s: %d", __FUNCTION__, id);
  MPSShape* mpsShape = getMPSShape(id);
  MPSDataType mpsDataType = getMPSDataType(id);
  _idToMPSGraphTensor[id] = [_mpsGraph placeholderWithShape:mpsShape
                                                  dataType:mpsDataType
                                                      name:nil];
  _feeds[_idToMPSGraphTensor[id]] = [[MPSGraphShapedType alloc] initWithShape:mpsShape
                                                                     dataType:mpsDataType];
  return Error::Ok;
}

MPSGraph*
MPSGraphBuilder::getMPSGraph() {
  return _mpsGraph;
}

MPSGraphExecutable*
MPSGraphBuilder::getMPSGraphExecutable() {
  if (_mpsGraphExecutable) {
    return _mpsGraphExecutable;
  }

  _mpsGraphExecutable = [_mpsGraph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:MPSDevice::getInstance()->device()]
                                               feeds:_feeds
                                       targetTensors:_targetTensors
                                    targetOperations:nil
                               compilationDescriptor:nil];


  // [_mpsGraphExecutable specializeWithDevice:[MPSGraphDevice deviceWithMTLDevice:MPSDevice::getInstance()->device()]
  //                 inputTypes:[_feeds allValues]
  //      compilationDescriptor:nil];

  return _mpsGraphExecutable;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
