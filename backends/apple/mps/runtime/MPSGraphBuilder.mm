//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>
#include <executorch/backends/apple/mps/runtime/MPSDevice.h>
#include <executorch/backends/apple/mps/runtime/MPSDelegateHeader.h>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

MPSGraphBuilder::MPSGraphBuilder(
  const void* buffer_pointer,
  size_t num_bytes,
  std::unordered_map<MPSGraphTensor*, int32_t>& mpsGraphTensorToId) :
    _mpsGraphTensorToId(mpsGraphTensorToId), _buffer_pointer(buffer_pointer), _num_bytes(num_bytes) {

  _mpsGraph = [MPSGraph new];
  _feeds = [NSMutableDictionary dictionary];
  _targetTensors = [NSMutableArray new];

  _mpsGraphExecutable = nil;
  _metal_kernel = false;
}

Error
MPSGraphBuilder::compileModel() {
  Error err = Error::Ok;

  Result<MPSDelegateHeader> header = MPSDelegateHeader::Parse(_buffer_pointer, _num_bytes);
  const uint8_t* flatbuffer_data_ptr = nullptr;

  if (header.ok()) {
    flatbuffer_data_ptr = reinterpret_cast<const uint8_t*>(_buffer_pointer) +
        header->flatbuffer_offset;
    _constant_data_ptr = reinterpret_cast<const uint8_t*>(_buffer_pointer) +
        header->constant_data_offset;
  } else if (header.error() == Error::NotFound) {
    ET_LOG(
        Error,
        "MPSDelegateHeader version mismatch: '%.4s' != expected '%.4s'",
        // Header Magic and FlatbufferIdentifier are same offset and size
        flatbuffers::GetBufferIdentifier(_buffer_pointer),
        MPSDelegateHeader::kMagic);
    return header.error();
  } else {
    ET_LOG(Error, "MPSDelegateHeader may be corrupt");
    return header.error();
  }

  ET_CHECK(flatbuffer_data_ptr != nullptr);
  ET_CHECK_OR_RETURN_ERROR(
    mpsgraph::MPSGraphBufferHasIdentifier(flatbuffer_data_ptr),
    DelegateInvalidCompatibility,
    "MPS Delegate Serialization Format version identifier '%.4s' != expected '%.4s'",
    flatbuffers::GetBufferIdentifier(flatbuffer_data_ptr),
    mpsgraph::MPSGraphIdentifier());

  _flatBufferGraph = mpsgraph::GetMPSGraph(flatbuffer_data_ptr);
  switch (_flatBufferGraph->graph_type()) {
    case mpsgraph::OpType::metal_kernel:
    {
      _metal_kernel = true;
      err = compileMetalKernel();
      break;
    }
    case mpsgraph::OpType::mps_graph:
    {
      err = compileMPSGraph();
      break;
    }
    default:
      ET_CHECK_OR_RETURN_ERROR(
      false,
      DelegateInvalidCompatibility,
      "Received an invalid operation type: expected MPSGraph or metal kernel, but got: %s",
      EnumNameOpType(_flatBufferGraph->graph_type()));
  }

  return err;
}

Error
MPSGraphBuilder::compileMPSGraph() {
  Error err = Error::Ok;

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
MPSGraphBuilder::compileMetalKernel() {
  Error err = Error::Ok;

  ET_CHECK_OR_RETURN_ERROR(
    _flatBufferGraph->mps_nodes()->size() == 1,
    DelegateInvalidCompatibility,
    "Currently supporting dispatching a single Metal kernel.");
  ET_CHECK_OR_RETURN_ERROR(
    _flatBufferGraph->constant_ids()->size() == 0,
    DelegateInvalidCompatibility,
    "Currently not supporting dispatching Metal kernels with constants.");

  // Compile the corresponding Metal kernel
  for (auto node : *_flatBufferGraph->mps_nodes()) {
    err = compileMetalKernel(node);
    if (err != Error::Ok) {
      return err;
    }
  }

  return err;
}

Error
MPSGraphBuilder::mpsGraphRankedPlaceholder(int32_t id) {
  ET_LOG(Debug, "%s: %d", __FUNCTION__, id);
  MPSShape* mpsShape = getMPSShape(id);
  MPSDataType mpsDataType = getMPSDataType(id);
  MPSGraphTensor* placeholder = [_mpsGraph placeholderWithShape:mpsShape
                                                  dataType:mpsDataType
                                                      name:nil];
  _idToMPSGraphTensor[id] = placeholder;
  _feeds[placeholder] = [[MPSGraphShapedType alloc] initWithShape:mpsShape
                                                         dataType:mpsDataType];
  _mpsGraphTensorToId[placeholder] = id;
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

  return _mpsGraphExecutable;

}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
