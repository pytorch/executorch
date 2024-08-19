//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//
// clang-format off
#pragma once


#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <executorch/backends/apple/mps/runtime/operations/OperationUtils.h>
#include <executorch/backends/apple/mps/runtime/MPSStream.h>

#include <map>
#include <memory>
#include <vector>

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

class MPSExecutor {
 private:
  MPSGraphExecutable* _executable;
  NSArray<MPSGraphShapedType *> * _inputShapes;
  NSArray<MPSGraphShapedType *> * _outputShapes;

  NSMutableArray<MPSGraphTensorData *>* _inputsArray;
  NSMutableArray<MPSGraphTensorData *>* _outputsArray;

  // Flag whatever to use shared memory or not
  // Shared memory flag will be set as following (based on HW and target config):
  //   - True: Apple Silicon and macOS15+/iOS17+/iPadOS17+
  //   - False: Simulator or x86 or pre-macOS15/iOS17/iPadOS17
  bool _use_shared_mem;
  bool _buffers_initialized;

  // Input/Output GPU buffer pointer
  std::vector<id<MTLBuffer>> _inputGPUBuffers;
  std::vector<id<MTLBuffer>> _outputGPUBuffers;

  // Input/Output CPU buffer pointers
  std::vector<CPUBufferWrapper> _inputCPUBuffers;
  std::vector<CPUBufferWrapper> _outputCPUBuffers;

  std::unordered_map<MPSGraphTensor*, int32_t> _mpsGraphTensorToId;
 public:
  MPSExecutor();
  ~MPSExecutor() {
    if (_inputsArray) {
      [_inputsArray release];
      _inputsArray = nil;
    }
    if (_outputsArray) {
      [_outputsArray release];
    }

    _inputsArray = nil;
    _outputsArray = nil;
  }

  inline size_t getNumInputs() {
    return [_inputShapes count];
  }

  inline size_t getNumOutputs() {
    return [_outputShapes count];
  }

  inline MPSGraphExecutable* getMPSGraphExecutable() {
    return _executable;
  }

  ET_NODISCARD Error forward(std::vector<const Tensor*>& outputs);

  ET_NODISCARD Error
  set_inputs_outputs(std::vector<const Tensor*>& inputs, std::vector<const Tensor*>& outputs);

  Error initDataBuffers();
  Error updateDataBuffers(std::vector<const Tensor*>& inputs, std::vector<const Tensor*>& outputs);
  Error syncOutputBuffers(std::vector<const Tensor*>& outputs);

  friend class MPSCompiler;
};

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
// clang-format on
