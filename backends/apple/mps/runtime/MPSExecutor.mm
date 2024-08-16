//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/backends/apple/mps/schema_generated.h>
#include <executorch/backends/apple/mps/runtime/MPSExecutor.h>
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface MPSGraphExecutable()
-(NSArray<MPSGraphShapedType *> *) getInputShapes;
-(NSArray<MPSGraphShapedType *> *) getOutputShapes;
@end


namespace torch {
namespace executor {
namespace mps {
namespace delegate {

MPSExecutor::MPSExecutor() {
  _use_shared_mem = true;
  _buffers_initialized = false;

#if TARGET_OS_SIMULATOR or defined(__x86_64__)
  _use_shared_mem = false;
#endif
  if (!is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_0_PLUS)) {
    _use_shared_mem = false;
  }

  _inputsArray = [[NSMutableArray<MPSGraphTensorData *> alloc]  initWithCapacity:getNumInputs()];
  _outputsArray = [[NSMutableArray<MPSGraphTensorData *> alloc] initWithCapacity:getNumOutputs()];
}

__ET_NODISCARD Error
MPSExecutor::set_inputs_outputs(std::vector<const Tensor*>& inputs, std::vector<const Tensor*>& outputs) {
  ET_CHECK_OR_RETURN_ERROR(inputs.size() == getNumInputs(), Internal, "Inputs mismatch");
  ET_CHECK_OR_RETURN_ERROR(outputs.size() == getNumOutputs(), Internal, "Outputs mismatch");
  // updateDataBuffers is a no-op for devices with shared memory.
  // In case of devices with non-shared memory, it will blit the contents to a private GPU buffer.
  updateDataBuffers(inputs, outputs);
  for (MPSGraphTensor *tensor in [_executable feedTensors]) {
    int i = _mpsGraphTensorToId[tensor];
    MPSGraphTensorData* tensorData = [[[MPSGraphTensorData alloc]initWithMTLBuffer:_inputGPUBuffers[i]
                                                                            shape:[_inputShapes[i] shape]
                                                                          dataType:[_inputShapes[i] dataType]] autorelease];
    _inputsArray[i] = tensorData;
  }

  for (int i = 0; i < outputs.size(); i++) {
    MPSGraphTensorData* tensorData = [[[MPSGraphTensorData alloc] initWithMTLBuffer:_outputGPUBuffers[i]
                                                                              shape:[_outputShapes[i] shape]
                                                                          dataType:[_outputShapes[i] dataType]] autorelease];
    _outputsArray[i] = tensorData;
  }
  return Error::Ok;
}

__ET_NODISCARD Error MPSExecutor::forward(std::vector<const Tensor*>& outputs) {
  Error err = Error::Ok;
  MPSStream* mpsStream = getDefaultMPSStream();
  if (mpsStream->commitAndContinueEnabled() || mpsStream->hasLiveCommandBuffer()) {
    id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
    [_executable encodeToCommandBuffer:commandBuffer
                          inputsArray:_inputsArray
                          resultsArray:_outputsArray
                  executionDescriptor:nil];
  } else {
    [_executable runWithMTLCommandQueue:mpsStream->commandQueue()
                            inputsArray:_inputsArray
                           resultsArray:_outputsArray
                    executionDescriptor:nil];
  }
  syncOutputBuffers(outputs);

  // On simulator, the buffers are synchronized during `syncOutputBuffer`
#if !TARGET_OS_SIMULATOR
  if (mpsStream->commitAndContinueEnabled()) {
    err = mpsStream->synchronize(SyncType::COMMIT_AND_CONTINUE);
  } else {
    err = mpsStream->synchronize(SyncType::COMMIT_AND_WAIT);
  }

  ET_CHECK_OR_RETURN_ERROR(
    err == Error::Ok,
    Internal,
    "Could not synchronize on the MPSStream");
#endif

  return Error::Ok;
}

Error
MPSExecutor::initDataBuffers() {
  Error error = Error::Ok;

  _inputShapes = [[_executable getInputShapes] retain];
  _outputShapes = [[_executable getOutputShapes] retain];

  int nInputs = getNumInputs();
  int nOutputs = getNumOutputs();

  _inputGPUBuffers.resize(nInputs);
  _outputGPUBuffers.resize(nOutputs);

  if (!_use_shared_mem) {
    _inputCPUBuffers.resize(nInputs);
    _outputCPUBuffers.resize(nOutputs);
  }

  // In case of shared memory, the CPU raw buffer is used directly as an MTLBuffer.
  // In case of not being able to use shared memory, initialize the data buffers once
  // and keep reusing them across inference runs.
  auto getDataBuffer = [] (MPSShape* shape, MPSDataType mpsDataType) {
    __block int64_t length = 1;
    [shape enumerateObjectsUsingBlock:^(NSNumber * _Nonnull obj, NSUInteger idx, BOOL * _Nonnull stop) {
        length *= obj.intValue;
    }];
    // Get total size in bytes.
    length *= ((mpsDataType & 0xFFFF) >> 3);
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared;
    return [MPSDevice::getInstance()->device() newBufferWithLength:length
                                                            options:options];

  };

  // Preallocate at init time the GPU buffers used to run
  // the model in case shared memory is not being used.
  if (!_use_shared_mem) {
    for (int i = 0; i < nInputs; i++) {
      _inputGPUBuffers[i] = getDataBuffer([_inputShapes[i] shape], [_inputShapes[i] dataType]);
    }
    for (int i = 0; i < nOutputs; i++) {
      _outputGPUBuffers[i] = getDataBuffer([_outputShapes[i] shape], [_outputShapes[i] dataType]);
    }
  }

  return error;
}

Error
MPSExecutor::updateDataBuffers(
  std::vector<const Tensor*>& inputs, std::vector<const Tensor*>& outputs
) {
  for (int i = 0; i < inputs.size(); i++) {
    const Tensor& tensor = *inputs[i];
    void* host_src = tensor.mutable_data_ptr<void*>();
    if (_use_shared_mem) {
      // Use directly the CPU buffer when using shared memory.
      _inputGPUBuffers[i] = getMTLBufferStorage(tensor);
    } else {
      _inputCPUBuffers[i].flags = 0;
#if TARGET_OS_SIMULATOR
      // Simulator crashes when using newBufferWithBytesNoCopy.
      // Use memcpy directly instead of using blit to copy the CPU
      // data into the GPU buffer.
      _inputCPUBuffers[i].srcOffset = 0;
      _inputCPUBuffers[i].srcBuffer = host_src;
      _inputCPUBuffers[i].srcCpu = 1;
#else
      MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared;
      NSUInteger alignedLength = 0;
      void* alignedPtr = pageAlignedBlockPtr(host_src, (NSUInteger)tensor.nbytes(), &alignedLength);
      _inputCPUBuffers[i].srcOffset = uintptr_t(host_src) - uintptr_t(alignedPtr);
      _inputCPUBuffers[i].srcBuffer = [MPSDevice::getInstance()->device() newBufferWithBytesNoCopy:alignedPtr
                                                        length:alignedLength
                                                      options:options
                                                  deallocator:nil];

#endif
      _inputCPUBuffers[i].dstBuffer = _inputGPUBuffers[i];
      _inputCPUBuffers[i].dstOffset = 0;
      _inputCPUBuffers[i].length = tensor.nbytes();
    }
  }

  if (_use_shared_mem) {
    for (int i = 0; i < outputs.size(); i++) {
      _outputGPUBuffers[i] = getMTLBufferStorage(*outputs[i]);
    }
  }

  if (!_use_shared_mem) {
    MPSStream* mpsStream = getDefaultMPSStream();
      mpsStream->copy_and_sync(
        _inputCPUBuffers, /*non_blocking=*/true);
  }

  return Error::Ok;
}

Error
MPSExecutor::syncOutputBuffers(
  std::vector<const Tensor*>& outputs) {
  if (!_use_shared_mem)  {
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared;
    NSUInteger alignedLength = 0;
    MPSStream* mpsStream = getDefaultMPSStream();

  if (!_buffers_initialized) {
      for (int i = 0; i < outputs.size(); i++) {
        const Tensor& tensor = *outputs[i];
        void* host_dst = tensor.mutable_data_ptr<void*>();
        _outputCPUBuffers[i].flags = 0;
#if TARGET_OS_SIMULATOR
        _outputCPUBuffers[i].dstOffset = 0;
        _outputCPUBuffers[i].dstBuffer = host_dst;
        _outputCPUBuffers[i].dstCpu = 1;
#else
        void* alignedPtr = pageAlignedBlockPtr(host_dst, (NSUInteger)tensor.nbytes(), &alignedLength);
        _outputCPUBuffers[i].dstOffset = (uintptr_t(host_dst) - uintptr_t(alignedPtr));
        // 4 bytes alignment required on MacOS for blits.
        ET_CHECK_MSG(_outputCPUBuffers[i].dstOffset % 4 == 0, "Unaligned blit request");
        _outputCPUBuffers[i].dstBuffer = [MPSDevice::getInstance()->device() newBufferWithBytesNoCopy:alignedPtr
                                                              length:alignedLength
                                                            options:options
                                                        deallocator:nil];
#endif
        _outputCPUBuffers[i].srcBuffer = _outputGPUBuffers[i];
        _outputCPUBuffers[i].srcOffset = 0;
        _outputCPUBuffers[i].length = tensor.nbytes();
      }
    }

    mpsStream->copy_and_sync(
      _outputCPUBuffers, /*non_blocking=*/true
    );
  }

  _buffers_initialized = true;
  return Error::Ok;
}


} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
