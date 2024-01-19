//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <executorch/runtime/core/error.h>
#include <map>
#include "MPSDevice.h"

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

enum class SyncType {
  NONE, // no commit to command buffer
  COMMIT, // commit and flush the command buffer
  COMMIT_AND_WAIT, // flush and wait for command buffer execution to finish
  COMMIT_AND_CONTINUE, // commit and continue with a new underlying command
                       // buffer
  COMMIT_ADAPTIVE, // commit adaptively based on available memory
};

class MPSStream {
 public:
  MPSStream();

  ~MPSStream();
  id<MTLCommandQueue> commandQueue() const {
    return _commandQueue;
  };
  dispatch_queue_t queue() const {
    return _serialQueue;
  }

  bool hasLiveCommandBuffer();
  MPSCommandBuffer* commandBuffer();
  id<MTLComputeCommandEncoder> commandEncoder();
  void endKernelCoalescing();
  __ET_NODISCARD Error synchronize(SyncType syncType);
  // void commitAdaptive(const TensorList& inputTensors, const TensorList&
  // outputTensors, void* profilerHandle);
  bool commitAndContinueEnabled();

 private:
  id<MTLCommandQueue> _commandQueue = nil;
  MPSCommandBuffer* _commandBuffer = nil;
  MPSCommandBuffer* _prevCommandBuffer = nil;
  id<MTLComputeCommandEncoder> _commandEncoder = nil;
  MPSGraphExecutionDescriptor* _executionDescriptor = nil;
  MPSGraphExecutableExecutionDescriptor* _executableExecutionDescriptor = nil;
  MPSGraphCompilationDescriptor* _compilationDescriptor = nil;
  dispatch_queue_t _serialQueue = nullptr;
  // CommitAndContinue is disabled by default
  bool _enableCommitAndContinue = false;
  // accumulated sizes of resources encoded on command buffer
  size_t _commandBufferResourceSize = 0;
  // unfortunately, there's no way to get the underlying buffer from
  // an MPSGraphTensorData. so we need to keep a mapping of them here
  std::unordered_map<MPSGraphTensorData*, void*> _activeResources{};

  // use synchronize() to access any of these commit functions outside MPSStream
  void commit();
  void commitAndWait();
  void commitAndContinue();
  void flush();
};

/**
 * Get the current MPS stream
 */
MPSStream* getCurrentMPSStream();

/**
 * Get the default MPS stream
 */
MPSStream* getDefaultMPSStream();

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

class MPSStreamImpl {
 public:
  /**
   * Gets single instance of the MPSStream.
   */
  static MPSStream* getInstance();

 private:
  static MPSStream* _stream;
  MPSStreamImpl();
};

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
