//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include "MPSStream.h"
#include <executorch/runtime/platform/assert.h>

@interface MPSGraphExecutionDescriptor ()
@property (readwrite, atomic) BOOL enableCommitAndContinue;
@end

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

// threshold to perform adaptive commit if the accumulated size
// of resources encoded on the command buffer exceeds that.
static const size_t kCmdBufAdaptiveCommitThreshold = MB(64);

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

MPSStream::MPSStream() {
  _commandQueue = [MPSDevice::getInstance()->device() newCommandQueue];
  _serialQueue = dispatch_queue_create("metal gpu stream", nullptr);
  _executionDescriptor = [MPSGraphExecutionDescriptor new];
  _executableExecutionDescriptor = [MPSGraphExecutableExecutionDescriptor new];
  _compilationDescriptor = [MPSGraphCompilationDescriptor new];

  // internal CommitAndContinue heuristic of MPSGraph is disabled, and we
  // control it via Adaptive Commit in Executorch-side
  _executionDescriptor.enableCommitAndContinue = false;

  // Choose level which optimizes for GPU
  _compilationDescriptor.optimizationLevel = MPSGraphOptimizationLevel0;
  _executionDescriptor.compilationDescriptor =  _compilationDescriptor;
}

MPSStream::~MPSStream() {
  [_commandQueue release];
  _commandQueue = nil;
  [_executionDescriptor release];
  [_compilationDescriptor release];
  [_executableExecutionDescriptor release];

  _executionDescriptor = nil;
  _compilationDescriptor = nil;
  _executableExecutionDescriptor = nil;

  assert(_commandBuffer == nil);
}

bool MPSStream::hasLiveCommandBuffer() {
  return _commandBuffer;
}

MPSCommandBuffer* MPSStream::commandBuffer() {
  if (!_commandBuffer) {
    _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;
  }

  return _commandBuffer;
}

id<MTLComputeCommandEncoder> MPSStream::commandEncoder() {
  if (!_commandEncoder) {
    _commandEncoder = [commandBuffer() computeCommandEncoder].retain;
  }

  return _commandEncoder;
}

__ET_NODISCARD
Error MPSStream::synchronize(SyncType syncType) {
  endKernelCoalescing();
  switch(syncType) {
    case SyncType::COMMIT:
      commit();
      break;
    case SyncType::COMMIT_AND_WAIT:
      commitAndWait();
      break;
    case SyncType::COMMIT_AND_CONTINUE:
      ET_CHECK_OR_RETURN_ERROR(
        _enableCommitAndContinue == true,
        Internal,
        "CommitAndContinue is called but it is disabled globally!");
      commitAndContinue();
      break;
    default:
      ET_CHECK_OR_RETURN_ERROR(
        false,
        Internal,
        "Unhandled syncType type");
  }

  return Error::Ok;
}

bool MPSStream::commitAndContinueEnabled() {
  return _enableCommitAndContinue;
}

void MPSStream::commitAndContinue() {
  assert(_commandBuffer);
  [_commandBuffer commitAndContinue];
}

void MPSStream::endKernelCoalescing() {
  if (_commandEncoder) {
    [_commandEncoder endEncoding];
    [_commandEncoder release];
    _commandEncoder = nil;
  }
}

void MPSStream::commitAndWait() {
  if (_prevCommandBuffer) {
    // the previous command buffer (if exists) has already been committed,
    // so we just wait until it's completed and then dispose it.
    [_prevCommandBuffer waitUntilCompleted];
    [_prevCommandBuffer release];
    _prevCommandBuffer = nil;
  }

  if (_commandBuffer) {
    [_commandBuffer commit];
    [_commandBuffer waitUntilCompleted];
    [_commandBuffer release];
    _commandBuffer = nil;
    // reset the accumulated resource sizes for command buffer
    _commandBufferResourceSize = 0;
  }
}

void MPSStream::commit() {
  if (_enableCommitAndContinue) {
    [commandBuffer() commitAndContinue];
  } else {
    flush();
  }
  // reset the accumulated resource sizes for command buffer
  _commandBufferResourceSize = 0;
}

void MPSStream::flush() {
  if (_commandBuffer) {
    [_commandBuffer commit];
    // if commitAndContinue is disabled (e.g., for Profiler), we keep the command
    // buffer so we could wait on it later, if required.
    if (!_enableCommitAndContinue) {
      _prevCommandBuffer = _commandBuffer;
    } else {
      [_commandBuffer release];
    }
    _commandBuffer = nil;
  }
}

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

MPSStream* MPSStreamImpl::_stream = nullptr;

MPSStream* MPSStreamImpl::getInstance() {
  if (_stream == nullptr) {
    _stream =
        new MPSStream();
  }
  return _stream;
}

MPSStreamImpl::MPSStreamImpl() {}

MPSStream* getCurrentMPSStream() {
  return getDefaultMPSStream();
}

MPSStream* getDefaultMPSStream() {
  return MPSStreamImpl::getInstance();
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
