//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/backends/apple/mps/runtime/MPSStream.h>
#include <executorch/runtime/platform/assert.h>
#include <vector>

@interface MPSGraphExecutionDescriptor ()
@property (readwrite, atomic) BOOL enableCommitAndContinue;
@end

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

MPSStream::MPSStream() {
  _commandQueue = [MPSDevice::getInstance()->device() newCommandQueue];
  _serialQueue = dispatch_queue_create("metal gpu stream", nullptr);
}

MPSStream::~MPSStream() {
  [_commandQueue release];
  _commandQueue = nil;

  assert(_commandBuffer == nil);
}

bool MPSStream::hasLiveCommandBuffer() {
  return _commandBuffer;
}

API_AVAILABLE(ios(13.0))
MPSCommandBuffer* MPSStream::commandBuffer() {
  if (!_commandBuffer) {
    _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;
  }

  return _commandBuffer;
}

id<MTLComputeCommandEncoder> MPSStream::commandEncoder() {
  if (!_commandEncoder) {
    if (@available(iOS 13.0, *)) {
      _commandEncoder = [commandBuffer() computeCommandEncoder].retain;
    }
  }

  return _commandEncoder;
}

ET_NODISCARD
Error MPSStream::synchronize(SyncType syncType) {
  endKernelCoalescing();
  switch(syncType) {
    case SyncType::COMMIT:
      commit();
      break;
    case SyncType::COMMIT_AND_WAIT:
      commitAndWait();
      break;
    case SyncType::COMMIT_ADAPTIVE:
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
    if (@available(iOS 13.0, *)) {
      [commandBuffer() commitAndContinue];
    }
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

void MPSStream::copy(id<MTLBuffer> srcBuffer,
                     id<MTLBuffer> dstBuffer,
                     size_t length,
                     size_t srcOffset,
                     size_t dstOffset,
                     SyncType syncType) {
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      if (@available(iOS 13.0, *)) {
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

        [blitEncoder copyFromBuffer:srcBuffer
                       sourceOffset:(NSUInteger)srcOffset
                           toBuffer:dstBuffer
                  destinationOffset:(NSUInteger)dstOffset
                               size:(NSUInteger)length];
        [blitEncoder endEncoding];
      }
      ET_CHECK(synchronize(syncType) == Error::Ok);
    }
  });
}

void MPSStream::copy_and_sync(id<MTLBuffer> srcBuffer,
                              id<MTLBuffer> dstBuffer,
                              size_t length,
                              size_t srcOffset,
                              size_t dstOffset,
                              bool non_blocking) {
  copy(srcBuffer,
       dstBuffer,
       length,
       srcOffset,
       dstOffset,
       !non_blocking ? SyncType::COMMIT_AND_WAIT : SyncType::COMMIT_ADAPTIVE);
}

void MPSStream::copy(std::vector<CPUBufferWrapper>& dataBuffers,
                     SyncType syncType) {
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
#if TARGET_OS_SIMULATOR
      if (dataBuffers[0].dstCpu) {
        // If the destination is a CPU buffer,
        // wait for the GPU to finish executing
        // before copying into the CPU buffers.
        ET_CHECK(synchronize(SyncType::COMMIT_AND_WAIT) == Error::Ok);
      }
      for (int i = 0; i < dataBuffers.size(); i++) {
        uint8_t* src = nil;
        uint8_t* dst = nil;
        if (dataBuffers[i].srcCpu) {
          src = static_cast<uint8_t*>(dataBuffers[i].srcBuffer) + dataBuffers[i].srcOffset;
          dst = (uint8_t*)([(id<MTLBuffer>)dataBuffers[i].dstBuffer contents]) + dataBuffers[i].dstOffset;
        } else {
          ET_CHECK(dataBuffers[i].dstCpu);
          src = (uint8_t*)([(id<MTLBuffer>)dataBuffers[i].srcBuffer contents]) + dataBuffers[i].srcOffset;
          dst = static_cast<uint8_t*>(dataBuffers[i].dstBuffer) + dataBuffers[i].dstOffset;
        }
        memcpy(dst, src, dataBuffers[i].length);
      }
#else
      endKernelCoalescing();
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

      for (int i = 0; i < dataBuffers.size(); i++) {
        [blitEncoder copyFromBuffer:(id<MTLBuffer>)dataBuffers[i].srcBuffer
                       sourceOffset:(NSUInteger)dataBuffers[i].srcOffset
                           toBuffer:(id<MTLBuffer>)dataBuffers[i].dstBuffer
                  destinationOffset:(NSUInteger)dataBuffers[i].dstOffset
                               size:(NSUInteger)dataBuffers[i].length];
      }
      [blitEncoder endEncoding];
      ET_CHECK(synchronize(syncType) == Error::Ok);
#endif
    }
  });
}

void MPSStream::copy_and_sync(std::vector<CPUBufferWrapper>& dataBuffers,
                              bool non_blocking) {
  copy(dataBuffers,
       !non_blocking ? SyncType::COMMIT_AND_WAIT : SyncType::COMMIT_ADAPTIVE);
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
