//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/backends/apple/mps/schema_generated.h>
#include <executorch/backends/apple/mps/runtime/MPSExecutor.h>

@interface MPSNDArray ()
-(nonnull instancetype) initWithBuffer:(id<MTLBuffer> _Nonnull) buffer
                            descriptor:(MPSNDArrayDescriptor * _Nonnull) descriptor;
@end

@interface MPSNDArrayDescriptor ()
@property (readwrite, nonatomic) BOOL preferPackedRows;
@end


namespace torch {
namespace executor {
namespace mps {
namespace delegate {

__ET_NODISCARD Error
MPSExecutor::set_inputs_outputs(std::vector<const Tensor*>& inputs, std::vector<const Tensor*>& outputs) {
  ET_CHECK_OR_RETURN_ERROR(inputs.size() == getNumInputs(), Internal, "Inputs mismatch");
  ET_CHECK_OR_RETURN_ERROR(outputs.size() == getNumOutputs(), Internal, "Outputs mismatch");

#if !TARGET_OS_SIMULATOR
  if (outputsArray_ != nil) {
    return Error::Ok;
  }
#endif

  inputsArray_ = [[NSMutableArray<MPSGraphTensorData *> alloc] init];
  outputsArray_ = [[NSMutableArray<MPSGraphTensorData *> alloc] init];

#if TARGET_OS_SIMULATOR
  output_buffers_ = [[NSMutableArray<id<MTLBuffer>> alloc] init];
#endif

  for (int i = 0; i < inputs.size(); i++) {
    MPSNDArrayDescriptor *tensorDesc = [MPSNDArrayDescriptor descriptorWithDataType:[inputShapes_[i] dataType]
                                                                              shape:[inputShapes_[i] shape]];
    tensorDesc.preferPackedRows = YES;
    id<MTLBuffer> inputBuffer = getMTLBufferStorage(*inputs[i]);
    MPSNDArray *ndArrayData = [[MPSNDArray alloc] initWithBuffer:inputBuffer descriptor:tensorDesc];
    MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc] initWithMPSNDArray:ndArrayData];
    [inputsArray_ addObject:tensorData];
  }

  for (int i = 0; i < outputs.size(); i++) {
    MPSNDArrayDescriptor *tensorDesc = [MPSNDArrayDescriptor descriptorWithDataType:[outputShapes_[i] dataType]
                                                                              shape:[outputShapes_[i] shape]];
    tensorDesc.preferPackedRows = YES;
    id<MTLBuffer> outputBuffer = getMTLBufferStorage(*outputs[i]);
    MPSNDArray *ndArrayData = [[MPSNDArray alloc] initWithBuffer:outputBuffer descriptor:tensorDesc];
    MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc] initWithMPSNDArray:ndArrayData];
    [outputsArray_ addObject:tensorData];
#if TARGET_OS_SIMULATOR
    [output_buffers_ addObject:[outputBuffer retain]];
#endif
  }

  return Error::Ok;
}

__ET_NODISCARD Error MPSExecutor::forward(std::vector<const Tensor*>& outputs) {
  Error err = Error::Ok;
  MPSStream* mpsStream = getDefaultMPSStream();
  if (mpsStream->commitAndContinueEnabled() || mpsStream->hasLiveCommandBuffer() || true) {
    id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
    [executable_ encodeToCommandBuffer:commandBuffer
                          inputsArray:inputsArray_
                          resultsArray:outputsArray_
                  executionDescriptor:nil];
  } else {
    [executable_ runWithMTLCommandQueue:mpsStream->commandQueue()
                            inputsArray:inputsArray_
                           resultsArray:outputsArray_
                    executionDescriptor:nil];
  }

  if (mpsStream->commitAndContinueEnabled()) {
    err = mpsStream->synchronize(SyncType::COMMIT_AND_CONTINUE);
  } else {
    err = mpsStream->synchronize(SyncType::COMMIT_AND_WAIT);
#if TARGET_OS_SIMULATOR
  for (int i = 0; i < outputs.size(); i++) {
    uint8_t* data = outputs[i]->mutable_data_ptr<uint8_t>();
    memcpy(data, [output_buffers_[i] contents], [output_buffers_[i] length]);
  }
#endif
  }

  ET_CHECK_OR_RETURN_ERROR(
    err == Error::Ok,
    Internal,
    "Could not synchronize on the MPSStream");

  return Error::Ok;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
