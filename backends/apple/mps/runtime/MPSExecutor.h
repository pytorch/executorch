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
  MPSGraphExecutable* executable_;
  NSArray<MPSGraphShapedType *> * inputShapes_;
  NSArray<MPSGraphShapedType *> * outputShapes_;

  NSMutableArray<MPSGraphTensorData *>* inputsArray_;
  NSMutableArray<MPSGraphTensorData *>* outputsArray_;
#if TARGET_OS_SIMULATOR
  NSMutableArray<id<MTLBuffer>>* output_buffers_;
#endif

 public:
  MPSExecutor() = default;
  ~MPSExecutor() {
    [inputsArray_ release];
    [outputsArray_ release];
  }

  inline size_t getNumInputs() {
    return [inputShapes_ count];
  }

  inline size_t getNumOutputs() {
    return [outputShapes_ count];
  }

  inline MPSGraphExecutable* getMPSGraphExecutable() {
    return executable_;
  }

  __ET_NODISCARD Error forward(std::vector<const Tensor*>& outputs);

  __ET_NODISCARD Error
  set_inputs_outputs(std::vector<const Tensor*>& inputs, std::vector<const Tensor*>& outputs);

  friend class MPSCompiler;
};

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
// clang-format on
