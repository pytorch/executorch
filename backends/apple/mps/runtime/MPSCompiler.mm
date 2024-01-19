//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

// Obj-C headers
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// MPS headers
#include <executorch/backends/apple/mps/runtime/MPSDevice.h>
#include <executorch/backends/apple/mps/runtime/MPSCompiler.h>
#include <executorch/backends/apple/mps/runtime/MPSGraphBuilder.h>
#include <executorch/backends/apple/mps/schema_generated.h>

// Runtime headers
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <unordered_map>
#include <string>
#include <iostream>

#define MPS_UNUSED(x) ( (void)(x) )

@interface MPSGraphExecutable()
-(NSArray<MPSGraphShapedType *> *) getInputShapes;
-(NSArray<MPSGraphShapedType *> *) getOutputShapes;
@end

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

/*
Builds the mps runtime object using the buffer pointer. The buffer pointer
must be a valid pointer to the serialized mps object.
*/
__ET_NODISCARD Error MPSCompiler::compileModel(
  const void* buffer_pointer,
  size_t num_bytes,
  MPSExecutor* executor,
  MemoryAllocator* runtime_allocator,
  ArrayRef<CompileSpec> compile_specs) {
  MPS_UNUSED(compile_specs);

  Error err = Error::Ok;

  std::unique_ptr<MPSGraphBuilder> mpsGraphBuilder(new MPSGraphBuilder(buffer_pointer));
  err = mpsGraphBuilder->compileModel();
  ET_CHECK_OR_RETURN_ERROR(
    err == Error::Ok, Internal, "Failed to construct the MPS graph object");

  executor->executable_ = mpsGraphBuilder->getMPSGraphExecutable();
  ET_CHECK_OR_RETURN_ERROR(
      executor->executable_ != nil,
      InvalidProgram,
      "Invalid FlatBuffer contents - could not create MPSGraphExecutable");

  executor->inputShapes_ = [[executor->executable_ getInputShapes] retain];
  executor->outputShapes_ = [[executor->executable_ getOutputShapes] retain];

  ET_LOG(Debug, "MPSGraphExecutable num inputs: %lu", [executor->inputShapes_ count]);
  ET_LOG(Debug, "MPSGraphExecutable num outputs: %lu", [executor->outputShapes_ count]);

  return err;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
