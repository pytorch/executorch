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

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

/*
Builds the mps runtime object using the buffer pointer. The buffer pointer
must be a valid pointer to the serialized mps object.
*/
ET_NODISCARD Error MPSCompiler::compileModel(
  const void* buffer_pointer,
  size_t num_bytes,
  MPSExecutor* executor,
  MemoryAllocator* runtime_allocator,
  ArrayRef<CompileSpec> compile_specs) {
  MPS_UNUSED(compile_specs);

  Error err = Error::Ok;

  std::unique_ptr<MPSGraphBuilder> mpsGraphBuilder(
    new MPSGraphBuilder(buffer_pointer, num_bytes, executor->_mpsGraphTensorToId));
  err = mpsGraphBuilder->compileModel();
  ET_CHECK_OR_RETURN_ERROR(
    err == Error::Ok, Internal, "Failed to construct the MPS graph object");

  executor->_executable = mpsGraphBuilder->getMPSGraphExecutable();
  ET_CHECK_OR_RETURN_ERROR(
      executor->_executable != nil,
      InvalidProgram,
      "Invalid FlatBuffer contents - could not create MPSGraphExecutable");

  err = executor->initDataBuffers();
  ET_CHECK_OR_RETURN_ERROR(
      err == Error::Ok, Internal, "Could not allocate data buffers");

  ET_LOG(Debug, "MPSGraphExecutable total inputs: %lu", [executor->_inputShapes count]);
  ET_LOG(Debug, "MPSGraphExecutable total outputs: %lu", [executor->_outputShapes count]);

  return err;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
