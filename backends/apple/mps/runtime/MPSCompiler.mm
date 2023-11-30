//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "MPSCompiler.h"
#include <executorch/backends/apple/mps/utils/MPSGraphPackageExport.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <unordered_map>
#include <string>

#define MPS_UNUSED(x) ( (void)(x) )

@interface MPSGraphExecutable()
-(NSArray<MPSGraphShapedType *> *) getInputShapes;
-(NSArray<MPSGraphShapedType *> *) getOutputShapes;
@end

namespace torch {
namespace executor {
namespace mps {
namespace delegate {

void printLoadedGraph(MPSGraphExecutable* executable) {
  NSLog(@"Loaded graph: %@", [executable debugDescription]);
}

MPSGraphExecutable* loadExecutable(
  const void* buffer_pointer,
  size_t num_bytes) {
  ExirMPSGraphPackage* exirMPSGraphPackage = (ExirMPSGraphPackage*)buffer_pointer;
  NSData *new_manifest_plist_data = [NSData dataWithBytes:exirMPSGraphPackage->data length:exirMPSGraphPackage->model_0_offset];
  NSData *new_model_0_data = [NSData dataWithBytes:exirMPSGraphPackage->data + exirMPSGraphPackage->model_0_offset length:exirMPSGraphPackage->total_bytes - sizeof(ExirMPSGraphPackage) - exirMPSGraphPackage->model_0_offset];

  NSError* error = nil;
  NSString* packageName = [NSString stringWithUTF8String:(
      std::string("%@/mpsgraphmodule_") + std::to_string(arc4random_uniform(INT_MAX)) + ".mpsgraphpackage").c_str()];
#if TARGET_OS_IPHONE
  NSArray *paths = NSSearchPathForDirectoriesInDomains
      (NSDocumentDirectory, NSUserDomainMask, YES);
  NSString *documentsDirectory = [paths objectAtIndex:0];
#else
  NSString *documentsDirectory = @"/tmp";
#endif

  NSString *dataFileNSStr = [NSString stringWithFormat:packageName,
                                                        documentsDirectory];

  NSString* manifestFileStr = [NSString stringWithFormat:@"%@/manifest.plist", dataFileNSStr];
  NSString* model0FileStr = [NSString stringWithFormat:@"%@/model_0.mpsgraph", dataFileNSStr];

  NSFileManager *fileManager= [NSFileManager defaultManager];
  [fileManager createDirectoryAtPath:dataFileNSStr withIntermediateDirectories:NO attributes:nil error:&error];

  [new_manifest_plist_data writeToFile:manifestFileStr options:NSDataWritingAtomic error:&error];
  [new_model_0_data writeToFile:model0FileStr options:NSDataWritingAtomic error:&error];

  NSURL *bundleURL = [NSURL fileURLWithPath:dataFileNSStr];
  MPSGraphCompilationDescriptor *compilationDescriptor = [MPSGraphCompilationDescriptor new];
  MPSGraphExecutable *newExec = [[MPSGraphExecutable new] initWithMPSGraphPackageAtURL:bundleURL compilationDescriptor:compilationDescriptor];

  return newExec;
}

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

  id mpsCD = NSClassFromString(@"MPSGraph");
  static bool _macos_14_0_plus = [mpsCD instancesRespondToSelector:@selector(imToColWithSourceTensor:descriptor:name:)] == YES;
  ET_CHECK_OR_RETURN_ERROR(
      _macos_14_0_plus,
      NotSupported,
      "MPS Executorch runtime is supported only from macOS 14.0 and above.");

  MPSGraphExecutable* executable = loadExecutable(buffer_pointer, num_bytes);
  ET_CHECK_OR_RETURN_ERROR(
      executable != nil,
      InvalidProgram,
      "Invalid flatbuffer contents - could not deserialize MPSGraphExecutable");

  executor->inputShapes_ = [[executable getInputShapes] retain];
  executor->outputShapes_ = [[executable getOutputShapes] retain];

  ET_LOG(Info, "Num inputs: %lu", [executor->inputShapes_ count]);
  ET_LOG(Info, "Num outputs: %lu", [executor->outputShapes_ count]);

  executor->executable_ = executable;
  return err;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
