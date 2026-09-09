/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchDump.h"

#import "ExecuTorchDumpError.h"

#import <executorch/runtime/core/event_tracer_hooks.h>

NSErrorDomain const ExecuTorchDumpErrorDomain =
    @"org.pytorch.executorch.etdump";

@implementation ExecuTorchDump {
  ExecuTorchModule *_module;
  ExecuTorchDumpTracer *_tracer;
}

+ (BOOL)isAvailable {
  // Reports what this framework was compiled with, which is the same build as
  // the runtime it ships beside. A caller linking a runtime built separately
  // without tracing gets a truthful answer at read time instead, where an empty
  // trace is reported rather than silently returned.
  return executorch::runtime::internal::event_tracer_enabled();
}

- (nullable instancetype)initWithFilePath:(NSString *)filePath
                            dataFilePaths:(NSArray<NSString *> *)dataFilePaths
                                 loadMode:(ExecuTorchModuleLoadMode)loadMode
                                    error:(NSError **)error {
  if (!ExecuTorchDump.isAvailable) {
    if (error) {
      *error = [NSError
          errorWithDomain:ExecuTorchDumpErrorDomain
                     code:ExecuTorchDumpErrorCodeUnavailable
                 userInfo:@{
                   NSLocalizedDescriptionKey :
                       @"This ExecuTorch runtime was built without event "
                       @"tracing, so no profile can be recorded."
                 }];
    }
    return nil;
  }
  self = [super init];
  if (self) {
    _tracer = [[ExecuTorchDumpTracer alloc] init];
    _module = [[ExecuTorchModule alloc] initWithFilePath:filePath
                                           dataFilePaths:dataFilePaths
                                                loadMode:loadMode
                                             eventTracer:_tracer];
  }
  return self;
}

- (nullable instancetype)initWithFilePath:(NSString *)filePath
                                    error:(NSError **)error {
  return [self initWithFilePath:filePath
                  dataFilePaths:@[]
                       loadMode:ExecuTorchModuleLoadModeFile
                          error:error];
}

- (ExecuTorchModule *)module {
  return _module;
}

- (nullable NSData *)takeDataWithError:(NSError **)error {
  return [_tracer takeDataWithError:error];
}

- (BOOL)takeDataToFile:(NSString *)path error:(NSError **)error {
  return [_tracer takeDataToFile:path error:error];
}

@end
