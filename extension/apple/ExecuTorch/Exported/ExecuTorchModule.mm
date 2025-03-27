/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "ExecuTorchModule.h"

#import "ExecuTorchError.h"

#import <executorch/extension/module/module.h>
#import <executorch/extension/tensor/tensor.h>

using namespace executorch::extension;

@implementation ExecuTorchModule {
  std::unique_ptr<Module> _module;
}

- (instancetype)initWithFilePath:(NSString *)filePath
                        loadMode:(ExecuTorchModuleLoadMode)loadMode {
  self = [super init];
  if (self) {
    _module = std::make_unique<Module>(
      filePath.UTF8String,
      static_cast<Module::LoadMode>(loadMode)
    );
  }
  return self;
}

- (instancetype)initWithFilePath:(NSString *)filePath {
  return [self initWithFilePath:filePath loadMode:ExecuTorchModuleLoadModeFile];
}

@end
