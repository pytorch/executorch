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

@implementation ExecuTorchModule {
  std::unique_ptr<executorch::extension::Module> _module;
}

@end
