/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Foundation/Foundation.h>
#include <executorch/runtime/core/result.h>
#include "coreai_file.h"

namespace executorch::backends::coreai {

// Resolves the default location without creating or modifying storage.
runtime::Result<NSString*> default_coreai_assets_root();
// Requires an absolute directory path. With Create, makes the directory and
// excludes it from backup; otherwise a missing directory is InvalidArgument.
runtime::Result<NSString*> prepare_storage_root(
    NSString* path,
    StorageCreationMode mode);

} // namespace executorch::backends::coreai
