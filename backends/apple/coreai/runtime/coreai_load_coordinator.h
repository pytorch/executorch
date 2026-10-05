/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#import "ETCoreAIBridge.h"
#import "coreai_bookmarks.h"

namespace executorch::backends::coreai {

// The key lock outlives all SDK callbacks and raw bookmark publication.
runtime::Result<id<ETCoreAIPreparedModel>> acquire_bookmark_model(
    const Manifest& selected,
    const runtime::NamedDataMap* named_data,
    NSString* root,
    NSString* platform,
    NSString* architecture,
    id<ETCoreAIModelLoading> loader);

} // namespace executorch::backends::coreai
