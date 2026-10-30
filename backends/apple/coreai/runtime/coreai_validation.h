/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Foundation/Foundation.h>

namespace executorch::backends::coreai {

// Nonempty text without embedded NULs.
bool is_nonempty_string(id value);
// Relative bundle path without escapes, backslashes, or empty components.
bool is_relative_path(NSString* path);
// Single path component for an fd-relative open: no slashes, ".", or "..".
// Unlike is_relative_path, backslashes are ordinary bytes on this filesystem.
bool is_storage_component(NSString* name);
// Lowercase hex of any positive length, as in the export hash.
bool is_lowercase_hex(NSString* value);
// Lowercase hex of exactly length characters, as in bundle digests.
bool is_lowercase_hex(NSString* value, NSUInteger length);

} // namespace executorch::backends::coreai
