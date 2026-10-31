/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "coreai_validation.h"

namespace executorch::backends::coreai {
namespace {

NSCharacterSet* invalid_hex_characters() {
  static NSCharacterSet* invalid = [[NSCharacterSet
      characterSetWithCharactersInString:@"0123456789abcdef"] invertedSet];
  return invalid;
}

} // namespace

bool is_nonempty_string(id value) {
  return [value isKindOfClass:NSString.class] && [value length] > 0 &&
      [value rangeOfString:@"\0"].location == NSNotFound;
}

bool is_relative_path(NSString* path) {
  if (!is_nonempty_string(path) || path.isAbsolutePath ||
      [path containsString:@"\\"]) {
    return false;
  }
  for (NSString* component in [path componentsSeparatedByString:@"/"]) {
    if (component.length == 0 || [component isEqualToString:@"."] ||
        [component isEqualToString:@".."]) {
      return false;
    }
  }
  return true;
}

bool is_storage_component(NSString* name) {
  return is_nonempty_string(name) && ![name isEqualToString:@"."] &&
      ![name isEqualToString:@".."] && ![name containsString:@"/"];
}

bool is_lowercase_hex(NSString* value) {
  return is_nonempty_string(value) &&
      [value rangeOfCharacterFromSet:invalid_hex_characters()].location ==
      NSNotFound;
}

bool is_lowercase_hex(NSString* value, NSUInteger length) {
  return is_lowercase_hex(value) && [value length] == length;
}

} // namespace executorch::backends::coreai
