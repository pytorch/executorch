//
//  objc_safe_cast.h
//  util
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#pragma once

#import <Foundation/Foundation.h>

inline id check_class(id obj, Class cls) { return [obj isKindOfClass:cls] ? obj : nil; }

#define SAFE_CAST(Object, Type) ((Type*)check_class(Object, [Type class]))
