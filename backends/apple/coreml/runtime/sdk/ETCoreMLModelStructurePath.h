//
// ETCoreModelStructurePath.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

#import "program_path.h"

NS_ASSUME_NONNULL_BEGIN
/// A class representing the path to a node in the model structure.
///
/// For a ML Program (ExecuTorch program), the structure is comprised of `Program`, `Function`, `Block`, and `Operation`
/// nodes. The path can refer to any node in the structure.
///
/// The class is a thin wrapper over `executorchcoreml::modelstructure::path`.
///
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLModelStructurePath : NSObject<NSCopying>

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

/// Constructs an `ETCoreMLModelStructurePath` instance.
///
/// @param value The cpp value.
- (instancetype)initWithUnderlyingValue:(executorchcoreml::modelstructure::Path)value NS_DESIGNATED_INITIALIZER;

/// Constructs an `ETCoreMLModelStructurePath` instance.
///
/// @param components The path components.`
- (instancetype)initWithComponents:(NSArray<NSDictionary<NSString*, id>*>*)components;

/// The underlying value.
@property (readonly, assign, nonatomic) executorchcoreml::modelstructure::Path underlyingValue;

/// If the path refers to an operation then it returns the operation's output name otherwise `nil`.
@property (readonly, copy, nonatomic, nullable) NSString* operationOutputName;

@end

NS_ASSUME_NONNULL_END
