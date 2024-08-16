//
// ETCoreMLModelDebugInfo.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

@class ETCoreMLModelStructurePath;

NS_ASSUME_NONNULL_BEGIN

/// A class representing the profiling info of an operation.
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLModelDebugInfo : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;


/// Constructs an `ETCoreMLModelDebugInfo` instance.
///
/// @param pathToDebugSymbolMap Operation path to debug symbol map.
/// @param pathToDebugHandlesMap Operation path to debug handles map.
- (instancetype)initWithPathToDebugSymbolMap:(NSDictionary<ETCoreMLModelStructurePath*, NSString*>*)pathToDebugSymbolMap
                       pathToDebugHandlesMap:
                           (NSDictionary<ETCoreMLModelStructurePath*, NSArray<NSString*>*>*)pathToDebugHandlesMap
    NS_DESIGNATED_INITIALIZER;

/// Constructs an `ETCoreMLModelDebugInfo` instance.
///
/// @param data The json data.
/// @param error   On failure, error is filled with the failure information.
+ (nullable instancetype)modelDebugInfoFromData:(NSData*)data error:(NSError* __autoreleasing*)error;

/// Operation path to debug symbol map.
@property (readonly, strong, nonatomic) NSDictionary<ETCoreMLModelStructurePath*, NSString*>* pathToDebugSymbolMap;

/// Operation path to debug handles map.
@property (readonly, strong, nonatomic)
    NSDictionary<ETCoreMLModelStructurePath*, NSArray<NSString*>*>* pathToDebugHandlesMap;

@end

NS_ASSUME_NONNULL_END
