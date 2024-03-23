//
// ETCoreMLAsset.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

#import <asset.h>

NS_ASSUME_NONNULL_BEGIN

/// Represents an asset on the filesystem.
@interface ETCoreMLAsset : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

/// Constructs an `ETCoreMLAsset` instance from `ModelAsset`
///
/// @param backingAsset The cpp asset.
- (instancetype)initWithBackingAsset:(executorchcoreml::Asset)backingAsset NS_DESIGNATED_INITIALIZER;

/// The unique identifier.
@property (copy, readonly, nonatomic) NSString* identifier;

/// The absolute URL of the asset.
@property (copy, readonly, nonatomic) NSURL* contentURL;

/// The total size of the directory in bytes.
@property (assign, readonly, nonatomic) NSUInteger totalSizeInBytes;

/// Returns `YES` is the asset is valid otherwise `NO`.
@property (assign, readonly, nonatomic) BOOL isValid;

/// Returns `YES` is the asset is alive otherwise `NO`.
@property (assign, readonly, nonatomic) BOOL isAlive;

/// Keeps the asset alive by opening and hanging on to the file handles in the asset.
/// The file handles are closed when `close` is called or at the time of deallocation.
///
/// @param error   On failure, error is filled with the failure information.
/// @retval `YES` is the file handles could be opened otherwise `NO`.
- (BOOL)keepAliveAndReturnError:(NSError* __autoreleasing*)error;

/// Pre-warms the asset by doing an advisory async read to all the content files.
///
/// @param error   On failure, error is filled with the failure information.
/// @retval `YES` is the advisory read to all the files succeeded otherwise`NO`.
- (BOOL)prewarmAndReturnError:(NSError* __autoreleasing*)error;

/// Closes all the file handles opened by the `keepAliveAndReturnError` call.
- (void)close;

@end

NS_ASSUME_NONNULL_END
