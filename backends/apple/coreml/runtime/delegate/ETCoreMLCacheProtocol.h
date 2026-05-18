//
// ETCoreMLCacheProtocol.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Protocol defining the interface for CoreML model caching.
///
/// This protocol abstracts the cache implementation
@protocol ETCoreMLCache <NSObject>

/// Returns the URL of a cached model if it exists and is valid, otherwise nil.
///
/// @param identifier The unique identifier for the cached model.
/// @param error On failure, error is filled with the failure information.
/// @return The URL to the cached model bundle, or nil if not found or invalid.
///
/// @warning The returned URL may become invalid before the caller uses it if another
/// process deletes or replaces the cached model. Callers MUST handle MLModel load
/// failures gracefully by treating them as cache misses and recompiling.
- (nullable NSURL*)cachedModelURLForIdentifier:(NSString*)identifier error:(NSError**)error;

/// Stores a compiled model in the cache. Returns the cached URL on success.
///
/// @param compiledModelURL The URL of the compiled model bundle to cache. Must exist.
/// @param identifier The unique identifier for this model.
/// @param error On failure, error is filled with the failure information.
/// @return The URL of the cached model, or nil on failure.
- (nullable NSURL*)storeModelAtURL:(NSURL*)compiledModelURL withIdentifier:(NSString*)identifier error:(NSError**)error;

/// Removes a specific cached model.
///
/// @param identifier The unique identifier for the cached model to remove.
/// @param error On failure, error is filled with the failure information.
/// @return YES if the model was removed or didn't exist. Returns NO only on I/O errors.
- (BOOL)removeCachedModelWithIdentifier:(NSString*)identifier error:(NSError**)error;

/// Clears the entire cache, including all cached models.
///
/// @param error On failure, error is filled with the failure information.
/// @return YES if the cache was purged successfully, otherwise NO.
- (BOOL)purgeAndReturnError:(NSError**)error;

/// Returns a temp URL where intermediate files can be written during compilation.
/// This is guaranteed to be on the same filesystem as the cache, ensuring atomic moves.
///
/// @param error On failure, error is filled with the failure information.
/// @return A temp URL where intermediate files can be written, or nil on failure.
///
/// @note The temp URL is unique (UUID-based) to avoid conflicts.
/// @note Temp entries are cleaned up automatically after 24 hours.
- (nullable NSURL*)temporaryDirectoryWithError:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
