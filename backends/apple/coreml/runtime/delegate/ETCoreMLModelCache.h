//
// ETCoreMLModelCache.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

#import "ETCoreMLCacheProtocol.h"

NS_ASSUME_NONNULL_BEGIN

extern NSString* const ETCoreMLModelCacheErrorDomain;

typedef NS_ENUM(NSInteger, ETCoreMLModelCacheErrorCode) {
    ETCoreMLModelCacheErrorCodeUnknown = 0,
    ETCoreMLModelCacheErrorCodeInitializationFailed = 1,
    ETCoreMLModelCacheErrorCodeInvalidIdentifier = 2,
    ETCoreMLModelCacheErrorCodeSourceNotFound = 3,
    ETCoreMLModelCacheErrorCodeDiskFull = 4,
    ETCoreMLModelCacheErrorCodeIOError = 5,
    ETCoreMLModelCacheErrorCodeCorruptedCache = 6,
};

/// A simplified, filesystem-based cache for compiled CoreML models.
///
/// This class provides a cache implementation that stores compiled models as directories
/// in a versioned cache structure. It uses atomic writes (rename) to ensure cache integrity
/// even in the presence of crashes or concurrent access.
///
/// Directory structure:
/// ```
/// cache_root/
/// ├── version.txt                         (cache format version)
/// ├── models/
/// │   ├── {identifier}.mlmodelc/          (compiled model bundle)
/// │   ├── {identifier}.accessed           (last access time for LRU eviction)
/// │   └── ...
/// └── temp/
///     └── {uuid}/                         (mlpackage files awaiting compilation)
/// ```
///
/// ## Thread Safety and Concurrency Guarantees
///
/// This class provides **NO internal synchronization**. It is designed to be used in one of
/// two ways:
///
/// 1. **Single-threaded access**: All calls to a single instance from one thread/queue.
///
/// 2. **External serialization**: When used via `ETCoreMLModelManager`, access is serialized
///    by the manager's per-identifier loading queue. This is the expected usage pattern.
///
/// **Multi-process safety** is provided by:
/// - Atomic filesystem operations (`rename()`)
/// - Unique temp paths (UUID-based) to avoid conflicts
/// - "Last writer wins" semantics (acceptable since all writers produce identical output)
///
/// **Multiple instances** pointing to the same directory are safe because:
/// - Each write uses a unique temp path
/// - Final placement uses atomic `moveItemAtURL:` (POSIX `rename()`)
/// - Concurrent writes result in "last writer wins" (both write identical data)
/// - Cleanup only targets entries older than 24 hours
///
/// **Callers are responsible for**:
/// - Handling `MLModel` load failures gracefully (cache entry may be replaced/deleted
///   between URL retrieval and model load)
/// - Not relying on returned URLs remaining valid indefinitely
@interface ETCoreMLModelCache : NSObject <ETCoreMLCache>

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

/// The root directory for all cache data (contains models/, temp/, version.txt).
@property (nonatomic, readonly) NSURL* cacheRootDirectory;

/// Whether the cache was initialized successfully and is ready for use.
/// If NO, all operations will fail. Check this after initialization.
@property (nonatomic, readonly, getter=isReady) BOOL ready;

/// If `ready` is NO, this contains the error that occurred during initialization.
@property (nonatomic, readonly, nullable) NSError* initializationError;

/// Initializes the cache with the given root directory.
/// Creates the directory structure if it doesn't exist.
/// Check the `ready` property after initialization to verify success.
/// If initialization fails, `initializationError` will contain the reason.
///
/// @param cacheRootDirectory The root directory for all cache data.
- (instancetype)initWithCacheRootDirectory:(NSURL*)cacheRootDirectory NS_DESIGNATED_INITIALIZER;

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
/// @param identifier The unique identifier for this model. Must not contain '/' or '..'.
/// @param error On failure, contains the error. Check for ETCoreMLModelCacheErrorCodeDiskFull
///              to handle out-of-space conditions specially.
/// @return The URL of the cached model, or nil on failure.
- (nullable NSURL*)storeModelAtURL:(NSURL*)compiledModelURL withIdentifier:(NSString*)identifier error:(NSError**)error;

/// Removes a specific cached model. This is a best-effort operation that removes
/// the model bundle and access time files for the given identifier.
///
/// @param identifier The unique identifier for the cached model to remove.
/// @param error On failure, error is filled with the failure information.
/// @return YES on success (including if the model didn't exist), NO on validation errors.
- (BOOL)removeCachedModelWithIdentifier:(NSString*)identifier error:(NSError**)error;

/// Clears the entire cache, including all cached models.
/// Recreates the empty directory structure after clearing.
///
/// @param error On failure, error is filled with the failure information.
/// @return YES if the cache was purged successfully, otherwise NO.
- (BOOL)purgeAndReturnError:(NSError**)error;

#pragma mark - Temp Directory (for mlpackage extraction before compilation)

/// Returns a temp URL where an mlpackage can be extracted before compilation.
/// The caller is responsible for cleaning up this directory after compilation completes.
///
/// @param error On failure, error is filled with the failure information.
/// @return A temp URL where the mlpackage can be extracted, or nil on failure.
///
/// @note The temp URL is unique and includes a UUID to avoid conflicts.
/// @note Temp entries are automatically cleaned up after 24 hours if not removed.
- (nullable NSURL*)temporaryDirectoryWithError:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
