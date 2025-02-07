//
// ETCoreMLAssetManager.h
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <Foundation/Foundation.h>

#import <database.hpp>

@class ETCoreMLAsset;

NS_ASSUME_NONNULL_BEGIN

/// A class responsible for managing the assets created by the delegate.
@interface ETCoreMLAssetManager : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

/// Constructs an `ETCoreMLAssetManager` instance.
///
/// @param database The sqlite database that will be used to keep track of assets.
/// @param assetsDirectoryURL The directory URL where the assets will be stored.
/// @param trashDirectoryURL   The directory URL where the assets will be moved before deletion.
/// @param maxAssetsSizeInBytes   The maximum assets size in bytes.
/// @param error   On failure, error is filled with the failure information.
- (nullable instancetype)initWithDatabase:(const std::shared_ptr<executorchcoreml::sqlite::Database>&)database
                       assetsDirectoryURL:(NSURL*)assetsDirectoryURL
                        trashDirectoryURL:(NSURL*)trashDirectoryURL
                     maxAssetsSizeInBytes:(NSInteger)maxAssetsSizeInBytes
                                    error:(NSError* __autoreleasing*)error NS_DESIGNATED_INITIALIZER;

/// Constructs an `ETCoreMLAssetManager` instance.
///
/// @param databaseURL The URL to the database that will be used to keep track of assets.
/// @param assetsDirectoryURL The directory URL where the assets will be stored.
/// @param trashDirectoryURL   The directory URL where the assets will be moved before deletion.
/// @param maxAssetsSizeInBytes   The maximum assets size in bytes.
/// @param error   On failure, error is filled with the failure information.
- (nullable instancetype)initWithDatabaseURL:(NSURL*)databaseURL
                          assetsDirectoryURL:(NSURL*)assetsDirectoryURL
                           trashDirectoryURL:(NSURL*)trashDirectoryURL
                        maxAssetsSizeInBytes:(NSInteger)maxAssetsSizeInBytes
                                       error:(NSError* __autoreleasing*)error;

/// Stores an asset at the url. The contents of the url are moved to the assets directory and it's
/// lifecycle is managed by the`ETCoreMLAssetManager`.
///
/// @param url The URL to the directory.
/// @param identifier An unique identifier to associate the asset with.
/// @param error   On failure, error is filled with the failure information.
/// @retval The `ETCoreMLAsset` instance representing the asset if the store was successful
/// otherwise `nil`.
- (nullable ETCoreMLAsset*)storeAssetAtURL:(NSURL*)url
                            withIdentifier:(NSString*)identifier
                                     error:(NSError* __autoreleasing*)error;

/// Returns the asset associated with the identifier.
///
/// @param identifier The asset identifier.
/// @param error   On failure, error is filled with the failure information.
/// @retval The `ETCoreMLAsset` instance representing the asset if the retrieval was successful
/// otherwise `nil`.
- (nullable ETCoreMLAsset*)assetWithIdentifier:(NSString*)identifier error:(NSError* __autoreleasing*)error;

/// Removes an asset associated with the identifier.
///
/// @param identifier The asset identifier.
/// @param error   On failure, error is filled with the failure information.
/// @retval `YES` is the asset was removed or didn't exist otherwise `NO`.
- (BOOL)removeAssetWithIdentifier:(NSString*)identifier error:(NSError* __autoreleasing*)error;

/// Checks if the asset associated with the identifier exists.
///
/// @param identifier The asset identifier.
/// @param error   On failure, error is filled with the failure information.
/// @retval `YES` is the asset exists otherwise `NO`.
- (BOOL)hasAssetWithIdentifier:(NSString*)identifier error:(NSError* __autoreleasing*)error;

/// Returns an array of most recently used assets. Assets are sorted in descending order by their
/// access time and the first `maxCount` assets are returned. The access time of an asset is updated
/// when the asset is stored or retrieved.
///
/// @param maxCount The max count of assets to return.
/// @param error   On failure, error is filled with the failure information.
/// @retval An array of most recently used assets
- (nullable NSArray<ETCoreMLAsset*>*)mostRecentlyUsedAssetsWithMaxCount:(NSUInteger)maxCount
                                                                  error:(NSError* __autoreleasing*)error;

/// Compacts the assets storage. The assets are moved to the trash directory and are asynchronously
/// deleted.
///
/// @param sizeInBytes The maximum size of the assets storage that the compaction should achieve.
/// @param error   On failure, error is filled with the failure information.
/// @retval The size of the assets store after the compaction.
- (NSUInteger)compact:(NSUInteger)sizeInBytes error:(NSError* __autoreleasing*)error;


/// Purges the assets storage. The assets are moved to the trash directory and are asynchronously
/// deleted.
///
/// @param error   On failure, error is filled with the failure information.
/// @retval `YES` is the assets are purged otherwise `NO`.
- (BOOL)purgeAndReturnError:(NSError* __autoreleasing*)error;

/// The estimated size of the assets store. The returned value might not correct if the asset
/// directory is tampered externally.
@property (assign, readonly, atomic) NSInteger estimatedSizeInBytes;

/// The maximum size of the assets store.
@property (assign, readonly, nonatomic) NSInteger maxAssetsSizeInBytes;

/// The trash directory URL, the assets before removal are moved to this directory. The directory
/// contents are deleted asynchronously.
@property (copy, readonly, nonatomic) NSURL* trashDirectoryURL;

/// The file manager.
@property (strong, readonly, nonatomic) NSFileManager* fileManager;

@end

NS_ASSUME_NONNULL_END
