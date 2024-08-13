//
// ETCoreMLAssetManager.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLAssetManager.h"
#import <ETCoreMLAsset.h>
#import <ETCoreMLLogging.h>
#import <database.hpp>
#import <iostream>
#import <json_key_value_store.hpp>
#import <serde_json.h>
#import <sstream>

namespace  {

using namespace executorchcoreml;
using namespace executorchcoreml::sqlite;

constexpr size_t kBusyTimeIntervalInMS = 100;

constexpr std::string_view kModelAssetsStoreName = "MODEL_ASSETS_STORE";
constexpr std::string_view kModelAssetsMetaStoreName = "MODEL_ASSETS_STORE_META";

class ModelAssetsStore {
public:
    using StoreType = JSONKeyValueStore<std::string, Asset>;
    
    ModelAssetsStore(std::unique_ptr<StoreType> impl) noexcept
    :impl_(std::move(impl))
    {}
    
    ModelAssetsStore() noexcept
    :impl_(nullptr)
    {}
    
    ModelAssetsStore(const ModelAssetsStore &) = delete;
    ModelAssetsStore &operator=(const ModelAssetsStore &) = delete;
    
    ModelAssetsStore& operator=(ModelAssetsStore&& rhs) noexcept {
        rhs.impl_.swap(impl_);
        return *this;
    }
    
    ModelAssetsStore(ModelAssetsStore&& rhs) noexcept
    :impl_(std::move(rhs.impl_))
    {}
    
    inline StoreType *impl() {
        return impl_.get();
    }
    
private:
    std::unique_ptr<StoreType> impl_;
};

class ModelAssetsMetaStore {
public:
    using StoreType = KeyValueStore<std::string, size_t>;
    
    ModelAssetsMetaStore() noexcept
    :impl_(nullptr)
    {}
    
    ModelAssetsMetaStore(std::unique_ptr<StoreType> impl) noexcept
    :impl_(std::move(impl))
    {}
    
    ModelAssetsMetaStore(const ModelAssetsMetaStore &) = delete;
    ModelAssetsMetaStore &operator=(const ModelAssetsMetaStore &) = delete;
    
    ModelAssetsMetaStore& operator=(ModelAssetsMetaStore&& rhs) noexcept {
        rhs.impl_.swap(impl_);
        return *this;
    }
    
    ModelAssetsMetaStore(ModelAssetsMetaStore&& rhs) noexcept
    :impl_(std::move(rhs.impl_))
    {}
    
    inline StoreType *impl() {
        return impl_.get();
    }
    
private:
    std::unique_ptr<StoreType> impl_;
};

void set_error_from_error_code(const std::error_code& cppError, NSError * __autoreleasing *error) {
    if (!error || !cppError) {
        return;
    }
    
    NSString *message = @(cppError.message().c_str());
    NSString *domain =  @(cppError.category().name());
    NSInteger code = cppError.value();
    NSError *localError = [NSError errorWithDomain:domain code:code userInfo:@{NSLocalizedDescriptionKey : message}];
    *error = localError;
}

std::shared_ptr<Database> make_database(NSURL *database_url,
                                        NSTimeInterval busy_time_interval,
                                        NSError * __autoreleasing *error) {
    Database::OpenOptions options;
    options.set_read_write_option(true);
    options.set_create_option(true);
    
    std::error_code ec;
    auto database = Database::make(database_url.path.UTF8String,
                                   options,
                                   Database::SynchronousMode::Normal,
                                   busy_time_interval,
                                   ec);
    if (!database) {
        ::set_error_from_error_code(ec, error);
        return nullptr;
    }
    
    return database;
}

ModelAssetsStore make_assets_store(const std::shared_ptr<Database>& database,
                                   NSError * __autoreleasing *error) {
    std::error_code ec;
    auto store = ModelAssetsStore::StoreType::make(std::move(database), std::string(kModelAssetsStoreName), ec);
    if (!store) {
        ::set_error_from_error_code(ec, error);
        return ModelAssetsStore(nullptr);
    }
    
    return ModelAssetsStore(std::move(store));
}

ModelAssetsMetaStore make_assets_meta_store(const std::shared_ptr<Database>& database,
                                            NSError * __autoreleasing *error) {
    std::error_code ec;
    auto store = ModelAssetsMetaStore::StoreType::make(std::move(database), std::string(kModelAssetsMetaStoreName), ec);
    if (!store) {
        ::set_error_from_error_code(ec, error);
        return ModelAssetsMetaStore(nullptr);
    }
    
    return ModelAssetsMetaStore(std::move(store));
}

std::optional<size_t> get_total_assets_size(ModelAssetsMetaStore& store,
                                            std::error_code& ec) {
    std::string name = std::string(kModelAssetsStoreName);
    auto total_size = store.impl()->get(name, ec);
    
    if (!total_size && !store.impl()->put(name, size_t(0), ec)) {
        return std::nullopt;
    }
    
    size_t result = total_size.has_value() ? total_size.value() : size_t(0);
    return result;
}

bool set_total_assets_size(size_t total_size,
                           ModelAssetsMetaStore& store,
                           std::error_code& ec) {
    if (!store.impl()->put(std::string(kModelAssetsStoreName), total_size, ec)) {
        return false;
    }
    
    return true;
}

bool exclude_item_from_backup(NSURL *url, NSError * __autoreleasing *error) {
    return [url setResourceValue:@(YES) forKey:NSURLIsExcludedFromBackupKey error:error];
}

NSURL * _Nullable create_directory_if_needed(NSURL *url,
                                             NSString *name,
                                             NSFileManager *fm,
                                             NSError * __autoreleasing *error) {
    NSURL *directory_url = [url URLByAppendingPathComponent:name];
    if (![fm fileExistsAtPath:directory_url.path] &&
        ![fm createDirectoryAtURL:directory_url withIntermediateDirectories:NO attributes:@{} error:error]) {
        return nil;
    }
    
    ::exclude_item_from_backup(directory_url, nil);
    
    return directory_url;
}

bool is_directory_empty(NSURL *url, NSFileManager *fm, NSError * __autoreleasing *error) {
    BOOL is_directory = NO;
    if (![fm fileExistsAtPath:url.path isDirectory:&is_directory] && !is_directory) {
        return true;
    }
    
    __block NSError *local_error = nil;
    BOOL (^errorHandler)(NSURL *url, NSError *error) = ^BOOL(NSURL *url, NSError *enumeration_error) {
        local_error = enumeration_error;
        return NO;
    };
    
    NSDirectoryEnumerator *enumerator = [fm enumeratorAtURL:url
                                 includingPropertiesForKeys:@[]
                                                    options:NSDirectoryEnumerationProducesRelativePathURLs
                                               errorHandler:errorHandler];
    if (local_error && error) {
        *error = local_error;
    }
    
    return [enumerator nextObject] == nil;
}

NSURL * _Nullable get_asset_url(const Asset& asset) {
    return [NSURL fileURLWithPath:@(asset.path.c_str())];
}

BOOL is_asset_alive(NSMapTable<NSString *, ETCoreMLAsset *> *assets_in_use_map, NSString *identifier) {
    ETCoreMLAsset *asset = [assets_in_use_map objectForKey:identifier];
    return asset && asset.isAlive;
}

std::vector<executorchcoreml::Asset>
get_assets_to_remove(ModelAssetsStore& store,
                     ssize_t bytes_to_remove,
                     NSMapTable<NSString *, ETCoreMLAsset *> *assets_in_use_map,
                     std::error_code &error) {
    std::vector<Asset> assets;
    store.impl()->get_keys_sorted_by_access_count([store = store.impl(),
                                                   &bytes_to_remove,
                                                   &assets,
                                                   assets_in_use_map,
                                                   &error](const std::string& key) {
        if (bytes_to_remove <= 0) {
            return false;
        }
        
        NSString *identifier = @(key.c_str());
        // Asset is in use, we can't remove it
        if (::is_asset_alive(assets_in_use_map, identifier)) {
            return true;
        }
        
        auto asset = store->get(key, error);
        if (asset) {
            auto& asset_value = asset.value();
            bytes_to_remove -= static_cast<ssize_t>(asset_value.total_size_in_bytes());
            assets.emplace_back(std::move(asset_value));
        }
        
        return true;
    }, SortOrder::Ascending, error);
    
    return assets;
}
} //namespace

@interface ETCoreMLAssetManager () <NSFileManagerDelegate> {
    ModelAssetsStore _assetsStore;
    ModelAssetsMetaStore _assetsMetaStore;
}

@property (assign, readwrite, atomic) NSInteger estimatedSizeInBytes;
@property (copy, readonly, nonatomic) NSURL *assetsDirectoryURL;
@property (strong, readonly, nonatomic) dispatch_queue_t syncQueue;
@property (strong, readonly, nonatomic) dispatch_queue_t trashQueue;
@property (strong, readonly, nonatomic) NSMapTable<NSString *, ETCoreMLAsset *> *assetsInUseMap;

@end

@implementation ETCoreMLAssetManager

- (nullable instancetype)initWithDatabase:(const std::shared_ptr<Database>&)database
                       assetsDirectoryURL:(NSURL *)assetsDirectoryURL
                        trashDirectoryURL:(NSURL *)trashDirectoryURL
                     maxAssetsSizeInBytes:(NSInteger)maxAssetsSizeInBytes
                                    error:(NSError * __autoreleasing *)error {
    
    auto assetsStore = ::make_assets_store(database, error);
    if (assetsStore.impl() == nullptr) {
        return nil;
    }
    
    auto assetsMetaStore = ::make_assets_meta_store(database, error);
    if (assetsMetaStore.impl() == nullptr) {
        return nil;
    }
    
    std::error_code ec;
    auto sizeInBytes = ::get_total_assets_size(assetsMetaStore, ec);
    if (!sizeInBytes) {
        ::set_error_from_error_code(ec, error);
        return nil;
    }
    
    NSFileManager *fileManager = [[NSFileManager alloc] init];
    NSURL *managedAssetsDirectoryURL = ::create_directory_if_needed(assetsDirectoryURL, @"models", fileManager, error);
    if (!managedAssetsDirectoryURL) {
        return nil;
    }
    
    NSURL *managedTrashDirectoryURL = ::create_directory_if_needed(trashDirectoryURL, @"models", fileManager, error);
    if (!managedTrashDirectoryURL) {
        return nil;
    }
    
    // If directory is empty then purge the stores
    if (::is_directory_empty(managedAssetsDirectoryURL, fileManager, nil)) {
        assetsMetaStore.impl()->purge(ec);
        assetsStore.impl()->purge(ec);
    }
    
    if (self = [super init]) {
        _assetsStore = std::move(assetsStore);
        _assetsMetaStore = std::move(assetsMetaStore);
        _assetsDirectoryURL = managedAssetsDirectoryURL;
        _trashDirectoryURL = managedTrashDirectoryURL;
        _estimatedSizeInBytes = sizeInBytes.value();
        _maxAssetsSizeInBytes = maxAssetsSizeInBytes;
        
        _fileManager = fileManager;
        _trashQueue = dispatch_queue_create("com.executorchcoreml.assetmanager.trash", DISPATCH_QUEUE_SERIAL_WITH_AUTORELEASE_POOL);
        _syncQueue = dispatch_queue_create("com.executorchcoreml.assetmanager.sync", DISPATCH_QUEUE_SERIAL_WITH_AUTORELEASE_POOL);
        _assetsInUseMap = [NSMapTable strongToWeakObjectsMapTable];
    }
    
    [self triggerCompaction];
    return self;
}

- (nullable instancetype)initWithDatabaseURL:(NSURL *)databaseURL
                          assetsDirectoryURL:(NSURL *)assetsDirectoryURL
                           trashDirectoryURL:(NSURL *)trashDirectoryURL
                        maxAssetsSizeInBytes:(NSInteger)maxAssetsSizeInBytes
                                       error:(NSError * __autoreleasing *)error {
    auto database = make_database(databaseURL, kBusyTimeIntervalInMS, error);
    if (!database) {
        return nil;
    }
    
    return [self initWithDatabase:database
               assetsDirectoryURL:assetsDirectoryURL
                trashDirectoryURL:trashDirectoryURL
             maxAssetsSizeInBytes:maxAssetsSizeInBytes
                            error:error];
}

- (nullable NSURL *)moveURL:(NSURL *)url
     toUniqueURLInDirectory:(NSURL *)directoryURL
                      error:(NSError * __autoreleasing *)error {
    NSURL *dstURL = [directoryURL URLByAppendingPathComponent:[NSUUID UUID].UUIDString];
    if (![self.fileManager moveItemAtURL:url toURL:dstURL error:error]) {
        return nil;
    }
    
    return dstURL;
}

- (void)cleanupAssetIfNeeded:(ETCoreMLAsset *)asset {
    if (!asset || asset.isValid) {
        return;
    }
    
    NSString *identifier = asset.identifier;
    dispatch_async(self.syncQueue, ^{
        NSError *cleanupError = nil;
        if (![self _removeAssetWithIdentifier:asset.identifier error:&cleanupError]) {
            ETCoreMLLogError(cleanupError,
                             "%@: Failed to remove asset with identifier = %@",
                             NSStringFromClass(ETCoreMLAssetManager.class),
                             identifier);
        }
    });
}

- (nullable ETCoreMLAsset *)_storeAssetAtURL:(NSURL *)srcURL
                              withIdentifier:(NSString *)identifier
                                       error:(NSError * __autoreleasing *)error {
    dispatch_assert_queue(self.syncQueue);
    NSString *extension = srcURL.lastPathComponent.pathExtension;
    NSURL *dstURL = [self.assetsDirectoryURL URLByAppendingPathComponent:[NSString stringWithFormat:@"%@.%@", identifier, extension]];
    auto asset = Asset::make(srcURL, identifier, self.fileManager, error);
    if (!asset) {
        return nil;
    }
    
    auto& assetValue = asset.value();
    size_t assetSizeInBytes = assetValue.total_size_in_bytes();
    std::error_code ec;
    // Update the stores inside a transaction, if anything fails it will automatically rollback to the previous state.
    bool status = _assetsStore.impl()->transaction([self, &assetValue, assetSizeInBytes, srcURL, dstURL, &ec, error]() {
        const std::string& assetIdentifier = assetValue.identifier;
        // If an asset exists with the same identifier then remove it.
        if (![self _removeAssetWithIdentifier:@(assetIdentifier.c_str()) error:error]) {
            return false;
        }
        
        // Update asset path.
        assetValue.path = dstURL.path.UTF8String;
        // Store the asset.
        if (!_assetsStore.impl()->put(assetIdentifier, assetValue, ec)) {
            return false;
        }
        
        // Update the size of the store.
        if (!::set_total_assets_size(_estimatedSizeInBytes + assetSizeInBytes, _assetsMetaStore, ec)) {
            return false;
        }
        
        // If an asset exists move it
        [self moveURL:dstURL toUniqueURLInDirectory:self.trashDirectoryURL error:nil];
        
        // Move the asset to assets directory.
        if (![self.fileManager moveItemAtURL:srcURL toURL:dstURL error:error]) {
            return false;
        }
        
        return true;
    }, Database::TransactionBehavior::Immediate, ec);
    
    // Update the estimated size if the transaction succeeded.
    _estimatedSizeInBytes += status ? assetSizeInBytes : 0;
    ::set_error_from_error_code(ec, error);
    
    ETCoreMLAsset *result = status ? [[ETCoreMLAsset alloc] initWithBackingAsset:assetValue] : nil;
    if ([result keepAliveAndReturnError:error]) {
        [self.assetsInUseMap setObject:result forKey:identifier];
    } else {
        [self cleanupAssetIfNeeded:result];
    }
    
    return result;
}

- (void)triggerCompaction {
    if (self.estimatedSizeInBytes < self.maxAssetsSizeInBytes) {
        return;
    }
    
    __weak __typeof(self) weakSelf = self;
    dispatch_async(self.syncQueue, ^{
        NSError *localError = nil;
        if (![weakSelf _compact:self.maxAssetsSizeInBytes error:&localError]) {
            ETCoreMLLogError(localError,
                             "%@: Failed to compact asset store.",
                             NSStringFromClass(ETCoreMLAssetManager.class));
        }
    });
}

- (nullable ETCoreMLAsset *)storeAssetAtURL:(NSURL *)url
                             withIdentifier:(NSString *)identifier
                                      error:(NSError * __autoreleasing *)error {
    __block ETCoreMLAsset *result = nil;
    dispatch_sync(self.syncQueue, ^{
        result = [self _storeAssetAtURL:url withIdentifier:identifier error:error];
    });
    
    [self triggerCompaction];
    return result;
}

- (nullable ETCoreMLAsset *)_assetWithIdentifier:(NSString *)identifier
                                           error:(NSError * __autoreleasing *)error {
    dispatch_assert_queue(self.syncQueue);
    std::string assetIdentifier(identifier.UTF8String);
    std::error_code ec;
    auto asset = _assetsStore.impl()->get(assetIdentifier, ec);
    if (!asset) {
        ::set_error_from_error_code(ec, error);
        return nil;
    }
    
    const auto& assetValue = asset.value();
    ETCoreMLAsset *modelAsset = [[ETCoreMLAsset alloc] initWithBackingAsset:assetValue];
    [self.assetsInUseMap setObject:modelAsset forKey:identifier];
    
    return modelAsset;
}

- (nullable ETCoreMLAsset *)assetWithIdentifier:(NSString *)identifier
                                          error:(NSError * __autoreleasing *)error {
    __block ETCoreMLAsset *result = nil;
    dispatch_sync(self.syncQueue, ^{
        result = [self _assetWithIdentifier:identifier error:error];
    });
    
    if ([result keepAliveAndReturnError:error]) {
        [self.assetsInUseMap setObject:result forKey:identifier];
    } else {
        [self cleanupAssetIfNeeded:result];
    }
    
    return result;
}

- (BOOL)_containsAssetWithIdentifier:(NSString *)identifier
                               error:(NSError * __autoreleasing *)error {
    dispatch_assert_queue(self.syncQueue);
    std::error_code ec;
    BOOL result = static_cast<BOOL>(_assetsStore.impl()->exists(std::string(identifier.UTF8String), ec));
    ::set_error_from_error_code(ec, error);
    
    return result;
}

- (BOOL)hasAssetWithIdentifier:(NSString *)identifier
                         error:(NSError * __autoreleasing *)error {
    __block BOOL result = NO;
    dispatch_sync(self.syncQueue, ^{
        result = [self _containsAssetWithIdentifier:identifier error:error];
    });
    
    return result;
}

- (BOOL)_removeAssetWithIdentifier:(NSString *)identifier
                             error:(NSError * __autoreleasing *)error {
    dispatch_assert_queue(self.syncQueue);
    // Asset is alive we can't delete it.
    if (is_asset_alive(self.assetsInUseMap, identifier)) {
        return NO;
    }
    
    std::error_code ec;
    std::string assetIdentifier(identifier.UTF8String);
    auto asset = _assetsStore.impl()->get(assetIdentifier, ec);
    // If it's an error then we can't proceed.
    if (ec) {
        ::set_error_from_error_code(ec, error);
        return NO;
    }
    
    // Asset doesn't exists, we are good.
    if (!asset) {
        return YES;
    }
    
    const auto& assetValue = asset.value();
    size_t assetSizeInBytes = std::min(_estimatedSizeInBytes, static_cast<NSInteger>(assetValue.total_size_in_bytes()));
    // Update the stores inside a transaction, if anything fails it will automatically rollback to the previous state.
    bool status = _assetsStore.impl()->transaction([self, &assetValue, assetSizeInBytes, &ec, error]() {
        if (!self->_assetsStore.impl()->remove(assetValue.identifier, ec)) {
            return false;
        }
        
        if (!::set_total_assets_size(_estimatedSizeInBytes - assetSizeInBytes, _assetsMetaStore, ec)) {
            return false;
        }
        
        NSURL *assetURL = ::get_asset_url(assetValue);
        if ([self.fileManager fileExistsAtPath:assetURL.path] &&
            ![self moveURL:assetURL toUniqueURLInDirectory:self.trashDirectoryURL error:error]) {
            return false;
        }
        
        return true;
    }, Database::TransactionBehavior::Immediate, ec);
    
    // Update the estimated size if the transaction succeeded.
    _estimatedSizeInBytes -= status ? assetSizeInBytes : 0;
    ::set_error_from_error_code(ec, error);
    return static_cast<BOOL>(status);
}

- (BOOL)removeAssetWithIdentifier:(NSString *)identifier
                            error:(NSError * __autoreleasing *)error {
    __block BOOL result = NO;
    dispatch_sync(self.syncQueue, ^{
        result = [self _removeAssetWithIdentifier:identifier error:error];
    });
    
    return result;
}

- (nullable NSArray<ETCoreMLAsset *> *)_recentlyUsedAssetsWithMaxCount:(NSUInteger)maxCount
                                                                 error:(NSError * __autoreleasing *)error {
    dispatch_assert_queue(self.syncQueue);
    
    NSMutableArray<ETCoreMLAsset *> *assets = [NSMutableArray arrayWithCapacity:maxCount];
    std::error_code ec;
    bool status = _assetsStore.impl()->get_keys_sorted_by_access_time([self, maxCount, assets](const std::string& key) {
        NSError *localError = nil;
        NSString *identifier = @(key.c_str());
        ETCoreMLAsset *asset = [self _assetWithIdentifier:identifier error:&localError];
        
        if (asset) {
            [assets addObject:asset];
        } else if (localError) {
            ETCoreMLLogError(localError,
                             "%@: Failed to retrieve asset with identifier = %@",
                             NSStringFromClass(ETCoreMLAssetManager.class),
                             identifier);
        }
        
        return assets.count < maxCount;
    }, SortOrder::Descending, ec);
    
    ::set_error_from_error_code(ec, error);
    return status ? assets : nil;
}

- (nullable NSArray<ETCoreMLAsset *> *)mostRecentlyUsedAssetsWithMaxCount:(NSUInteger)maxCount
                                                                    error:(NSError * __autoreleasing *)error {
    __block NSArray<ETCoreMLAsset *> *result = nil;
    dispatch_sync(self.syncQueue, ^{
        result = [self _recentlyUsedAssetsWithMaxCount:maxCount error:error];
    });
    
    return result;
}

- (BOOL)_canPurgeStore {
    dispatch_assert_queue(self.syncQueue);
    
    NSEnumerator *keyEnumerator = self.assetsInUseMap.keyEnumerator;
    for (NSString *identifier in keyEnumerator) {
        if (is_asset_alive(self.assetsInUseMap, identifier)) {
            return NO;
        }
    }
    
    return YES;
}

- (NSUInteger)_compact:(NSUInteger)sizeInBytes error:(NSError * __autoreleasing *)error {
    dispatch_assert_queue(self.syncQueue);
    
    if (sizeInBytes == 0 && [self _canPurgeStore]) {
        return [self _purge:error] ? 0 : _estimatedSizeInBytes;
    }
    
    if (_estimatedSizeInBytes <= sizeInBytes) {
        return _estimatedSizeInBytes;
    }
    
    std::error_code ec;
    ssize_t bytesToRemove = _estimatedSizeInBytes - sizeInBytes;
    const auto& assets = ::get_assets_to_remove(_assetsStore, bytesToRemove, self.assetsInUseMap, ec);
    
    if (ec) {
        ::set_error_from_error_code(ec, error);
        return _estimatedSizeInBytes;
    }
    
    for (const auto& asset : assets) {
        NSError *cleanupError = nil;
        NSString *identifier = @(asset.identifier.c_str());
        if (![self _removeAssetWithIdentifier:identifier error:&cleanupError] && cleanupError) {
            ETCoreMLLogError(cleanupError,
                             "%@: Failed to remove asset with identifier = %@",
                             NSStringFromClass(ETCoreMLAssetManager.class),
                             identifier);
        }
    }
    
    // Trigger cleanup.
    __weak __typeof(self) weakSelf = self;
    dispatch_async(self.trashQueue, ^{
        [weakSelf removeFilesInTrashDirectory];
    });
    
    return _estimatedSizeInBytes;
}

- (NSUInteger)compact:(NSUInteger)sizeInBytes error:(NSError * __autoreleasing *)error {
    __block NSUInteger result = 0;
    dispatch_sync(self.syncQueue, ^{
        result = [self _compact:sizeInBytes error:error];
    });
    
    return result;
}

- (void)removeFilesInTrashDirectory {
    dispatch_assert_queue(self.trashQueue);
    
    NSFileManager *fileManager = [[NSFileManager alloc] init];
    fileManager.delegate = self;
    __block NSError *localError = nil;
    BOOL (^errorHandler)(NSURL *url, NSError *error) = ^BOOL(NSURL *url, NSError *enumerationError) {
        localError = enumerationError;
        return YES;
    };
    
    NSDirectoryEnumerator *enumerator = [fileManager enumeratorAtURL:self.trashDirectoryURL
                                          includingPropertiesForKeys:@[]
                                                             options:NSDirectoryEnumerationSkipsSubdirectoryDescendants
                                                        errorHandler:errorHandler];
    for (NSURL *itemURL in enumerator) {
        if (![fileManager removeItemAtURL:itemURL error:&localError]) {
            ETCoreMLLogError(localError,
                             "%@: Failed to remove item in trash directory with name = %@",
                             NSStringFromClass(ETCoreMLAssetManager.class),
                             itemURL.lastPathComponent);
        }
    }
}

- (BOOL)_purge:(NSError * __autoreleasing *)error {
    dispatch_assert_queue(self.syncQueue);
    
    std::error_code ec;
    bool status = _assetsStore.impl()->transaction([self, &ec, error]() {
        // Purge the assets store.
        if (!self->_assetsStore.impl()->purge(ec)) {
            return false;
        }
        
        // Purge the assets size store.
        if (!self->_assetsMetaStore.impl()->purge(ec)) {
            return false;
        }
        
        // Move the the whole assets directory to the temp directory.
        if (![self moveURL:self.assetsDirectoryURL toUniqueURLInDirectory:self.trashDirectoryURL error:error]) {
            return false;
        }
        
        self->_estimatedSizeInBytes = 0;
        NSError *localError = nil;
        // Create the assets directory, if we fail here it's okay.
        if (![self.fileManager createDirectoryAtURL:self.assetsDirectoryURL withIntermediateDirectories:NO attributes:@{} error:&localError]) {
            ETCoreMLLogError(localError,
                             "%@: Failed to create assets directory",
                             NSStringFromClass(ETCoreMLAssetManager.class));
        }
        
        return true;
    }, Database::TransactionBehavior::Immediate, ec);
    
    ::set_error_from_error_code(ec, error);
    // Trigger cleanup
    if (status) {
        __weak __typeof(self) weakSelf = self;
        dispatch_async(self.trashQueue, ^{
            [weakSelf removeFilesInTrashDirectory];
        });
    }
    
    return static_cast<BOOL>(status);
}

- (BOOL)purgeAndReturnError:(NSError * __autoreleasing *)error {
    __block BOOL result = 0;
    dispatch_sync(self.syncQueue, ^{
        result = [self _purge:error];
    });
    
    return result;
}

- (BOOL)fileManager:(NSFileManager *)fileManager shouldProceedAfterError:(NSError *)error removingItemAtURL:(NSURL *)URL {
    return YES;
}

@end
