//
// ETCoreMLModelCache.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLModelCache.h"

#import "ETCoreMLLogging.h"

NSString* const ETCoreMLModelCacheErrorDomain = @"com.executorch.coreml.cache";


// Timing
static NSTimeInterval const kStaleEntryTimeoutSeconds = 24 * 60 * 60; // 24 hours

// Cache structure
static NSString* const kCacheFormatVersion = @"1";
static NSString* const kModelsDirectoryName = @"models";
static NSString* const kTempDirectoryName = @"temp";
static NSString* const kVersionFileName = @"version.txt";

// File extensions (without leading dot - added during URL construction)
static NSString* const kModelBundleExtension = @"mlmodelc";
static NSString* const kAccessedFileExtension = @"accessed";

@interface ETCoreMLModelCache ()
@property (nonatomic, strong) NSURL* modelsDirectory;
@property (nonatomic, strong) NSURL* tempDirectory;
@property (nonatomic, readwrite, getter=isReady) BOOL ready;
@property (nonatomic, readwrite, nullable) NSError* initializationError;
@end

@implementation ETCoreMLModelCache

#pragma mark - Initialization

- (instancetype)initWithCacheRootDirectory:(NSURL*)cacheRootDirectory {
    self = [super init];
    if (self) {
        _cacheRootDirectory = cacheRootDirectory;
        _modelsDirectory = [cacheRootDirectory URLByAppendingPathComponent:kModelsDirectoryName
                                                               isDirectory:YES];
        _tempDirectory = [cacheRootDirectory URLByAppendingPathComponent:kTempDirectoryName
                                                             isDirectory:YES];

        NSError* initError = nil;
        _ready = [self initializeDirectoriesWithError:&initError];
        _initializationError = initError;

        if (_ready) {
            // Cleanup stale entries synchronously during init.
            // These operations are fast (just a directory scan and a few deletions)
            // and only target entries older than 24 hours, so no race with active writes.
            [self cleanupStaleEntriesInDirectory:_tempDirectory name:@"temp"];
            [self cleanupOrphanedEntries];
        }
    }
    return self;
}

- (BOOL)initializeDirectoriesWithError:(NSError**)error {
    NSFileManager* fm = [NSFileManager defaultManager];

    // Create all directories
    NSArray<NSURL*>* directories = @[_cacheRootDirectory, _modelsDirectory, _tempDirectory];
    for (NSURL* dir in directories) {
        if (![fm createDirectoryAtURL:dir
          withIntermediateDirectories:YES
                           attributes:nil
                                error:error]) {
            return NO;
        }
        [self applyFileProtectionToDirectory:dir];
    }

    // Exclude from iCloud/iTunes backup. Only needed on root since this IS inherited
    // by children. Cached compiled models are regenerable, so backing them up wastes
    // the user's iCloud storage and backup time.
    [_cacheRootDirectory setResourceValue:@YES forKey:NSURLIsExcludedFromBackupKey error:nil];

    // Handle version file
    NSURL* versionURL = [_cacheRootDirectory URLByAppendingPathComponent:kVersionFileName];
    if ([fm fileExistsAtPath:versionURL.path]) {
        // Validate existing version
        NSString* existingVersion = [NSString stringWithContentsOfURL:versionURL
                                                             encoding:NSUTF8StringEncoding
                                                                error:nil];
        if (existingVersion) {
            existingVersion = [existingVersion stringByTrimmingCharactersInSet:
                              [NSCharacterSet whitespaceAndNewlineCharacterSet]];
        }

        if (!existingVersion || ![existingVersion isEqualToString:kCacheFormatVersion]) {
            // Version mismatch or unreadable - purge and recreate
            if (existingVersion) {
                ETCoreMLLogInfo("Cache version mismatch (found '%@', expected '%@'). Purging cache.",
                                existingVersion, kCacheFormatVersion);
            } else {
                ETCoreMLLogInfo("Cache version file unreadable. Purging cache.");
            }

            // Remove all contents of models and temp directories
            for (NSURL* dir in @[_modelsDirectory, _tempDirectory]) {
                [self clearContentsOfDirectory:dir];
            }

            // Update version file
            if (![kCacheFormatVersion writeToURL:versionURL
                                      atomically:YES
                                        encoding:NSUTF8StringEncoding
                                           error:error]) {
                return NO;
            }
        }
    } else {
        // Write new version file
        if (![kCacheFormatVersion writeToURL:versionURL
                                  atomically:YES
                                    encoding:NSUTF8StringEncoding
                                       error:error]) {
            return NO;
        }
    }

    return YES;
}

#pragma mark - Directory Utilities

- (BOOL)ensureDirectoryExists:(NSURL*)directory error:(NSError**)error {
    NSFileManager* fm = [NSFileManager defaultManager];

    // Fast path: directory already exists
    if ([fm fileExistsAtPath:directory.path]) {
        return YES;
    }

    // Slow path: recreate directory (withIntermediateDirectories also recreates parents)
    if (![fm createDirectoryAtURL:directory
        withIntermediateDirectories:YES
                         attributes:nil
                              error:error]) {
        return NO;
    }

    // Reapply file protection attribute to recreated directory
    [self applyFileProtectionToDirectory:directory];
    return YES;
}

- (void)applyFileProtectionToDirectory:(NSURL*)directory {
    NSFileManager* fm = [NSFileManager defaultManager];
    NSDictionary* protectionAttrs = @{NSFileProtectionKey: NSFileProtectionCompleteUntilFirstUserAuthentication};
    [fm setAttributes:protectionAttrs ofItemAtPath:directory.path error:nil]; // best-effort
}

- (void)clearContentsOfDirectory:(NSURL*)directory {
    NSFileManager* fm = [NSFileManager defaultManager];
    NSArray* contents = [fm contentsOfDirectoryAtURL:directory
                          includingPropertiesForKeys:nil
                                             options:0
                                               error:nil];
    for (NSURL* item in contents) {
        [fm removeItemAtURL:item error:nil];
    }
}

#pragma mark - Identifier Validation

- (BOOL)isValidIdentifier:(NSString*)identifier error:(NSError**)error {
    if (identifier.length == 0) {
        if (error) {
            *error = [NSError errorWithDomain:ETCoreMLModelCacheErrorDomain
                                         code:ETCoreMLModelCacheErrorCodeInvalidIdentifier
                                     userInfo:@{NSLocalizedDescriptionKey: @"Identifier cannot be empty"}];
        }
        return NO;
    }

    if ([identifier containsString:@"/"] || [identifier containsString:@".."]) {
        if (error) {
            *error = [NSError errorWithDomain:ETCoreMLModelCacheErrorDomain
                                         code:ETCoreMLModelCacheErrorCodeInvalidIdentifier
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"Identifier cannot contain '/' or '..'"}];
        }
        return NO;
    }

    return YES;
}

- (BOOL)validateReadyAndIdentifier:(NSString*)identifier error:(NSError**)error {
    if (!self.isReady) {
        if (error) *error = self.initializationError;
        return NO;
    }
    return [self isValidIdentifier:identifier error:error];
}

#pragma mark - URL Helpers

- (NSURL*)modelURLForIdentifier:(NSString*)identifier {
    NSString* bundleName = [NSString stringWithFormat:@"%@.%@", identifier, kModelBundleExtension];
    return [_modelsDirectory URLByAppendingPathComponent:bundleName isDirectory:YES];
}

- (NSURL*)accessedURLForIdentifier:(NSString*)identifier {
    NSString* fileName = [NSString stringWithFormat:@"%@.%@", identifier, kAccessedFileExtension];
    return [_modelsDirectory URLByAppendingPathComponent:fileName isDirectory:NO];
}

#pragma mark - Cache Operations

- (nullable NSURL*)cachedModelURLForIdentifier:(NSString*)identifier
                                         error:(NSError**)error {
    if (!self.isReady) {
        if (error) *error = self.initializationError;
        return nil;
    }

    NSFileManager* fm = [NSFileManager defaultManager];

    // Check bundle exists (cheap stat() call)
    NSURL* modelURL = [self modelURLForIdentifier:identifier];
    BOOL isDirectory = NO;
    if (![fm fileExistsAtPath:modelURL.path isDirectory:&isDirectory]) {
        return nil;
    }

    if (!isDirectory) {
        [fm removeItemAtURL:modelURL error:nil];
        return nil;
    }

    // Update access time for LRU eviction (also recreates file if deleted externally)
    [self touchAccessedFileForIdentifier:identifier];

    return modelURL;
}

- (nullable NSURL*)storeModelAtURL:(NSURL*)compiledModelURL
                    withIdentifier:(NSString*)identifier
                             error:(NSError**)error {
    if (![self validateReadyAndIdentifier:identifier error:error]) {
        return nil;
    }

    if (![self ensureDirectoryExists:_modelsDirectory error:error]) {
        return nil;
    }

    NSFileManager* fm = [NSFileManager defaultManager];
    if (![fm fileExistsAtPath:compiledModelURL.path]) {
        if (error) {
            *error = [NSError errorWithDomain:ETCoreMLModelCacheErrorDomain
                                         code:ETCoreMLModelCacheErrorCodeSourceNotFound
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         @"Source model URL does not exist"}];
        }
        return nil;
    }

    // Remove existing model if present
    NSURL* destURL = [self modelURLForIdentifier:identifier];
    if ([fm fileExistsAtPath:destURL.path]) {
        NSError* removeError = nil;
        if (![fm removeItemAtURL:destURL error:&removeError]) {
            if (error) {
                *error = [NSError errorWithDomain:ETCoreMLModelCacheErrorDomain
                                             code:ETCoreMLModelCacheErrorCodeIOError
                                         userInfo:@{
                                             NSLocalizedDescriptionKey: @"Cannot remove existing cached model",
                                             NSUnderlyingErrorKey: removeError
                                         }];
            }
            return nil;
        }
    }

    // Remove any orphaned accessed file for this identifier
    [fm removeItemAtURL:[self accessedURLForIdentifier:identifier] error:nil];

    // Move source directly to final location (atomic if same filesystem)
    NSError* moveError = nil;
    if (![fm moveItemAtURL:compiledModelURL toURL:destURL error:&moveError]) {
        if ([moveError.domain isEqualToString:NSCocoaErrorDomain] &&
            moveError.code == NSFileWriteOutOfSpaceError) {
            ETCoreMLLogInfo("Disk full while storing model: %@", identifier);
            if (error) {
                *error = [NSError errorWithDomain:ETCoreMLModelCacheErrorDomain
                                             code:ETCoreMLModelCacheErrorCodeDiskFull
                                         userInfo:@{
                                             NSLocalizedDescriptionKey: @"Disk full",
                                             NSUnderlyingErrorKey: moveError
                                         }];
            }
        } else {
            if (error) {
                *error = [NSError errorWithDomain:ETCoreMLModelCacheErrorDomain
                                             code:ETCoreMLModelCacheErrorCodeIOError
                                         userInfo:@{
                                             NSLocalizedDescriptionKey: @"Cannot move model to cache",
                                             NSUnderlyingErrorKey: moveError
                                         }];
            }
        }
        return nil;
    }

    // Record access time for LRU eviction
    [self touchAccessedFileForIdentifier:identifier];

    return destURL;
}

- (BOOL)removeCachedModelWithIdentifier:(NSString*)identifier
                                  error:(NSError**)error {
    if (![self validateReadyAndIdentifier:identifier error:error]) {
        return NO;
    }

    [self removeAllFilesForIdentifier:identifier];
    return YES;
}

- (void)removeAllFilesForIdentifier:(NSString*)identifier {
    NSFileManager* fm = [NSFileManager defaultManager];

    // Remove all files for this identifier (best-effort, ignore errors)
    [fm removeItemAtURL:[self modelURLForIdentifier:identifier] error:nil];
    [fm removeItemAtURL:[self accessedURLForIdentifier:identifier] error:nil];
}

- (BOOL)purgeAndReturnError:(NSError**)error {
    if (!self.isReady) {
        if (error) *error = self.initializationError;
        return NO;
    }

    NSFileManager* fm = [NSFileManager defaultManager];

    // Remove models directory contents
    if ([fm fileExistsAtPath:_modelsDirectory.path]) {
        if (![fm removeItemAtURL:_modelsDirectory error:error]) {
            return NO;
        }
    }

    // Recreate all directories (in case any were deleted externally)
    for (NSURL* dir in @[_cacheRootDirectory, _modelsDirectory, _tempDirectory]) {
        if (![self ensureDirectoryExists:dir error:error]) {
            return NO;
        }
    }

    // Reapply file protection attributes
    for (NSURL* dir in @[_cacheRootDirectory, _modelsDirectory, _tempDirectory]) {
        [self applyFileProtectionToDirectory:dir];
    }

    // Clean temp contents
    [self clearContentsOfDirectory:_tempDirectory];

    return YES;
}

#pragma mark - Temp Directory (for mlpackage extraction before compilation)

- (nullable NSURL*)temporaryDirectoryWithError:(NSError**)error {
    if (!self.isReady) {
        if (error) *error = self.initializationError;
        return nil;
    }

    NSString* tempName = [NSUUID UUID].UUIDString;
    NSURL* tempURL = [_tempDirectory URLByAppendingPathComponent:tempName isDirectory:YES];

    NSFileManager* fm = [NSFileManager defaultManager];
    NSError* createError = nil;
    if (![fm createDirectoryAtURL:tempURL
      withIntermediateDirectories:YES
                       attributes:nil
                            error:&createError]) {
        if (error) *error = createError;
        return nil;
    }

    [self applyFileProtectionToDirectory:tempURL];
    return tempURL;
}

#pragma mark - Access Time Tracking

- (void)touchAccessedFileForIdentifier:(NSString*)identifier {
    NSURL* accessedURL = [self accessedURLForIdentifier:identifier];
    NSFileManager* fm = [NSFileManager defaultManager];

    if ([fm fileExistsAtPath:accessedURL.path]) {
        // Update mtime
        NSDictionary* attrs = @{NSFileModificationDate: [NSDate date]};
        [fm setAttributes:attrs ofItemAtPath:accessedURL.path error:nil];
    } else {
        // Create empty file
        [fm createFileAtPath:accessedURL.path contents:nil attributes:nil];
    }
}

- (NSTimeInterval)lastAccessTimeForIdentifier:(NSString*)identifier {
    NSURL* accessedURL = [self accessedURLForIdentifier:identifier];
    NSDate* mtime;
    if ([accessedURL getResourceValue:&mtime forKey:NSURLContentModificationDateKey error:nil] && mtime) {
        return mtime.timeIntervalSince1970;
    }
    return 0;
}

#pragma mark - Cleanup

- (void)cleanupStaleEntriesInDirectory:(NSURL*)directory name:(NSString*)name {
    NSFileManager* fm = [NSFileManager defaultManager];
    NSArray* contents = [fm contentsOfDirectoryAtURL:directory
                          includingPropertiesForKeys:@[NSURLCreationDateKey]
                                             options:0
                                               error:nil];

    NSDate* cutoff = [NSDate dateWithTimeIntervalSinceNow:-kStaleEntryTimeoutSeconds];
    NSUInteger cleanedCount = 0;

    for (NSURL* url in contents) {
        NSDate* creationDate;
        [url getResourceValue:&creationDate forKey:NSURLCreationDateKey error:nil];

        if (creationDate && [creationDate compare:cutoff] == NSOrderedAscending) {
            if ([fm removeItemAtURL:url error:nil]) {
                cleanedCount++;
            }
        }
    }

    if (cleanedCount > 0) {
        ETCoreMLLogInfo("Cleaned up %lu stale %@ entries", (unsigned long)cleanedCount, name);
    }
}

- (void)cleanupOrphanedEntries {
    NSFileManager* fm = [NSFileManager defaultManager];

    NSArray* contents = [fm contentsOfDirectoryAtURL:_modelsDirectory
                          includingPropertiesForKeys:@[NSURLIsDirectoryKey]
                                             options:0
                                               error:nil];
    if (!contents) {
        return;
    }

    // Collect identifiers from bundles and accessed files
    NSMutableSet<NSString*>* bundleIdentifiers = [NSMutableSet set];
    NSMutableSet<NSString*>* accessedIdentifiers = [NSMutableSet set];

    NSString* modelExtWithDot = [NSString stringWithFormat:@".%@", kModelBundleExtension];
    NSString* accessedExtWithDot = [NSString stringWithFormat:@".%@", kAccessedFileExtension];

    for (NSURL* url in contents) {
        NSString* lastComponent = url.lastPathComponent;

        if ([lastComponent hasSuffix:modelExtWithDot]) {
            NSString* identifier = [lastComponent substringToIndex:lastComponent.length - modelExtWithDot.length];
            [bundleIdentifiers addObject:identifier];
        } else if ([lastComponent hasSuffix:accessedExtWithDot]) {
            NSString* identifier = [lastComponent substringToIndex:lastComponent.length - accessedExtWithDot.length];
            [accessedIdentifiers addObject:identifier];
        }
    }

    NSUInteger orphanedAccessed = 0;

    // .accessed without bundle → delete .accessed
    for (NSString* identifier in accessedIdentifiers) {
        if (![bundleIdentifiers containsObject:identifier]) {
            NSURL* accessedURL = [self accessedURLForIdentifier:identifier];
            if ([fm removeItemAtURL:accessedURL error:nil]) {
                orphanedAccessed++;
            }
        }
    }

    if (orphanedAccessed > 0) {
        ETCoreMLLogInfo("Cleaned up %lu orphaned accessed files",
                        (unsigned long)orphanedAccessed);
    }
}

@end
