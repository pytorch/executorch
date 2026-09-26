//
// ETCoreMLModelCacheTests.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <XCTest/XCTest.h>

#import "ETCoreMLModelCache.h"

@interface ETCoreMLModelCacheTests : XCTestCase

@property (strong, nonatomic, nullable) ETCoreMLModelCache *cache;
@property (strong, nonatomic, nullable) NSFileManager *fileManager;
@property (copy, nonatomic) NSURL *testDirectoryURL;
@property (copy, nonatomic) NSURL *cacheRootURL;

@end

@implementation ETCoreMLModelCacheTests

- (void)setUp {
    [super setUp];
    self.fileManager = [[NSFileManager alloc] init];
    self.testDirectoryURL = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:[NSUUID UUID].UUIDString]];
    self.cacheRootURL = [self.testDirectoryURL URLByAppendingPathComponent:@"cache"];

    // ETCoreMLModelCache creates its directory structure in init
    self.cache = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:self.cacheRootURL];
    XCTAssertTrue(self.cache.isReady, @"Cache should be ready after initialization: %@", self.cache.initializationError);
}

- (void)tearDown {
    self.cache = nil;
    NSError *error = nil;
    if ([self.fileManager fileExistsAtPath:self.testDirectoryURL.path]) {
        [self.fileManager removeItemAtURL:self.testDirectoryURL error:&error];
    }
    self.fileManager = nil;
    [super tearDown];
}

#pragma mark - Helper Methods

- (NSURL *)createMockCompiledModelWithIdentifier:(NSString *)identifier {
    // Create a mock .mlmodelc directory structure
    NSURL *modelURL = [[self.testDirectoryURL URLByAppendingPathComponent:identifier] URLByAppendingPathExtension:@"mlmodelc"];
    NSError *error = nil;

    XCTAssertTrue([self.fileManager createDirectoryAtURL:modelURL withIntermediateDirectories:YES attributes:@{} error:&error],
                  @"Failed to create mock model directory: %@", error);

    // Add a mock model.mil file with identifier-specific content
    NSURL *modelFileURL = [modelURL URLByAppendingPathComponent:@"model.mil"];
    NSString *mockContent = [NSString stringWithFormat:@"mock model data for %@", identifier];
    NSData *mockData = [mockContent dataUsingEncoding:NSUTF8StringEncoding];
    [mockData writeToURL:modelFileURL atomically:YES];

    // Add a mock coremldata.bin file
    NSURL *dataFileURL = [modelURL URLByAppendingPathComponent:@"coremldata.bin"];
    NSData *mockBinaryData = [[NSData alloc] initWithBytes:"binary" length:6];
    [mockBinaryData writeToURL:dataFileURL atomically:YES];

    return modelURL;
}

#pragma mark - Initialization Tests

- (void)testInitializationCreatesDirectoryStructure {
    XCTAssertTrue(self.cache.isReady);
    XCTAssertNil(self.cache.initializationError);

    // Verify directory structure exists
    BOOL isDirectory = NO;
    XCTAssertTrue([self.fileManager fileExistsAtPath:self.cacheRootURL.path isDirectory:&isDirectory]);
    XCTAssertTrue(isDirectory);

    // Verify models directory exists
    NSURL *modelsURL = [self.cacheRootURL URLByAppendingPathComponent:@"models"];
    XCTAssertTrue([self.fileManager fileExistsAtPath:modelsURL.path isDirectory:&isDirectory]);
    XCTAssertTrue(isDirectory);

    // Verify temp directory exists
    NSURL *tempURL = [self.cacheRootURL URLByAppendingPathComponent:@"temp"];
    XCTAssertTrue([self.fileManager fileExistsAtPath:tempURL.path isDirectory:&isDirectory]);
    XCTAssertTrue(isDirectory);

    // Verify version.txt exists
    NSURL *versionURL = [self.cacheRootURL URLByAppendingPathComponent:@"version.txt"];
    XCTAssertTrue([self.fileManager fileExistsAtPath:versionURL.path]);
}

- (void)testInitializationWithExistingDirectory {
    // Create a second cache pointing to the same directory
    ETCoreMLModelCache *cache2 = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:self.cacheRootURL];
    XCTAssertTrue(cache2.isReady);
    XCTAssertNil(cache2.initializationError);
}

#pragma mark - Store and Retrieve Tests

- (void)testStoreAndRetrieveModel {
    NSString *identifier = @"test_model_123";
    NSURL *sourceModelURL = [self createMockCompiledModelWithIdentifier:identifier];

    NSError *error = nil;
    NSURL *cachedURL = [self.cache storeModelAtURL:sourceModelURL withIdentifier:identifier error:&error];

    XCTAssertNotNil(cachedURL, @"Store should return cached URL");
    XCTAssertNil(error, @"Store should not produce error: %@", error);
    XCTAssertTrue([self.fileManager fileExistsAtPath:cachedURL.path], @"Cached model should exist");

    // Retrieve the model
    NSURL *retrievedURL = [self.cache cachedModelURLForIdentifier:identifier error:&error];

    XCTAssertNotNil(retrievedURL, @"Should retrieve cached model URL");
    XCTAssertNil(error, @"Retrieve should not produce error: %@", error);
    XCTAssertEqualObjects(cachedURL.path, retrievedURL.path, @"Retrieved URL should match stored URL");
}

- (void)testRetrieveNonExistentModelReturnsNil {
    NSError *error = nil;
    NSURL *retrievedURL = [self.cache cachedModelURLForIdentifier:@"nonexistent_model" error:&error];

    XCTAssertNil(retrievedURL, @"Should return nil for non-existent model");
    // Cache miss is not an error - just returns nil
    XCTAssertNil(error, @"Cache miss should not produce error");
}

- (void)testStoreModelOverwritesExisting {
    NSString *identifier = @"overwrite_test";

    // Store first version
    NSURL *sourceModelURL1 = [self createMockCompiledModelWithIdentifier:@"source1"];
    NSError *error = nil;
    NSURL *cachedURL1 = [self.cache storeModelAtURL:sourceModelURL1 withIdentifier:identifier error:&error];
    XCTAssertNotNil(cachedURL1);

    // Store second version with same identifier
    NSURL *sourceModelURL2 = [self createMockCompiledModelWithIdentifier:@"source2"];
    NSURL *cachedURL2 = [self.cache storeModelAtURL:sourceModelURL2 withIdentifier:identifier error:&error];
    XCTAssertNotNil(cachedURL2);

    // Retrieve should return the latest
    NSURL *retrievedURL = [self.cache cachedModelURLForIdentifier:identifier error:&error];
    XCTAssertNotNil(retrievedURL);
    XCTAssertTrue([self.fileManager fileExistsAtPath:retrievedURL.path]);

    // Verify content matches source2, not source1
    NSURL *modelFileURL = [retrievedURL URLByAppendingPathComponent:@"model.mil"];
    NSString *content = [NSString stringWithContentsOfURL:modelFileURL encoding:NSUTF8StringEncoding error:&error];
    XCTAssertNil(error, @"Should be able to read model.mil: %@", error);
    XCTAssertEqualObjects(content, @"mock model data for source2",
                          @"Retrieved model should contain content from the second store, not the first");
}

#pragma mark - Remove Tests

- (void)testRemoveCachedModel {
    NSString *identifier = @"model_to_remove";
    NSURL *sourceModelURL = [self createMockCompiledModelWithIdentifier:identifier];

    NSError *error = nil;
    [self.cache storeModelAtURL:sourceModelURL withIdentifier:identifier error:&error];
    XCTAssertNil(error);

    // Verify model exists
    NSURL *cachedURL = [self.cache cachedModelURLForIdentifier:identifier error:&error];
    XCTAssertNotNil(cachedURL);

    // Remove the model
    BOOL success = [self.cache removeCachedModelWithIdentifier:identifier error:&error];
    XCTAssertTrue(success, @"Remove should succeed");
    XCTAssertNil(error, @"Remove should not produce error: %@", error);

    // Verify model no longer exists
    NSURL *retrievedURL = [self.cache cachedModelURLForIdentifier:identifier error:&error];
    XCTAssertNil(retrievedURL, @"Model should no longer exist after removal");
}

- (void)testRemoveNonExistentModelSucceeds {
    NSError *error = nil;
    BOOL success = [self.cache removeCachedModelWithIdentifier:@"nonexistent" error:&error];

    // Removing non-existent model should succeed (idempotent)
    XCTAssertTrue(success, @"Remove should succeed even for non-existent model");
    XCTAssertNil(error);
}

#pragma mark - Purge Tests

- (void)testPurgeRemovesAllCachedModels {
    // Store multiple models
    NSArray *identifiers = @[@"model_a", @"model_b", @"model_c"];
    NSError *error = nil;

    for (NSString *identifier in identifiers) {
        NSURL *sourceURL = [self createMockCompiledModelWithIdentifier:identifier];
        [self.cache storeModelAtURL:sourceURL withIdentifier:identifier error:&error];
        XCTAssertNil(error);
    }

    // Verify all models exist
    for (NSString *identifier in identifiers) {
        NSURL *cachedURL = [self.cache cachedModelURLForIdentifier:identifier error:&error];
        XCTAssertNotNil(cachedURL, @"Model %@ should exist before purge", identifier);
    }

    // Purge the cache
    BOOL success = [self.cache purgeAndReturnError:&error];
    XCTAssertTrue(success, @"Purge should succeed");
    XCTAssertNil(error, @"Purge should not produce error: %@", error);

    // Verify all models are gone
    for (NSString *identifier in identifiers) {
        NSURL *cachedURL = [self.cache cachedModelURLForIdentifier:identifier error:&error];
        XCTAssertNil(cachedURL, @"Model %@ should not exist after purge", identifier);
    }

    // Cache should still be usable after purge
    XCTAssertTrue(self.cache.isReady);
}

#pragma mark - Temporary Directory Tests

- (void)testTemporaryDirectoryCreation {
    NSError *error = nil;

    NSURL *tempURL = [self.cache temporaryDirectoryWithError:&error];
    XCTAssertNotNil(tempURL, @"Should return temp URL");
    XCTAssertNil(error, @"Temp URL should not produce error: %@", error);

    // Verify temp URL is in the temp directory
    XCTAssertTrue([tempURL.path containsString:@"temp"], @"Temp URL should be in temp directory");

    // Verify directory was created
    BOOL isDirectory = NO;
    XCTAssertTrue([self.fileManager fileExistsAtPath:tempURL.path isDirectory:&isDirectory]);
    XCTAssertTrue(isDirectory, @"Temp URL should be a directory");
}

- (void)testMultipleTemporaryDirectoriesAreUnique {
    NSError *error = nil;

    NSURL *tempURL1 = [self.cache temporaryDirectoryWithError:&error];
    NSURL *tempURL2 = [self.cache temporaryDirectoryWithError:&error];

    XCTAssertNotNil(tempURL1);
    XCTAssertNotNil(tempURL2);
    XCTAssertNotEqualObjects(tempURL1.path, tempURL2.path, @"Each temp URL should be unique");
}

#pragma mark - Identifier Validation Tests

- (void)testInvalidIdentifierWithSlash {
    NSString *invalidIdentifier = @"path/to/model";
    NSURL *sourceURL = [self createMockCompiledModelWithIdentifier:@"valid"];

    NSError *error = nil;
    NSURL *cachedURL = [self.cache storeModelAtURL:sourceURL withIdentifier:invalidIdentifier error:&error];

    XCTAssertNil(cachedURL, @"Store should fail for identifier with slash");
    XCTAssertNotNil(error, @"Should produce error for invalid identifier");
    XCTAssertEqual(error.code, ETCoreMLModelCacheErrorCodeInvalidIdentifier);
}

- (void)testInvalidIdentifierWithDotDot {
    NSString *invalidIdentifier = @"model..name";
    NSURL *sourceURL = [self createMockCompiledModelWithIdentifier:@"valid"];

    NSError *error = nil;
    NSURL *cachedURL = [self.cache storeModelAtURL:sourceURL withIdentifier:invalidIdentifier error:&error];

    XCTAssertNil(cachedURL, @"Store should fail for identifier with ..");
    XCTAssertNotNil(error, @"Should produce error for invalid identifier");
    XCTAssertEqual(error.code, ETCoreMLModelCacheErrorCodeInvalidIdentifier);
}

- (void)testEmptyIdentifier {
    NSURL *sourceURL = [self createMockCompiledModelWithIdentifier:@"valid"];

    NSError *error = nil;
    NSURL *cachedURL = [self.cache storeModelAtURL:sourceURL withIdentifier:@"" error:&error];

    XCTAssertNil(cachedURL, @"Store should fail for empty identifier");
    XCTAssertNotNil(error, @"Should produce error for empty identifier");
    XCTAssertEqual(error.code, ETCoreMLModelCacheErrorCodeInvalidIdentifier);
}

#pragma mark - Source Not Found Tests

- (void)testStoreNonExistentSourceFails {
    NSURL *nonExistentURL = [self.testDirectoryURL URLByAppendingPathComponent:@"nonexistent.mlmodelc"];

    NSError *error = nil;
    NSURL *cachedURL = [self.cache storeModelAtURL:nonExistentURL withIdentifier:@"test" error:&error];

    XCTAssertNil(cachedURL, @"Store should fail for non-existent source");
    XCTAssertNotNil(error, @"Should produce error for non-existent source");
    XCTAssertEqual(error.code, ETCoreMLModelCacheErrorCodeSourceNotFound);
}

#pragma mark - Multiple Identifiers Tests

- (void)testMultipleModelsWithDifferentIdentifiers {
    NSError *error = nil;

    // Store model A
    NSURL *sourceA = [self createMockCompiledModelWithIdentifier:@"sourceA"];
    NSURL *cachedA = [self.cache storeModelAtURL:sourceA withIdentifier:@"model_A" error:&error];
    XCTAssertNotNil(cachedA);

    // Store model B
    NSURL *sourceB = [self createMockCompiledModelWithIdentifier:@"sourceB"];
    NSURL *cachedB = [self.cache storeModelAtURL:sourceB withIdentifier:@"model_B" error:&error];
    XCTAssertNotNil(cachedB);

    // Retrieve both and verify they're different
    NSURL *retrievedA = [self.cache cachedModelURLForIdentifier:@"model_A" error:&error];
    NSURL *retrievedB = [self.cache cachedModelURLForIdentifier:@"model_B" error:&error];

    XCTAssertNotNil(retrievedA);
    XCTAssertNotNil(retrievedB);
    XCTAssertNotEqualObjects(retrievedA.path, retrievedB.path);

    // Remove one, verify the other still exists
    [self.cache removeCachedModelWithIdentifier:@"model_A" error:&error];

    XCTAssertNil([self.cache cachedModelURLForIdentifier:@"model_A" error:&error]);
    XCTAssertNotNil([self.cache cachedModelURLForIdentifier:@"model_B" error:&error]);
}

#pragma mark - Access Time Tests

- (void)testAccessFileIsCreatedOnStore {
    NSString *identifier = @"access_test_model";
    NSURL *sourceModelURL = [self createMockCompiledModelWithIdentifier:identifier];

    NSError *error = nil;
    NSURL *cachedURL = [self.cache storeModelAtURL:sourceModelURL withIdentifier:identifier error:&error];
    XCTAssertNotNil(cachedURL);
    XCTAssertNil(error);

    // Verify .accessed file exists after store
    NSURL *modelsURL = [self.cacheRootURL URLByAppendingPathComponent:@"models"];
    NSString *accessedFilename = [identifier stringByAppendingString:@".accessed"];
    NSURL *accessedURL = [modelsURL URLByAppendingPathComponent:accessedFilename];

    XCTAssertTrue([self.fileManager fileExistsAtPath:accessedURL.path],
                  @"Accessed file should be created after store");
}

- (void)testAccessTimeUpdatedOnRetrieval {
    // Access time is updated on retrieval (cache hit during model load) to support
    // future LRU eviction. This is NOT called during predict — only during load.
    NSString *identifier = @"access_time_model";
    NSURL *sourceModelURL = [self createMockCompiledModelWithIdentifier:identifier];

    NSError *error = nil;
    [self.cache storeModelAtURL:sourceModelURL withIdentifier:identifier error:&error];
    XCTAssertNil(error);

    // Verify .accessed file exists after store
    NSURL *modelsURL = [self.cacheRootURL URLByAppendingPathComponent:@"models"];
    NSString *accessedFilename = [identifier stringByAppendingString:@".accessed"];
    NSURL *accessedURL = [modelsURL URLByAppendingPathComponent:accessedFilename];

    XCTAssertTrue([self.fileManager fileExistsAtPath:accessedURL.path],
                  @"Accessed file should be created after store");

    // Get initial modification time
    NSDictionary *initialAttrs = [self.fileManager attributesOfItemAtPath:accessedURL.path error:&error];
    XCTAssertNil(error);
    NSDate *initialModTime = initialAttrs[NSFileModificationDate];
    XCTAssertNotNil(initialModTime);

    // Wait a small amount to ensure time difference would be measurable
    [NSThread sleepForTimeInterval:1.0];

    // Retrieve the model (should update access time)
    NSURL *retrievedURL = [self.cache cachedModelURLForIdentifier:identifier error:&error];
    XCTAssertNotNil(retrievedURL);

    // Verify .accessed file WAS updated
    NSDictionary *newAttrs = [self.fileManager attributesOfItemAtPath:accessedURL.path error:&error];
    XCTAssertNil(error);
    NSDate *newModTime = newAttrs[NSFileModificationDate];
    XCTAssertNotNil(newModTime);

    XCTAssertEqual([newModTime compare:initialModTime], NSOrderedDescending,
                   @"Access time should be updated on retrieval (model load)");
}

#pragma mark - Multiple Cache Instances Tests

- (void)testMultipleCacheInstancesSameDirectory {
    // Create two cache instances pointing to the same directory
    ETCoreMLModelCache *cache1 = self.cache;
    ETCoreMLModelCache *cache2 = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:self.cacheRootURL];

    XCTAssertTrue(cache1.isReady);
    XCTAssertTrue(cache2.isReady);

    NSError *error = nil;

    // Store via cache1
    NSString *identifier1 = @"cache1_model";
    NSURL *source1 = [self createMockCompiledModelWithIdentifier:@"source1"];
    NSURL *cached1 = [cache1 storeModelAtURL:source1 withIdentifier:identifier1 error:&error];
    XCTAssertNotNil(cached1);
    XCTAssertNil(error);

    // Retrieve via cache2 (should see the model stored by cache1)
    NSURL *retrieved1 = [cache2 cachedModelURLForIdentifier:identifier1 error:&error];
    XCTAssertNotNil(retrieved1, @"Cache2 should be able to retrieve model stored by cache1");
    XCTAssertNil(error);

    // Store via cache2
    NSString *identifier2 = @"cache2_model";
    NSURL *source2 = [self createMockCompiledModelWithIdentifier:@"source2"];
    NSURL *cached2 = [cache2 storeModelAtURL:source2 withIdentifier:identifier2 error:&error];
    XCTAssertNotNil(cached2);
    XCTAssertNil(error);

    // Retrieve via cache1 (should see the model stored by cache2)
    NSURL *retrieved2 = [cache1 cachedModelURLForIdentifier:identifier2 error:&error];
    XCTAssertNotNil(retrieved2, @"Cache1 should be able to retrieve model stored by cache2");
    XCTAssertNil(error);

    // Both caches can see both models
    XCTAssertNotNil([cache1 cachedModelURLForIdentifier:identifier1 error:&error]);
    XCTAssertNotNil([cache1 cachedModelURLForIdentifier:identifier2 error:&error]);
    XCTAssertNotNil([cache2 cachedModelURLForIdentifier:identifier1 error:&error]);
    XCTAssertNotNil([cache2 cachedModelURLForIdentifier:identifier2 error:&error]);
}

- (void)testConcurrentStoresSameIdentifier {
    // Two caches storing the same identifier should both succeed (last writer wins)
    ETCoreMLModelCache *cache1 = self.cache;
    ETCoreMLModelCache *cache2 = [[ETCoreMLModelCache alloc] initWithCacheRootDirectory:self.cacheRootURL];

    NSString *identifier = @"contested_model";
    NSError *error = nil;

    // Store via cache1
    NSURL *source1 = [self createMockCompiledModelWithIdentifier:@"source1"];
    NSURL *cached1 = [cache1 storeModelAtURL:source1 withIdentifier:identifier error:&error];
    XCTAssertNotNil(cached1);

    // Store via cache2 with same identifier (should overwrite)
    NSURL *source2 = [self createMockCompiledModelWithIdentifier:@"source2"];
    NSURL *cached2 = [cache2 storeModelAtURL:source2 withIdentifier:identifier error:&error];
    XCTAssertNotNil(cached2);

    // Both caches should be able to retrieve (they'll get the same model - last writer)
    NSURL *retrieved1 = [cache1 cachedModelURLForIdentifier:identifier error:&error];
    NSURL *retrieved2 = [cache2 cachedModelURLForIdentifier:identifier error:&error];

    XCTAssertNotNil(retrieved1);
    XCTAssertNotNil(retrieved2);
    XCTAssertEqualObjects(retrieved1.path, retrieved2.path, @"Both caches should retrieve the same model");
}

@end
