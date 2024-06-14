//
// ETCoreMLAssetManagerTests.mm
//
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLTestUtils.h"
#import <ETCoreMLAsset.h>
#import <ETCoreMLAssetManager.h>
#import <XCTest/XCTest.h>
#import <executorch/runtime/platform/runtime.h>

@interface ETCoreMLAssetManagerTests : XCTestCase

@property (strong, nonatomic, nullable) ETCoreMLAssetManager *assetManager;
@property (strong, nonatomic, nullable) NSFileManager *fileManager;
@property (copy, nonatomic) NSURL *testDirectoryURL;

@end

@implementation ETCoreMLAssetManagerTests

+ (void)setUp {
    torch::executor::runtime_init();
}

- (void)setUp {
    @autoreleasepool {
        NSError *localError = nil;
        self.fileManager = [[NSFileManager alloc] init];
        self.testDirectoryURL = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:[NSUUID UUID].UUIDString]];
        self.assetManager = [ETCoreMLTestUtils createAssetManagerWithURL:self.testDirectoryURL error:&localError];
        XCTAssertNotNil(self.assetManager);
    }
}

- (void)tearDown {
    @autoreleasepool {
        NSError *localError = nil;
        self.assetManager = nil;
        [self.fileManager removeItemAtURL:self.testDirectoryURL error:&localError];
        self.fileManager = nil;
    }
}

- (void)testAssetManagerCreation {
    XCTAssertNotNil(self.assetManager);
}

- (void)testPutAsset {
    NSUInteger n = 5;
    NSError *localError = nil;
    for (NSUInteger i = 0; i < n; i++) {
        NSString *identifier = [NSUUID UUID].UUIDString;
        NSURL *assetURL = [ETCoreMLTestUtils createUniqueAssetInDirectoryAtURL:self.testDirectoryURL withContent:@"testing" fileManager:self.fileManager error:&localError];
        XCTAssertNotNil(assetURL);
        XCTAssertTrue([self.assetManager storeAssetAtURL:assetURL withIdentifier:identifier error:&localError]);
        XCTAssertTrue([self.assetManager hasAssetWithIdentifier:identifier error:&localError]);
    }
}

- (void)testGetAsset {
    NSUInteger n = 5;
    NSError *localError = nil;
    for (NSUInteger i = 0; i < n; i++) {
        NSString *identifier = [NSUUID UUID].UUIDString;
        NSURL *assetURL = [ETCoreMLTestUtils createUniqueAssetInDirectoryAtURL:self.testDirectoryURL withContent:@"testing" fileManager:self.fileManager error:&localError];
        XCTAssertNotNil(assetURL);
        XCTAssertTrue([self.assetManager storeAssetAtURL:assetURL withIdentifier:identifier error:&localError]);
        XCTAssertTrue([[self.assetManager assetWithIdentifier:identifier error:&localError].identifier isEqualToString:identifier]);
    }
}

- (void)testRemoveAsset {
    NSUInteger n = 5;
    NSError *localError = nil;
    NSMutableArray<NSString *> *identifiers = [NSMutableArray arrayWithCapacity:n];
    @autoreleasepool {
        for (NSUInteger i = 0; i < n; i++) {
            NSString *identifier = [NSUUID UUID].UUIDString;
            NSURL *assetURL = [ETCoreMLTestUtils createUniqueAssetInDirectoryAtURL:self.testDirectoryURL withContent:@"testing" fileManager:self.fileManager error:&localError];
            XCTAssertNotNil(assetURL);
            ETCoreMLAsset *asset = [self.assetManager storeAssetAtURL:assetURL withIdentifier:identifier error:&localError];
            (void)asset;
            // The asset is alive, it must not be removed.
            XCTAssertFalse([self.assetManager removeAssetWithIdentifier:identifier error:&localError]);
            [identifiers addObject:identifier];
        }
    }
    
    // The asset will only be removed if it's not in use.
    for (NSString *identifier in identifiers) {
        XCTAssertTrue([self.assetManager hasAssetWithIdentifier:identifier error:&localError]);
        XCTAssertTrue([self.assetManager removeAssetWithIdentifier:identifier error:&localError]);
        XCTAssertFalse([self.assetManager hasAssetWithIdentifier:identifier error:&localError]);
    }
}

- (void)testEstimatedSizeInBytes {
    NSUInteger n = 5;
    NSError *localError = nil;
    for (NSUInteger i = 0; i < n; i++) {
        NSString *identifier = [NSUUID UUID].UUIDString;
        NSURL *assetURL = [ETCoreMLTestUtils createUniqueAssetInDirectoryAtURL:self.testDirectoryURL withContent:@"testing" fileManager:self.fileManager error:&localError];
        XCTAssertNotNil(assetURL);
        XCTAssertNotNil([self.assetManager storeAssetAtURL:assetURL withIdentifier:identifier error:&localError]);
        XCTAssertTrue(self.assetManager.estimatedSizeInBytes > 0);
    }

    for (NSUInteger i = 0; i < n; i++) {
        NSString *identifier = [NSUUID UUID].UUIDString;
        NSURL *assetURL = [ETCoreMLTestUtils createUniqueAssetInDirectoryAtURL:self.testDirectoryURL withContent:@"testing" fileManager:self.fileManager error:&localError];
        XCTAssertNotNil(assetURL);
        NSInteger oldSize = self.assetManager.estimatedSizeInBytes;
        ETCoreMLAsset *asset = [self.assetManager storeAssetAtURL:assetURL withIdentifier:identifier error:&localError];
        NSInteger newSize = asset.totalSizeInBytes + oldSize;
        XCTAssertEqual(newSize, self.assetManager.estimatedSizeInBytes);
        [asset close];
        XCTAssertTrue([self.assetManager removeAssetWithIdentifier:identifier error:&localError]);
        XCTAssertEqual(oldSize, self.assetManager.estimatedSizeInBytes);
    }
}

- (void)testCompaction {
    NSUInteger n = 5;
    NSError *localError = nil;
    for (NSUInteger i = 0; i < n; i++) {
        NSString *identifier = [NSUUID UUID].UUIDString;
        NSURL *assetURL = [ETCoreMLTestUtils createUniqueAssetInDirectoryAtURL:self.testDirectoryURL withContent:@"testing" fileManager:self.fileManager error:&localError];
        XCTAssertNotNil(assetURL);
        ETCoreMLAsset *asset = [self.assetManager storeAssetAtURL:assetURL withIdentifier:identifier error:&localError];
        // Close the asset so that it could be deleted.
        [asset close];
    }
    XCTAssertEqual([self.assetManager compact:100 error:&localError], 0);
}

- (void)testPurge {
    NSUInteger n = 5;
    NSError *localError = nil;
    for (NSUInteger i = 0; i < n; i++) {
        NSString *identifier = [NSUUID UUID].UUIDString;
        NSURL *assetURL = [ETCoreMLTestUtils createUniqueAssetInDirectoryAtURL:self.testDirectoryURL withContent:@"testing" fileManager:self.fileManager error:&localError];
        XCTAssertNotNil(assetURL);
        ETCoreMLAsset *asset = [self.assetManager storeAssetAtURL:assetURL withIdentifier:identifier error:&localError];
        // Close the asset so that it could be deleted.
        [asset close];
    }
    XCTAssertTrue([self.assetManager purgeAndReturnError:&localError]);
}

@end
