//
// ETCoreMLAssetTests.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <XCTest/XCTest.h>

#import <ETCoreMLAsset.h>

@interface ETCoreMLAssetTests : XCTestCase

@property (copy, nonatomic, nullable) NSURL *assetURL;
@property (strong, nonatomic, nullable) NSFileManager *fileManager;

@end

@implementation ETCoreMLAssetTests

using namespace executorchcoreml;

- (void)setUp {
    NSURL *assetURL = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:[NSUUID UUID].UUIDString];
    NSFileManager *fileManager = [[NSFileManager alloc] init];
    NSError *localError = nil;
    XCTAssertTrue([fileManager createDirectoryAtURL:assetURL withIntermediateDirectories:NO attributes:@{} error:&localError]);
    // Write dir1
    {
        NSURL *dirURL = [assetURL URLByAppendingPathComponent:@"dir1"];
        XCTAssertTrue([fileManager createDirectoryAtURL:dirURL withIntermediateDirectories:NO attributes:@{} error:&localError]);
        NSURL *fileURL = [dirURL URLByAppendingPathComponent:@"content.txt"];
        NSString *testData = @"data1";
        NSData* data = [testData dataUsingEncoding:NSUTF8StringEncoding];
        XCTAssertTrue([data writeToURL:fileURL atomically:YES]);
    }
    // Write dir2
    {
        NSURL *dirURL = [assetURL URLByAppendingPathComponent:@"dir2"];
        XCTAssertTrue([fileManager createDirectoryAtURL:dirURL withIntermediateDirectories:NO attributes:@{} error:&localError]);
        NSURL *fileURL = [dirURL URLByAppendingPathComponent:@"content.txt"];
        NSString *testData = @"data2";
        NSData* data = [testData dataUsingEncoding:NSUTF8StringEncoding];
        XCTAssertTrue([data writeToURL:fileURL atomically:YES]);
    }
    
    self.assetURL = assetURL;
    self.fileManager = fileManager;
}

- (void)tearDown {
    NSError *localError = nil;
    XCTAssertTrue([self.fileManager removeItemAtURL:self.assetURL error:&localError]);
    self.assetURL = nil;
    self.fileManager = nil;
}

- (void)testAssetCreation {
    NSString *identifier = [NSUUID UUID].UUIDString;
    NSError *localError = nil;
    auto backingAsset = Asset::make(self.assetURL, identifier, self.fileManager, &localError);
    XCTAssertTrue(backingAsset.has_value());
    const auto& packageInfo = backingAsset.value().package_info;
    XCTAssertEqual(packageInfo.file_infos.size(), 2);
    const auto& fileInfos = packageInfo.file_infos;
    XCTAssert(std::find_if(fileInfos.begin(), fileInfos.end(), [](const auto& fileInfo) {
        return fileInfo.relative_path == "dir1/content.txt";
    }) != packageInfo.file_infos.end());
    
    XCTAssert(std::find_if(fileInfos.begin(), fileInfos.end(), [](const auto& fileInfo) {
        return fileInfo.relative_path == "dir2/content.txt";
    }) != packageInfo.file_infos.end());

    ETCoreMLAsset *asset = [[ETCoreMLAsset alloc] initWithBackingAsset:backingAsset.value()];
    XCTAssertTrue([asset.identifier isEqualToString:identifier]);
}

- (void)testAssetValidity {
    NSString *identifier = [NSUUID UUID].UUIDString;
    NSError *localError = nil;
    auto backingAsset = Asset::make(self.assetURL, identifier, self.fileManager, &localError);
    const auto& backingAssetValue = backingAsset.value();
    {
        ETCoreMLAsset *asset = [[ETCoreMLAsset alloc] initWithBackingAsset:backingAsset.value()];
        XCTAssertTrue(asset.isValid);
    }
    {
        const auto& packageInfo = backingAssetValue.package_info;
        const auto& fileInfos = packageInfo.file_infos;
        NSURL *dirURL = [NSURL fileURLWithPath:@(backingAssetValue.path.c_str())];
        NSURL *fileURL = [dirURL URLByAppendingPathComponent:@(fileInfos[0].relative_path.c_str())];
        // Delete a file, asset must not be valid.
        XCTAssert([self.fileManager removeItemAtURL:fileURL error:&localError]);
        ETCoreMLAsset *asset = [[ETCoreMLAsset alloc] initWithBackingAsset:backingAsset.value()];
        XCTAssertFalse(asset.isValid);
    }
}

- (void)testKeepAlive {
    NSString *identifier = [NSUUID UUID].UUIDString;
    NSError *localError = nil;
    auto backingAsset = Asset::make(self.assetURL, identifier, self.fileManager, &localError);
    XCTAssertTrue(backingAsset.has_value());
    ETCoreMLAsset *asset = [[ETCoreMLAsset alloc] initWithBackingAsset:backingAsset.value()];
    XCTAssertTrue([asset keepAliveAndReturnError:&localError]);
}

- (void)testPewarm {
    NSString *identifier = [NSUUID UUID].UUIDString;
    NSError *localError = nil;
    auto backingAsset = Asset::make(self.assetURL, identifier, self.fileManager, &localError);
    XCTAssertTrue(backingAsset.has_value());
    ETCoreMLAsset *asset = [[ETCoreMLAsset alloc] initWithBackingAsset:backingAsset.value()];
    XCTAssertTrue([asset prewarmAndReturnError:&localError]);
}

- (void)testClose {
    NSString *identifier = [NSUUID UUID].UUIDString;
    NSError *localError = nil;
    auto backingAsset = Asset::make(self.assetURL, identifier, self.fileManager, &localError);
    XCTAssertTrue(backingAsset.has_value());
    ETCoreMLAsset *asset = [[ETCoreMLAsset alloc] initWithBackingAsset:backingAsset.value()];
    XCTAssertTrue([asset keepAliveAndReturnError:&localError]);
    XCTAssertTrue(asset.isAlive);
    [asset close];
    XCTAssertFalse(asset.isAlive);
}

@end
