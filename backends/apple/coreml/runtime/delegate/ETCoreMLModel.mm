//
// ETCoreMLModel.mm
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <ETCoreMLModel.h>

#import <ETCoreMLAsset.h>

@implementation ETCoreMLModel

- (nullable instancetype)initWithAsset:(ETCoreMLAsset *)asset
                         configuration:(MLModelConfiguration *)configuration
                     orderedInputNames:(NSOrderedSet<NSString *> *)orderedInputNames
                    orderedOutputNames:(NSOrderedSet<NSString *> *)orderedOutputNames
                                 error:(NSError * __autoreleasing *)error {
    if (![asset keepAliveAndReturnError:error]) {
        return nil;
    }
    
    MLModel *mlModel = [MLModel modelWithContentsOfURL:asset.contentURL
                                         configuration:configuration
                                                 error:error];
    if (!mlModel) {
        return nil;
    }
    
    self = [super init];
    if (self) {
        _mlModel = mlModel;
        _asset = asset;
        _orderedInputNames = [orderedInputNames copy];
        _orderedOutputNames = [orderedOutputNames copy];
    }

    return self;
}

- (NSString *)identifier {
    return self.asset.identifier;
}

@end
