//
// ETCoreMLModelLoader.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLModelLoader.h"

#import "asset.h"
#import "ETCoreMLAsset.h"
#import "ETCoreMLAssetManager.h"
#import "ETCoreMLDefaultModelExecutor.h"
#import "ETCoreMLLogging.h"
#import "ETCoreMLModel.h"
#import "model_metadata.h"

using namespace executorchcoreml;

namespace {
    NSOrderedSet<NSString *> *get_ordered_set(const std::vector<std::string>& values) {
        NSMutableOrderedSet<NSString *> *result = [NSMutableOrderedSet orderedSetWithCapacity:values.size()];
        for (const auto& value : values) {
            [result addObject:@(value.c_str())];
        }
        
        return result;
    }
} // namespace

@implementation ETCoreMLModelLoader

+ (nullable ETCoreMLModel *)loadModelWithCompiledAsset:(ETCoreMLAsset *)compiledAsset
                                          configuration:(MLModelConfiguration *)configuration
                                               metadata:(const executorchcoreml::ModelMetadata&)metadata
                                                error:(NSError * __autoreleasing *)error {
    if (compiledAsset == nil) {
        return nil;
    }
    
    // Use the metadata's ordered input/output names.
    // For multifunction models, the caller should load the per-method metadata
    // which contains the correct input/output names for that method.
    NSOrderedSet<NSString *> *orderedInputNames = ::get_ordered_set(metadata.input_names);
    NSOrderedSet<NSString *> *orderedOutputNames = ::get_ordered_set(metadata.output_names);
    
    NSError *localError = nil;
    ETCoreMLModel *model = [[ETCoreMLModel alloc] initWithAsset:compiledAsset
                                                  configuration:configuration
                                              orderedInputNames:orderedInputNames
                                             orderedOutputNames:orderedOutputNames
                                                          error:&localError];
    if (model) {
        return model;
    }
    
    if (error) {
        *error = localError;
    }
    
    return nil;
}
                                        

+ (nullable ETCoreMLModel *)loadModelWithContentsOfURL:(NSURL *)compiledModelURL
                                         configuration:(MLModelConfiguration *)configuration
                                              metadata:(const executorchcoreml::ModelMetadata&)metadata
                                          assetManager:(ETCoreMLAssetManager *)assetManager
                                                 error:(NSError * __autoreleasing *)error {
    NSError *localError = nil;
    NSString *identifier = @(metadata.identifier.c_str());
    ETCoreMLAsset *asset = nil;
    if ([assetManager hasAssetWithIdentifier:identifier error:&localError]) {
        asset = [assetManager assetWithIdentifier:identifier error:&localError];
    } else {
        asset = [assetManager storeAssetAtURL:compiledModelURL withIdentifier:identifier error:&localError];
    }
    
    ETCoreMLModel *model;
    if (asset != nil) {
        model = [self loadModelWithCompiledAsset:asset configuration:configuration metadata:metadata error:&localError];
    } else {
        model = nil;
    }

    if (model) {
        return model;
    }

    if (error) {
        *error = localError;
    }

    return nil;
}

@end
