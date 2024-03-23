//
// ETCoreMLModelLoader.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <ETCoreMLAsset.h>
#import <ETCoreMLAssetManager.h>
#import <ETCoreMLDefaultModelExecutor.h>
#import <ETCoreMLLogging.h>
#import <ETCoreMLModel.h>
#import <ETCoreMLModelLoader.h>
#import <asset.h>
#import <model_metadata.h>

using namespace executorchcoreml;

namespace {
    NSOrderedSet<NSString *> *get_ordered_set(const std::vector<std::string>& values) {
        NSMutableOrderedSet<NSString *> *result = [NSMutableOrderedSet orderedSetWithCapacity:values.size()];
        for (const auto& value : values) {
            [result addObject:@(value.c_str())];
        }
        
        return result;
    }

    ETCoreMLModel * _Nullable get_model_from_asset(ETCoreMLAsset *asset,
                                                   MLModelConfiguration *configuration,
                                                   const executorchcoreml::ModelMetadata& metadata,
                                                   NSError * __autoreleasing *error) {
        NSOrderedSet<NSString *> *orderedInputNames = ::get_ordered_set(metadata.input_names);
        NSOrderedSet<NSString *> *orderedOutputNames = ::get_ordered_set(metadata.output_names);
        ETCoreMLModel *model = [[ETCoreMLModel alloc] initWithAsset:asset
                                                      configuration:configuration
                                                  orderedInputNames:orderedInputNames
                                                 orderedOutputNames:orderedOutputNames
                                                              error:error];
        return model;
    }
} // namespace

@implementation ETCoreMLModelLoader

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
    
    ETCoreMLModel *model = (asset != nil) ? get_model_from_asset(asset, configuration, metadata, &localError) : nil;
    if (model) {
        return model;
    }
    
    if (localError) {
        ETCoreMLLogError(localError,
                         "%@: Failed to load model from compiled asset with identifier = %@",
                         NSStringFromClass(ETCoreMLModelLoader.class),
                         identifier);
    }
    
    // If store failed then we will load the model from compiledURL.
    auto backingAsset = Asset::make(compiledModelURL, identifier, assetManager.fileManager, error);
    if (!backingAsset) {
        return nil;
    }
    
    asset = [[ETCoreMLAsset alloc] initWithBackingAsset:backingAsset.value()];
    return ::get_model_from_asset(asset, configuration, metadata, error);
}

@end
