//
//  ETCoreMLDefaultModelExecutor.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLAsset.h"
#import "ETCoreMLDefaultModelExecutor.h"
#import "ETCoreMLLogging.h"
#import "ETCoreMLModel.h"

@implementation ETCoreMLDefaultModelExecutor

- (instancetype)initWithModel:(ETCoreMLModel *)model {
    self = [super init];
    if (self) {
        _model = model;
    }
    
    return self;
}

- (nullable NSArray<MLMultiArray *> *)executeModelWithInputs:(id<MLFeatureProvider>)inputs
                                           predictionOptions:(MLPredictionOptions *)predictionOptions
                                              loggingOptions:(const executorchcoreml::ModelLoggingOptions& __unused)loggingOptions
                                                 eventLogger:(const executorchcoreml::ModelEventLogger* _Nullable __unused)eventLogger
                                                       error:(NSError * __autoreleasing *)error {
    if (self.ignoreOutputBackings) {
        if (@available(iOS 16.0, tvOS 16.0, watchOS 9.0, *)) {
            predictionOptions.outputBackings = @{};
        }
    }

    id<MLFeatureProvider> outputs = [self.model predictionFromFeatures:inputs
                                                               options:predictionOptions
                                                                 error:error];
    if (!outputs) {
        return nil;
    }
    
    NSOrderedSet<NSString*>* orderedOutputNames = self.model.orderedOutputNames;
    NSMutableArray<MLMultiArray *> *result = [NSMutableArray arrayWithCapacity:orderedOutputNames.count];
    for (NSString *outputName in orderedOutputNames) {
        MLFeatureValue *featureValue = [outputs featureValueForName:outputName];
        if (!featureValue.multiArrayValue) {
            ETCoreMLLogErrorAndSetNSError(error,
                                          ETCoreMLErrorBrokenModel,
                                          "Model is broken, expected multiarray for output=%@.",
                                          outputName);
            return nil;
        }
        
        [result addObject:featureValue.multiArrayValue];
    }
    
    return result;
}

@end
