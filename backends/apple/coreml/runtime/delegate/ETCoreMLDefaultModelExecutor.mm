//
//  ETCoreMLDefaultModelExecutor.m
//  executorchcoreml_tests
//
//  Created by Gyan Sinha on 2/25/24.
//

#import <ETCoreMLAsset.h>
#import <ETCoreMLDefaultModelExecutor.h>
#import <ETCoreMLLogging.h>
#import <ETCoreMLModel.h>

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
        predictionOptions.outputBackings = @{};
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
                                          "%@: Model is broken, expected multiarray for output=%@.",
                                          NSStringFromClass(self.class),
                                          outputName);
            return nil;
        }
        
        [result addObject:featureValue.multiArrayValue];
    }
    
    return result;
}

@end
