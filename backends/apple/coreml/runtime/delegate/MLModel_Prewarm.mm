//
// MLModel+Prewarm.m
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <MLModel_Prewarm.h>

#import <algorithm>

@interface MLMultiArray (PreHeat)

+ (nullable MLMultiArray *)zeroedMultiArrayWithShape:(NSArray<NSNumber *> *)shape
                                            dataType:(MLMultiArrayDataType)dataType
                                               error:(NSError * __autoreleasing *)error;

@end


@implementation MLMultiArray (PreHeat)

+ (MLMultiArray *)zeroedMultiArrayWithShape:(NSArray<NSNumber *> *)shape
                                   dataType:(MLMultiArrayDataType)dataType
                                      error:(NSError * __autoreleasing *)error {
    MLMultiArray *multiArray = [[MLMultiArray alloc] initWithShape:shape dataType:dataType error:error];
    if (!multiArray) {
        return nil;
    }
    
    [multiArray getMutableBytesWithHandler:^(void *mutableBytes, NSInteger size, NSArray<NSNumber *> * __unused strides) {
        uint8_t *start = reinterpret_cast<uint8_t *>(mutableBytes);
        uint8_t *end = start + size;
        std::fill(start, end, uint8_t(0));
    }];
    
    return multiArray;
}

@end

namespace {

id<MLFeatureProvider> _Nullable get_zeroed_inputs(MLModel *model, NSError * __autoreleasing *error) {
    NSMutableDictionary<NSString *, MLFeatureValue *> *inputs = [NSMutableDictionary dictionary];
    for (MLFeatureDescription *feature_desc in model.modelDescription.inputDescriptionsByName.allValues) {
        switch (feature_desc.type) {
            case MLFeatureTypeMultiArray: {
                MLMultiArrayConstraint *constraint = feature_desc.multiArrayConstraint;
                MLMultiArray *array = [MLMultiArray zeroedMultiArrayWithShape:constraint.shape 
                                                                     dataType:constraint.dataType
                                                                        error:error];
                MLFeatureValue *feature = (array != nil) ? [MLFeatureValue featureValueWithMultiArray:array] : nil;
                if (!feature) {
                    return nil;
                }
                inputs[feature_desc.name] = feature;
                break;
            }
                
            default: {
                return nil;
            }
        }
    }
    
    return [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs error:error];
}

} //namespace

@implementation MLModel (Prewarm)

- (BOOL)prewarmAndReturnError:(NSError * __autoreleasing *)error {
    @autoreleasepool {
        id<MLFeatureProvider> inputs = ::get_zeroed_inputs(self, error);
        if (!inputs) {
            return NO;
        }
        
        id<MLFeatureProvider> outputs = [self predictionFromFeatures:inputs error:error];
        return outputs != nil;
    }
}

@end
