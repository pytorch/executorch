//
// ETCoreMLModel.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <ETCoreMLModel.h>

#import "ETCoreMLAsset.h"
#import "ETCoreMLLogging.h"
#import "multiarray.h"
#import "objc_array_util.h"
#import "MLModel_Prewarm.h"
#import <functional>
#import <numeric>

#pragma mark - ETCoreMLMultiArrayDescriptor
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLMultiArrayDescriptor: NSObject <NSCopying>

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)initWithShape:(NSArray<NSNumber *> *)shape
                     dataType:(MLMultiArrayDataType)dataType NS_DESIGNATED_INITIALIZER;

@property (copy, readonly, nonatomic) NSArray<NSNumber *> *shape;

@property (assign, readonly, nonatomic) MLMultiArrayDataType dataType;

@end

@implementation ETCoreMLMultiArrayDescriptor

- (instancetype)initWithShape:(NSArray<NSNumber *> *)shape
                     dataType:(MLMultiArrayDataType)dataType {
    self = [super init];
    if (self) {
        _shape = shape;
        _dataType = dataType;
    }
    
    return self;
}

- (BOOL)isEqual:(id)object {
    if (object == self) {
        return YES;
    }
    
    if (![object isKindOfClass:self.class]) {
        return NO;
    }
    
    ETCoreMLMultiArrayDescriptor *other = (ETCoreMLMultiArrayDescriptor *)object;
    return [self.shape isEqualToArray:other.shape] && self.dataType == other.dataType;
}

- (NSUInteger)hash {
    return [self.shape hash] ^ (NSUInteger)self.dataType;
}

- (instancetype)copyWithZone:(NSZone *)zone {
    return [[ETCoreMLMultiArrayDescriptor allocWithZone:zone] initWithShape:self.shape
                                                                   dataType:self.dataType];
}

@end

namespace {

using namespace executorchcoreml;

size_t get_number_of_bytes(MLMultiArrayDataType data_type) {
    switch (data_type) {
        case MLMultiArrayDataTypeFloat16: {
            return 2;
        }
        case MLMultiArrayDataTypeFloat32: {
            return 4;
        }
        case MLMultiArrayDataTypeInt32: {
            return 4;
        }
        case MLMultiArrayDataTypeFloat64: {
            return 8;
        }
        default: {
            return 0;
        }
    }
}

std::vector<size_t> calculate_strides(const std::vector<size_t>& shape) {
    if (shape.size() == 0) {
        return {};
    }
    
    if (shape.size() == 1) {
        return {1};
    }
    
    std::vector<size_t> strides(shape.size(), 1);
    size_t product = 1;
    for (size_t i = shape.size(); i > 0; i--) {
        strides[i - 1] = product;
        product *= shape[i - 1];
    }
    
    return strides;
}

MLMultiArray * _Nullable make_ml_multi_array(const std::vector<size_t>& shape,
                                             MLMultiArrayDataType dataType,
                                             NSCache<ETCoreMLMultiArrayDescriptor *, NSMutableData *> *cache,
                                             NSError * __autoreleasing *error) {
    ETCoreMLMultiArrayDescriptor *descriptor = [[ETCoreMLMultiArrayDescriptor alloc] initWithShape:to_array(shape)
                                                                                          dataType:dataType];
    // Check the cache first otherwise allocate a new backing storage.
    NSMutableData *backing_storage = [cache objectForKey:descriptor];
    if (backing_storage) {
        [cache removeObjectForKey:descriptor];
    } else {
        size_t n = std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<size_t>{});
        backing_storage = [[NSMutableData alloc] initWithLength:n * get_number_of_bytes(dataType)];
    }
    
    __weak NSCache<ETCoreMLMultiArrayDescriptor *, NSMutableData *> *weakCache = cache;
    // Add the storage back to the cache when it gets deallocated, the next prediction would use the same storage.
    MLMultiArray *result = [[MLMultiArray alloc] initWithDataPointer:backing_storage.mutableBytes
                                                               shape:descriptor.shape
                                                            dataType:descriptor.dataType
                                                             strides:to_array(calculate_strides(shape))
                                                         deallocator:^(void * _Nonnull bytes) {[weakCache setObject:backing_storage forKey:descriptor];}
                                                               error:error];
    
    return result;
}

NSDictionary<NSString *, MLMultiArrayConstraint *> *
get_multi_array_constraints_by_name(NSDictionary<NSString *, MLFeatureDescription *> *feature_descriptions) {
    NSMutableDictionary<NSString *, MLMultiArrayConstraint *> *result = [NSMutableDictionary dictionaryWithCapacity:feature_descriptions.count];
    [feature_descriptions enumerateKeysAndObjectsUsingBlock:^(NSString *key, MLFeatureDescription *description, BOOL * _Nonnull stop) {
        result[key] = description.multiArrayConstraint;
    }];
    
    return result;
}

NSDictionary<NSString *, MLMultiArrayConstraint *> *get_multi_array_input_constraints_by_name(MLModelDescription *description) {
    return get_multi_array_constraints_by_name(description.inputDescriptionsByName);
}

NSDictionary<NSString *, MLMultiArrayConstraint *> *get_multi_array_output_constraints_by_name(MLModelDescription *description) {
    return get_multi_array_constraints_by_name(description.outputDescriptionsByName);
}

#if MODEL_STATE_IS_SUPPORTED
API_AVAILABLE(macos(15.0), ios(18.0), tvos(18.0), watchos(11.0))
void reset_state_for_feature_name(NSString *feature_name, MLState *state) {
    [state getMultiArrayForStateNamed:feature_name handler:^(MLMultiArray *buffer) {
        [buffer getMutableBytesWithHandler:^(void *mutableBytes, NSInteger size, NSArray<NSNumber *> * __unused strides) {
            uint8_t *start = reinterpret_cast<uint8_t *>(mutableBytes);
            uint8_t *end = start + size;
            std::fill(start, end, uint8_t(0));
        }];
    }];
}
#endif

}

#pragma mark - ETCoreMLModel
@interface ETCoreMLModel ()

@property (strong, readonly, nonatomic) NSCache<ETCoreMLMultiArrayDescriptor *, NSMutableData *> *cache;
@property (copy, readonly, nonatomic) NSDictionary<NSString *, MLMultiArrayConstraint *> *inputConstraintsByName;
@property (copy, readonly, nonatomic) NSDictionary<NSString *, MLMultiArrayConstraint *> *outputConstraintsByName;

@end


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
        _cache = [[NSCache alloc] init];
        _inputConstraintsByName = get_multi_array_input_constraints_by_name(mlModel.modelDescription);
        _outputConstraintsByName = get_multi_array_output_constraints_by_name(mlModel.modelDescription);
#if MODEL_STATE_IS_SUPPORTED
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)) {
            _state = mlModel.modelDescription.stateDescriptionsByName.count > 0 ? [_mlModel newState] : nil;
        }
#endif
    }
    
    return self;
}

- (NSString *)identifier {
    return self.asset.identifier;
}

- (nullable NSArray<MLMultiArray *> *)prepareArgs:(const std::vector<executorchcoreml::MultiArray>&)args
                                         argNames:(NSOrderedSet<NSString *> *)argNames
                             argConstraintsByName:(NSDictionary<NSString *, MLMultiArrayConstraint *> *)argConstraintsByName
                                         copyData:(const BOOL)copyData
                                            error:(NSError * __autoreleasing *)error {
    NSEnumerator *nameEnumerator = [argNames objectEnumerator];
    NSMutableArray<MLMultiArray *> *result = [NSMutableArray arrayWithCapacity:args.size()];
    for (const auto& arg : args) {
        BOOL lCopyData = copyData;
        NSString *argName = [nameEnumerator nextObject];
        MLMultiArrayConstraint *constraint = argConstraintsByName[argName];
        const auto& layout = arg.layout();
        auto dataType = to_ml_multiarray_data_type(layout.dataType());
        MLMultiArray *multiArrayArg = nil;
        if (dataType == constraint.dataType) {
            // We can use the same data storage.
            multiArrayArg = [[MLMultiArray alloc] initWithDataPointer:arg.data()
                                                                shape:to_array(layout.shape())
                                                             dataType:constraint.dataType
                                                              strides:to_array(layout.strides())
                                                          deallocator:^(void * _Nonnull bytes) {}
                                                                error:error];
            lCopyData = NO;
        } else {
            // We can't use the same data storage, data types are not the same.
            multiArrayArg = ::make_ml_multi_array(layout.shape(), constraint.dataType, self.cache, error);
        }
        
        if (!multiArrayArg) {
            return nil;
        }
        
        if (multiArrayArg && lCopyData) {
            [multiArrayArg getMutableBytesWithHandler:^(void *_Nonnull mutableBytes,
                                                        NSInteger __unused size,
                                                        NSArray<NSNumber *> *strides) {
                MultiArray buffer(mutableBytes, MultiArray::MemoryLayout(to_multiarray_data_type(constraint.dataType).value(),
                                                                         layout.shape(),
                                                                         to_vector<ssize_t>(strides)));
                arg.copy(buffer);
            }];
        }
        
        [result addObject:multiArrayArg];
    }
    
    return result;
}

- (nullable NSArray<MLMultiArray *> *)prepareInputs:(const std::vector<executorchcoreml::MultiArray>&)inputs
                                              error:(NSError * __autoreleasing *)error {
    return [self prepareArgs:inputs
                    argNames:self.orderedInputNames
        argConstraintsByName:self.inputConstraintsByName
                    copyData:YES
                       error:error];
    
}

- (nullable NSArray<MLMultiArray *> *)prepareOutputBackings:(const std::vector<executorchcoreml::MultiArray>&)outputs
                                                      error:(NSError * __autoreleasing *)error {
    return [self prepareArgs:outputs
                    argNames:self.orderedOutputNames
        argConstraintsByName:self.outputConstraintsByName
                    copyData:NO
                       error:error];
    
}

- (nullable id<MLFeatureProvider>)predictionFromFeatures:(id<MLFeatureProvider>)input
                                                 options:(MLPredictionOptions *)options
                                                   error:(NSError **)error {
#if MODEL_STATE_IS_SUPPORTED
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)) {
        if (self.state != nil) {
            return [self.mlModel predictionFromFeatures:input
                                             usingState:(MLState *)self.state
                                                options:options
                                                  error:error];
        }
    }
#endif

    id<MLFeatureProvider> result = [self.mlModel predictionFromFeatures:input
                                                                options:options
                                                                  error:error];

    return result;
}

- (BOOL)prewarmAndReturnError:(NSError* __autoreleasing*)error {
    NSError *localError = nil;
    BOOL result = [self.mlModel prewarmUsingState:self.state error:error];
    if (!result) {
        ETCoreMLLogError(localError,
                         "%@: Failed to prewarm model with identifier = %@",
                         NSStringFromClass(self.class),
                         self.identifier);
    }

#if MODEL_STATE_IS_SUPPORTED
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, *)) {
        NSDictionary<NSString *, MLFeatureDescription *> *stateDescriptions = self.mlModel.modelDescription.stateDescriptionsByName;
        [stateDescriptions enumerateKeysAndObjectsUsingBlock:^(NSString *featureName, MLFeatureDescription * __unused obj, BOOL * __unused stop) {
            reset_state_for_feature_name(featureName, (MLState *) self.state);
        }];
    }
#endif


    if (error) {
        *error = localError;
    }

    return result;
}

@end
