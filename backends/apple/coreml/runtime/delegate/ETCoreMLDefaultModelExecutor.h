//
//  ETCoreMLDefaultModelExecutor.h
//  executorchcoreml_tests
//
//  Created by Gyan Sinha on 2/25/24.
//

#import <CoreML/CoreML.h>

#import <ETCoreMLModelExecutor.h>

@class ETCoreMLModel;

NS_ASSUME_NONNULL_BEGIN
/// The default model executor, the executor ignores logging options.
__attribute__((objc_subclassing_restricted))
@interface ETCoreMLDefaultModelExecutor : NSObject<ETCoreMLModelExecutor>

+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)init NS_UNAVAILABLE;

/// Constructs an `ETCoreMLDefaultModelExecutor` from the given model.
///
/// @param model The model.
- (instancetype)initWithModel:(ETCoreMLModel*)model NS_DESIGNATED_INITIALIZER;

/// The model.
@property (readonly, strong, nonatomic) ETCoreMLModel* model;

/// If set to `YES` then output backing are ignored.
@property (readwrite, atomic) BOOL ignoreOutputBackings;

@end

NS_ASSUME_NONNULL_END
