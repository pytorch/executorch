//
// ETCoreMLProgramProfilerTests.mm
//
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLTestUtils.h"
#import <ETCoreMLModelAnalyzer.h>
#import <ETCoreMLModelProfiler.h>
#import <ETCoreMLModelStructurePath.h>
#import <ETCoreMLOperationProfilingInfo.h>
#import <XCTest/XCTest.h>
#import <executorch/runtime/platform/runtime.h>
#import <model_event_logger.h>
#import <model_logging_options.h>

namespace  {
using namespace executorchcoreml::modelstructure;

using NotifyFn = std::function<void(NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *op_path_to_profiling_info_map,
                                    NSDictionary<ETCoreMLModelStructurePath *, NSString *> *op_path_to_debug_symbol_name_map)>;

class ModelProfilingEventLoggerImpl: public executorchcoreml::ModelEventLogger {
public:
    explicit ModelProfilingEventLoggerImpl(NotifyFn fn)
    :fn_(fn)
    {}
    
    void log_profiling_infos(NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *op_path_to_profiling_info_map,
                             NSDictionary<ETCoreMLModelStructurePath *, NSString *> *op_path_to_debug_symbol_name_map) const noexcept {
        fn_(op_path_to_profiling_info_map, op_path_to_debug_symbol_name_map);
    }
    
    void log_intermediate_tensors(NSDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *op_path_to_value_map,
                                  NSDictionary<ETCoreMLModelStructurePath *, NSString *> *op_path_to_debug_symbol_name_map) const noexcept {}
    
private:
    NotifyFn fn_;
};

ETCoreMLModelStructurePath *make_path_with_output_name(const std::string& output_name,
                                                       const std::string& function_name = "main") {
    Path path;
    path.append_component(Path::Program());
    path.append_component(Path::Program::Function(function_name));
    path.append_component(Path::Program::Block(-1));
    path.append_component(Path::Program::Operation(output_name));
    
    return [[ETCoreMLModelStructurePath alloc] initWithUnderlyingValue:std::move(path)];
}
} // namespace

@interface ETCoreMLModelProfilerTests : XCTestCase

@end

@implementation ETCoreMLModelProfilerTests

+ (void)setUp {
    torch::executor::runtime_init();
}

+ (nullable NSURL *)bundledResourceWithName:(NSString *)name extension:(NSString *)extension {
    NSBundle *bundle = [NSBundle bundleForClass:ETCoreMLModelProfilerTests.class];
    return [bundle URLForResource:name withExtension:extension];
}

- (void)profileModelWithName:(NSString *)modelName
         repeatedInputValues:(NSArray<NSNumber *> *)repeatedInputValues
                      notify:(NotifyFn)notify {
    NSError *error = nil;
    NSURL *modelURL = [[self class] bundledResourceWithName:modelName extension:@"bin"];
    NSData *modelData = [NSData dataWithContentsOfURL:modelURL];
    NSURL *dstURL = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:[NSUUID UUID].UUIDString];
    NSFileManager *fm = [[NSFileManager alloc] init];
    XCTAssert([fm createDirectoryAtURL:dstURL withIntermediateDirectories:NO attributes:@{} error:&error]);
    @autoreleasepool {
        ETCoreMLModelAnalyzer *analyzer = [ETCoreMLTestUtils createAnalyzerWithAOTData:modelData
                                                                                dstURL:dstURL
                                                                                 error:&error];
        XCTAssertNotNil(analyzer);
        id<MLFeatureProvider> inputs = [ETCoreMLTestUtils inputFeaturesForModel:analyzer.model
                                                                 repeatedValues:repeatedInputValues
                                                                          error:&error];
        XCTAssertNotNil(inputs);
        MLPredictionOptions *predictionOptions = [[MLPredictionOptions alloc] init];
        executorchcoreml::ModelLoggingOptions loggingOptions;
        loggingOptions.log_profiling_info = true;
        ModelProfilingEventLoggerImpl eventLogger(notify);
        
        NSArray<MLMultiArray *> *outputs = [analyzer executeModelWithInputs:inputs
                                                          predictionOptions:predictionOptions
                                                             loggingOptions:loggingOptions
                                                                eventLogger:&eventLogger
                                                                      error:&error];
        XCTAssertNotNil(outputs);
        
    }
    [fm removeItemAtURL:dstURL error:nil];
}

- (void)testAddProgramProfiling {
#if MODEL_PROFILING_IS_AVAILABLE
    if (@available(macOS 14.4, iOS 17.4, watchOS 10.4, tvOS 17.4, *)) {
        NotifyFn notify = [](NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *profilingResult,
                             NSDictionary<ETCoreMLModelStructurePath *, NSString *> *__unused pathToSymbolNameMap) {
            // There is 1 add op, we verify that the profiling info exists for the ops.
            XCTAssertNotNil(profilingResult[make_path_with_output_name("aten_add_tensor_cast_fp16")]);
        };
        
        [self profileModelWithName:@"add_coreml_all"
               repeatedInputValues:@[@(1), @(2)]
                            notify:notify];
    }
#else
    XCTSkip(@"Model profiling is only available on macOS 14.4, iOS 17.4, tvOS 17.4, and watchOS 10.4.");
#endif
}

- (void)testMulProgramProfiling {
#if MODEL_PROFILING_IS_AVAILABLE
    if (@available(macOS 14.4, iOS 17.4, tvOS 17.4, watchOS 10.4, *)) {
        NotifyFn notify = [](NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *profilingResult,
                             NSDictionary<ETCoreMLModelStructurePath *, NSString *> *__unused pathToSymbolNameMap) {
            // There is 1 `mul` op, we verify that the profiling info exists for the op.
            XCTAssertNotNil(profilingResult[make_path_with_output_name("aten_mul_tensor_cast_fp16")]);
        };
        
        [self profileModelWithName:@"mul_coreml_all"
               repeatedInputValues:@[@(1), @(2)]
                            notify:notify];
    }
#else
    XCTSkip(@"Model profiling is only available on macOS 14.4, iOS 17.4, tvOS 17.4, and watchOS 10.4.");
#endif
}

- (void)testMV3ProgramProfiling {
#if MODEL_PROFILING_IS_AVAILABLE
    if (@available(macOS 14.4, iOS 17.4, tvOS 17.4, watchOS 10.4, *)) {
        NotifyFn notify = [](NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *profilingResult,
                             NSDictionary<ETCoreMLModelStructurePath *, NSString *> *__unused pathToSymbolNameMap) {
            // There are more than 200 ops, we verify the profiling info for specific ops.
            XCTAssertNotNil(profilingResult[make_path_with_output_name("aten__native_batch_norm_legit_no_training_default_13_cast_fp16")]);
            XCTAssertNotNil(profilingResult[make_path_with_output_name("_inversed_aten_div_tensor_24_cast_fp16")]);
            XCTAssertNotNil(profilingResult[make_path_with_output_name("aten_mean_dim_7_cast_fp16")]);
            XCTAssertNotNil(profilingResult[make_path_with_output_name("aten_clamp_default_54_cast_fp16")]);
            XCTAssertNotNil(profilingResult[make_path_with_output_name("aten__native_batch_norm_legit_no_training_default_22_cast_fp16")]);
            XCTAssertNotNil(profilingResult[make_path_with_output_name("aten_mul_tensor_27_cast_fp16")]);
        };
        
        [self profileModelWithName:@"mv3_coreml_all"
               repeatedInputValues:@[@(1), @(2)]
                            notify:notify];
    }
#else
    XCTSkip(@"Model profiling is only available on macOS 14.4, iOS 17.4, tvOS 17.4, and watchOS 10.4.");
#endif
}

@end
