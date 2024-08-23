//
// ETCoreMLModelDebuggerTests.mm
//
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import "ETCoreMLTestUtils.h"
#import <ETCoreMLModelAnalyzer.h>
#import <ETCoreMLModelStructurePath.h>
#import <ETCoreMLOperationProfilingInfo.h>
#import <XCTest/XCTest.h>
#import <executorch/runtime/platform/runtime.h>
#import <model_event_logger.h>
#import <model_logging_options.h>

namespace  {
using namespace executorchcoreml::modelstructure;

using NotifyFn = std::function<void(NSDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *op_path_to_value_map,
                                    NSDictionary<ETCoreMLModelStructurePath *, NSString *> *op_path_to_debug_symbol_name_map)>;

class ModelEventLoggerImpl: public executorchcoreml::ModelEventLogger {
public:
    explicit ModelEventLoggerImpl(NotifyFn fn)
    :fn_(fn)
    {}

    void log_profiling_infos(NSDictionary<ETCoreMLModelStructurePath *, ETCoreMLOperationProfilingInfo *> *op_path_to_profiling_info_map,
                             NSDictionary<ETCoreMLModelStructurePath *, NSString *> *op_path_to_debug_symbol_name_map) const noexcept {}

    void log_intermediate_tensors(NSDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *op_path_to_value_map,
                                  NSDictionary<ETCoreMLModelStructurePath *, NSString *> *op_path_to_debug_symbol_map) const noexcept {
        fn_(op_path_to_value_map, op_path_to_debug_symbol_map);
    }

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

void add_debugging_result(NSDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *debugging_result,
                          NSDictionary<ETCoreMLModelStructurePath *, NSString *> *path_to_symbol_name_map,
                          NSMutableDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *debugging_results) {
    for (ETCoreMLModelStructurePath *path in debugging_result) {
        NSString *debug_symbol = path_to_symbol_name_map[path];
        if (debug_symbol) {
            debugging_results[path] = debugging_result[path];
        }
    }
}

}

@interface ETCoreMLModelDebuggerTests : XCTestCase

@end

@implementation ETCoreMLModelDebuggerTests

+ (void)setUp {
    torch::executor::runtime_init();
}

+ (nullable NSURL *)bundledResourceWithName:(NSString *)name extension:(NSString *)extension {
    NSBundle *bundle = [NSBundle bundleForClass:ETCoreMLModelDebuggerTests.class];
    return [bundle URLForResource:name withExtension:extension];
}

- (void)debugModelWithName:(NSString *)modelName
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
        loggingOptions.log_intermediate_tensors = true;
        ModelEventLoggerImpl eventLogger(notify);

        NSArray<MLMultiArray *> *outputs = [analyzer executeModelWithInputs:inputs
                                                          predictionOptions:predictionOptions
                                                             loggingOptions:loggingOptions
                                                                eventLogger:&eventLogger
                                                                      error:&error];
        XCTAssertNotNil(outputs);
        
    }
    [fm removeItemAtURL:dstURL error:nil];
}

- (void)testAddProgramDebugging {
    NotifyFn notify = [](NSDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *debuggingResult,
                         NSDictionary<ETCoreMLModelStructurePath *, NSString *> *pathToSymbolNameMap) {
        // There are 3 add ops, we verify that we get the outputs for the ops.
        XCTAssertNotNil(debuggingResult[make_path_with_output_name("aten_add_tensor_cast_fp16")]);
    };
    
    [self debugModelWithName:@"add_coreml_all"
         repeatedInputValues:@[@(1), @(2)]
                      notify:notify];
}

- (void)testMulProgramDebugging {
    NotifyFn notify = [](NSDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *debuggingResult,
                         NSDictionary<ETCoreMLModelStructurePath *, NSString *> *pathToSymbolNameMap) {
        // There is 1 `mul` op, we verify that we get the output for the op.
        XCTAssertNotNil(debuggingResult[make_path_with_output_name("aten_mul_tensor_cast_fp16")]);
    };
    
    [self debugModelWithName:@"mul_coreml_all"
         repeatedInputValues:@[@(1), @(2)]
                      notify:notify];
}

- (void)testMV3ProgramDebugging {
    NSMutableDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *debuggingResults = [NSMutableDictionary new];
    NotifyFn notify = [debuggingResults](NSDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *debuggingResult,
                                         NSDictionary<ETCoreMLModelStructurePath *, NSString *> *pathToSymbolNameMap) mutable {
        add_debugging_result(debuggingResult, pathToSymbolNameMap, debuggingResults);
    };
    
    [self debugModelWithName:@"mv3_coreml_all"
         repeatedInputValues:@[@(1), @(2)]
                      notify:notify];

    // There are more than 200 ops, we verify the outputs for specific ops.
    XCTAssertNotNil(debuggingResults[make_path_with_output_name("aten__native_batch_norm_legit_no_training_default_13_cast_fp16")]);
    XCTAssertNotNil(debuggingResults[make_path_with_output_name("_inversed_aten_div_tensor_24_cast_fp16")]);
    XCTAssertNotNil(debuggingResults[make_path_with_output_name("aten_mean_dim_7_cast_fp16")]);
    XCTAssertNotNil(debuggingResults[make_path_with_output_name("aten_clamp_default_54_cast_fp16")]);
    XCTAssertNotNil(debuggingResults[make_path_with_output_name("aten__native_batch_norm_legit_no_training_default_22_cast_fp16")]);
    XCTAssertNotNil(debuggingResults[make_path_with_output_name("aten_mul_tensor_27_cast_fp16")]);
}

- (void)testAddMulProgramDebugging {
    NSMutableDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *debuggingResults = [NSMutableDictionary new];
    NotifyFn notify = [debuggingResults](NSDictionary<ETCoreMLModelStructurePath *, MLMultiArray *> *debuggingResult,
                                         NSDictionary<ETCoreMLModelStructurePath *, NSString *> *pathToSymbolNameMap) mutable {
        add_debugging_result(debuggingResult, pathToSymbolNameMap, debuggingResults);
    };

    [self debugModelWithName:@"add_mul_coreml_all"
         repeatedInputValues:@[@(1), @(2)]
                      notify:notify];

    // There are more than 200 ops, we verify the outputs for specific ops.
    XCTAssertNotNil(debuggingResults[make_path_with_output_name("aten_add_tensor")]);
    XCTAssertNotNil(debuggingResults[make_path_with_output_name("aten_mm_default_cast_fp16")]);
}

@end
