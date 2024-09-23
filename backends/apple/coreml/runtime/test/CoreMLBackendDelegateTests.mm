//
// CoreMLBackendDelegateTests.mm
//
// Copyright © 2024 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <XCTest/XCTest.h>
#import <executorch/runtime/core/data_loader.h>
#import <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#import <executorch/runtime/executor/method.h>
#import <executorch/runtime/executor/program.h>
#import <executorch/runtime/platform/runtime.h>
#import <string>

#import "MLModel_Prewarm.h"

static constexpr size_t kRuntimeMemorySize = 50 * 1024U * 1024U; // 50 MB

using namespace torch::executor;
using torch::executor::testing::TensorFactory;

namespace {
// TODO: Move the following methods to a utility class, so that it can be shared with `executor_runner.main.mm`
NSData * _Nullable read_data(const std::string& filePath) {
    NSURL *url = [NSURL fileURLWithPath:@(filePath.c_str())];
    return [NSData dataWithContentsOfURL:url];
}

class DataLoaderImpl final : public DataLoader {
public:
    DataLoaderImpl(std::string filePath)
    :data_(read_data(filePath))
    {}

    Result<FreeableBuffer> load(
        size_t offset, size_t size, ET_UNUSED const DataLoader::SegmentInfo& segment_info) const override {
        NSData *subdata = [data_ subdataWithRange:NSMakeRange(offset, size)];
        return FreeableBuffer(subdata.bytes, size, nullptr);
    }

    Result<size_t> size() const override {
        return data_.length;
    }

private:
   NSData * const data_;
};

using Buffer = std::vector<uint8_t>;

std::unique_ptr<Program> get_program(DataLoader *loader) {
    auto program = Program::load(loader);
    if (!program.ok()) {
        return nullptr;
    }

    return std::make_unique<Program>(std::move(program.get()));
}

Result<std::string> get_method_name(Program *program) {
    const auto method_name = program->get_method_name(0);
    if (!method_name.ok()) {
        return Error::InvalidProgram;
    }

    return std::string(method_name.get());
}

Result<std::vector<Buffer>>
get_planned_buffers(const std::string& method_name, Program *program) {
    auto method_meta = program->method_meta(method_name.c_str());
    if (!method_meta.ok()) {
        return Error::InvalidProgram;
    }

    std::vector<std::vector<uint8_t>> buffers;
    buffers.reserve(method_meta->num_memory_planned_buffers());
    for (size_t bufferID = 0; bufferID < method_meta->num_memory_planned_buffers(); ++bufferID) {
        auto buffer_size = method_meta->memory_planned_buffer_size(bufferID);
        std::vector<uint8_t> data(buffer_size.get(), 0);
        buffers.emplace_back(std::move(data));
    }

    return buffers;
}

std::vector<Span<uint8_t>> to_spans(std::vector<Buffer>& buffers) {
    std::vector<Span<uint8_t>> result;
    result.reserve(buffers.size());
    for (auto& buffer : buffers) {
        result.emplace_back(buffer.data(), buffer.size());
    }

    return result;
}

Result<std::vector<Buffer>> prepare_input_tensors(Method& method) {
     MethodMeta method_meta = method.method_meta();
     size_t num_inputs = method_meta.num_inputs();
     std::vector<std::vector<uint8_t>> buffers;
     for (size_t i = 0; i < num_inputs; i++) {
         Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
         if (!tensor_meta.ok()) {
             ET_LOG(Info, "Skipping non-tensor input %zu", i);
             continue;
         }
         Buffer buffer(tensor_meta->nbytes(), 0);
         auto sizes = tensor_meta->sizes();
         exec_aten::TensorImpl tensor_impl(tensor_meta->scalar_type(), std::size(sizes), const_cast<int *>(sizes.data()), buffer.data());
         exec_aten::Tensor tensor(&tensor_impl);
         EValue input_value(std::move(tensor));
         Error err = method.set_input(input_value, i);
         if (err != Error::Ok) {
             ET_LOG(Error, "Failed to prepare input %zu: 0x%" PRIx32, i, (uint32_t)err);
             return err;
         }
         buffers.emplace_back(std::move(buffer));
     }

     return buffers;
 }
}

@interface CoreMLBackendDelegateTests : XCTestCase

@end

@implementation CoreMLBackendDelegateTests

+ (void)setUp {
    torch::executor::runtime_init();
}

+ (nullable NSURL *)bundledResourceWithName:(NSString *)name extension:(NSString *)extension {
    NSBundle *bundle = [NSBundle bundleForClass:CoreMLBackendDelegateTests.class];
    return [bundle URLForResource:name withExtension:extension];
}

- (void)testProgramLoad {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"pte"];
    XCTAssertNotNil(modelURL);
    auto loader = std::make_unique<DataLoaderImpl>(modelURL.path.UTF8String);
    auto program = get_program(loader.get());
    XCTAssert(program != nullptr);
    auto methodName = get_method_name(program.get());
    XCTAssert(methodName.ok());
    auto plannedBuffers = get_planned_buffers(methodName.get(), program.get());
    XCTAssert(plannedBuffers.ok());
    Buffer methodBuffer(kRuntimeMemorySize, 0);
    MemoryAllocator methodAllocator(static_cast<int32_t>(methodBuffer.size()), methodBuffer.data());
    auto spans = to_spans(plannedBuffers.get());
    HierarchicalAllocator plannedAllocator({spans.data(), spans.size()});
    MemoryManager memoryManger(&methodAllocator, &plannedAllocator);
    auto method = program->load_method(methodName.get().c_str(), &memoryManger);
    XCTAssert(method.ok());
}

- (void)executeModelAtURL:(NSURL *)modelURL nLoads:(NSUInteger)nLoads nExecutions:(NSUInteger)nExecutions {
    for (NSUInteger i = 0; i < nLoads; ++i) {
        auto loader = std::make_unique<DataLoaderImpl>(modelURL.path.UTF8String);
        auto program = get_program(loader.get());
        XCTAssert(program != nullptr);
        auto methodName = get_method_name(program.get());
        XCTAssert(methodName.ok());
        auto plannedBuffers = get_planned_buffers(methodName.get(), program.get());
        XCTAssert(plannedBuffers.ok());
        Buffer methodBuffer(kRuntimeMemorySize, 0);
        __block MemoryAllocator methodAllocator(static_cast<int32_t>(methodBuffer.size()), methodBuffer.data());
        auto spans = to_spans(plannedBuffers.get());
        HierarchicalAllocator plannedAllocator({spans.data(), spans.size()});
        MemoryManager memoryManger(&methodAllocator, &plannedAllocator);
        __block auto method = program->load_method(methodName.get().c_str(), &memoryManger);
        XCTAssert(method.ok());
        auto inputs = ::prepare_input_tensors(method.get());
        auto outputs = methodAllocator.allocateList<EValue>(method->outputs_size());
        for (NSUInteger j = 0; j < nExecutions; ++j) {
            auto status = method->execute();
            XCTAssertEqual(status, Error::Ok);
            status = method->get_outputs(outputs, method->outputs_size());
            XCTAssertEqual(status, Error::Ok);
        }
    }
}

- (void)testAddProgramExecute {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"pte"];
    XCTAssertNotNil(modelURL);
    [self executeModelAtURL:modelURL nLoads:1 nExecutions:2];
}

- (void)testMulProgramExecute {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"mul_coreml_all" extension:@"pte"];
    XCTAssertNotNil(modelURL);
    [self executeModelAtURL:modelURL nLoads:1 nExecutions:2];
}

- (void)testMV3ProgramExecute {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"mv3_coreml_all" extension:@"pte"];
    XCTAssertNotNil(modelURL);
    [self executeModelAtURL:modelURL nLoads:1 nExecutions:2];
}

#if MODEL_STATE_IS_SUPPORTED
- (void)testStateProgramExecute {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"state_coreml_all" extension:@"pte"];
    XCTAssertNotNil(modelURL);
    [self executeModelAtURL:modelURL nLoads:1 nExecutions:2];
}
#endif

- (void)executeMultipleModelsConcurrently:(NSArray<NSURL *> *)modelURLs
                                   nLoads:(NSUInteger)nLoads
                              nExecutions:(NSUInteger)nExecutions
                                  timeout:(NSTimeInterval)timeout {
    NSMutableArray<XCTestExpectation *> *expectations = [NSMutableArray arrayWithCapacity:modelURLs.count];
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    for (NSURL *modelURL in modelURLs) {
        NSString *description = [NSString stringWithFormat:@"%@ must execute successfully", modelURL.lastPathComponent];
        XCTestExpectation *expectation = [[XCTestExpectation alloc] initWithDescription:description];
        [expectations addObject:expectation];
        dispatch_async(queue, ^{
            [self executeModelAtURL:modelURL nLoads:nLoads nExecutions:nExecutions];
            [expectation fulfill];
        });
    }

    [self waitForExpectations:expectations timeout:timeout];
}

- (void)testMultipleModelExecutionConcurrently {
    NSURL *modelURL1 = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"pte"];
    NSURL *modelURL2 = [[self class] bundledResourceWithName:@"mul_coreml_all" extension:@"pte"];
    NSURL *modelURL3 = [[self class] bundledResourceWithName:@"mv3_coreml_all" extension:@"pte"];
    [self executeMultipleModelsConcurrently:@[modelURL1, modelURL2, modelURL3]
                                     nLoads:5
                                nExecutions:2
                                    timeout:5 * 60];
}

- (void)testSameModelExecutionConcurrently {
    NSURL *modelURL1 = [[self class] bundledResourceWithName:@"mv3_coreml_all" extension:@"pte"];
    NSURL *modelURL2 = [[self class] bundledResourceWithName:@"mv3_coreml_all" extension:@"pte"];
    [self executeMultipleModelsConcurrently:@[modelURL1, modelURL2]
                                     nLoads:5
                                nExecutions:2
                                    timeout:5 * 60];
}

@end
