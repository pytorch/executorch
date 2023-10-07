//
// CoreMLBackendDelegateTests.mm
//
// Copyright © 2023 Apple Inc. All rights reserved.
//
// Please refer to the license found in the LICENSE file in the root directory of the source tree.

#import <XCTest/XCTest.h>

#import <string>

#import <executorch/runtime/core/data_loader.h>
#import <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#import <executorch/runtime/executor/method.h>
#import <executorch/runtime/executor/program.h>
#import <executorch/runtime/platform/runtime.h>
#import <executorch/util/util.h>

static constexpr size_t kRuntimeMemorySize = 10 * 1024U * 1024U; // 10 MB

using namespace torch::executor;
using torch::executor::testing::TensorFactory;

namespace {
// TODO: Move the following methods to a utility class, so that it can be shared with `executor_runner.main.mm`
NSData * _Nullable read_data(const std::string& filePath) {
    NSURL *url = [NSURL fileURLWithPath:@(filePath.c_str())];
    return [NSData dataWithContentsOfURL:url];
}
    
class DataLoaderImpl: public DataLoader {
public:
    DataLoaderImpl(std::string filePath)
    :data_(read_data(filePath))
    {}
    
    Result<FreeableBuffer> Load(size_t offset, size_t size) override {
        NSData *subdata = [data_ subdataWithRange:NSMakeRange(offset, size)];
        return FreeableBuffer(subdata.bytes, size, nullptr);
    }
    
    Result<size_t> size() const override {
        return data_.length;
    }
     
private:
   NSData *data_;
};
    
using Buffer = std::vector<uint8_t>;

std::unique_ptr<Program> get_program(NSURL *url) {
    DataLoaderImpl dataLoader(url.path.UTF8String);
    auto program = Program::load(&dataLoader);
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
}

@interface CoreMLBackendDelegateTests : XCTestCase

@end

@implementation CoreMLBackendDelegateTests

+ (void)setUp {
    runtime_init();
}

+ (nullable NSURL *)bundledResourceWithName:(NSString *)name extension:(NSString *)extension {
    NSBundle *bundle = [NSBundle bundleForClass:CoreMLBackendDelegateTests.class];
    return [bundle URLForResource:name withExtension:extension];
}

- (void)testProgramLoad {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"pte"];
    XCTAssertNotNil(modelURL);
    auto program = get_program(modelURL);
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

- (void)executeModelAtURL:(NSURL *)modelURL nTimes:(NSUInteger)nTimes {
    for (NSUInteger i = 0; i < nTimes; i++) {
        auto program = get_program(modelURL);
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
        auto inputs = util::PrepareInputTensors(method.get());
        
        auto status = method->execute();
        XCTAssertEqual(status, Error::Ok);
        auto outputs = methodAllocator.allocateList<EValue>(method->outputs_size());
        status = method->get_outputs(outputs, method->outputs_size());
        XCTAssertEqual(status, Error::Ok);
        
        util::FreeInputs(inputs);
    }
}

- (void)testAddProgramExecute {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"add_coreml_all" extension:@"pte"];
    XCTAssertNotNil(modelURL);
    [self executeModelAtURL:modelURL nTimes:10];
}

- (void)testMulProgramExecute {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"mul_coreml_all" extension:@"pte"];
    XCTAssertNotNil(modelURL);
    [self executeModelAtURL:modelURL nTimes:10];
}

- (void)testMV3ProgramExecute {
    NSURL *modelURL = [[self class] bundledResourceWithName:@"mv3_coreml_all" extension:@"pte"];
    XCTAssertNotNil(modelURL);
    [self executeModelAtURL:modelURL nTimes:10];
}

- (void)executeMultipleModelsConcurrently:(NSArray<NSURL *> *)modelURLs
                                   nTimes:(NSUInteger)nTimes
                                  timeout:(NSTimeInterval)timeout {
    NSMutableArray<XCTestExpectation *> *expectations = [NSMutableArray arrayWithCapacity:modelURLs.count];
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    for (NSURL *modelURL in modelURLs) {
        NSString *description = [NSString stringWithFormat:@"%@ must execute successfully", modelURL.lastPathComponent];
        XCTestExpectation *expectation = [[XCTestExpectation alloc] initWithDescription:description];
        [expectations addObject:expectation];
        dispatch_async(queue, ^{
            [self executeModelAtURL:modelURL nTimes:nTimes];
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
                                     nTimes:10
                                    timeout:5 * 60];
}

- (void)testSameModelExecutionConcurrently {
    NSURL *modelURL1 = [[self class] bundledResourceWithName:@"mv3_coreml_all" extension:@"pte"];
    NSURL *modelURL2 = [[self class] bundledResourceWithName:@"mv3_coreml_all" extension:@"pte"];
    [self executeMultipleModelsConcurrently:@[modelURL1, modelURL2]
                                     nTimes:10
                                    timeout:5 * 60];
}

@end
