/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <ETCoreMLAssetManager.h>
#import <ETCoreMLModel.h>
#import <ETCoreMLModelExecutor.h>
#import <ETCoreMLModelManager.h>
#import <ETCoreMLStrings.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>
#import <backend_delegate.h>
#import <coreml_backend/delegate.h>
#import <inmemory_filesystem_utils.hpp>
#import <model_logging_options.h>
#import <multiarray.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/program_generated.h>
#include <flatbuffers/flatbuffers.h>
#include <format/Model.pb.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

using executorch::backends::coreml::CoreMLBackendDelegate;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorchcoreml::BackendDelegate;
using executorchcoreml::Buffer;
using executorchcoreml::ModelLoggingOptions;
using executorchcoreml::MultiArray;

@interface ETCoreMLModelManager (PixelBufferBindingTest)
- (id<ETCoreMLModelExecutor>)executorWithHandle:(ModelHandle *)handle;
@end

namespace {

constexpr size_t kHeight = 8;
constexpr size_t kWidth = 16;
constexpr std::array<size_t, 4> kShape = {1, 1, kHeight, kWidth};

struct PixelBufferDeleter {
    void operator()(CVPixelBufferRef pixelBuffer) const { CVPixelBufferRelease(pixelBuffer); }
};
using PixelBufferPtr = std::unique_ptr<std::remove_pointer_t<CVPixelBufferRef>, PixelBufferDeleter>;

PixelBufferPtr makePixelBuffer(OSType format, bool ioSurfaceBacked = true) {
    NSDictionary *attributes = ioSurfaceBacked ? @{(id)kCVPixelBufferIOSurfacePropertiesKey: @{}} : @{};
    CVPixelBufferRef pixelBuffer = nullptr;
    CVPixelBufferCreate(kCFAllocatorDefault, kWidth, kHeight, format, (__bridge CFDictionaryRef)attributes, &pixelBuffer);
    return PixelBufferPtr(pixelBuffer);
}

void *baseAddress(CVPixelBufferRef pixelBuffer) {
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    void *address = CVPixelBufferGetBaseAddress(pixelBuffer);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return address;
}

_Float16 *row(CVPixelBufferRef pixelBuffer, size_t y) {
    auto *base = static_cast<uint8_t *>(CVPixelBufferGetBaseAddress(pixelBuffer));
    return reinterpret_cast<_Float16 *>(base + y * CVPixelBufferGetBytesPerRow(pixelBuffer));
}

IOSurfaceID surfaceID(CVPixelBufferRef pixelBuffer) {
    IOSurfaceRef surface = CVPixelBufferGetIOSurface(pixelBuffer);
    return surface != nullptr ? IOSurfaceGetID(surface) : 0;
}

MultiArray makeArg(void *data, std::vector<size_t> shape, MultiArray::DataType dataType = MultiArray::DataType::Float16) {
    std::vector<ssize_t> strides(shape.size());
    ssize_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        strides[i - 1] = stride;
        stride *= shape[i - 1];
    }
    return MultiArray(data, MultiArray::MemoryLayout(dataType, std::move(shape), std::move(strides)));
}

template <size_t N>
MultiArray makeArg(void *data, const std::array<size_t, N>& shape, MultiArray::DataType dataType = MultiArray::DataType::Float16) {
    return makeArg(data, std::vector<size_t>(shape.begin(), shape.end()), dataType);
}

float inputValue(size_t x, size_t y, int frame) {
    return float((y * kWidth + x + frame) % 64) / 8;
}

std::string toString(NSString *string) {
    const char *utf8 = string.UTF8String;
    return utf8 != nullptr ? utf8 : "";
}

/// Returns an ML program that computes `z = x + x` for an fp16 or fp32 `x` of shape `kShape`.
std::string makeModelSpec(bool useFloat16 = true) {
    namespace spec = CoreML::Specification;
    namespace mil = CoreML::Specification::MILSpec;
    auto setTensorType = [useFloat16](mil::ValueType *type) {
        auto *tensorType = type->mutable_tensortype();
        tensorType->set_datatype(useFloat16 ? mil::FLOAT16 : mil::FLOAT32);
        tensorType->set_rank(kShape.size());
        for (size_t size : kShape) {
            tensorType->add_dimensions()->mutable_constant()->set_size(size);
        }
    };
    auto setFeature = [useFloat16](spec::FeatureDescription *feature, const char *name) {
        feature->set_name(name);
        auto *arrayType = feature->mutable_type()->mutable_multiarraytype();
        arrayType->set_datatype(useFloat16 ? spec::ArrayFeatureType::FLOAT16 : spec::ArrayFeatureType::FLOAT32);
        for (size_t size : kShape) {
            arrayType->add_shape(size);
        }
    };

    spec::Model model;
    // Core ML 6 is the first version with fp16 model inputs and outputs.
    model.set_specificationversion(7);
    setFeature(model.mutable_description()->add_input(), "x");
    setFeature(model.mutable_description()->add_output(), "z");
    auto *program = model.mutable_mlprogram();
    program->set_version(1);
    auto& function = (*program->mutable_functions())["main"];
    function.set_opset("CoreML6");
    auto *input = function.add_inputs();
    input->set_name("x");
    setTensorType(input->mutable_type());
    auto& block = (*function.mutable_block_specializations())["CoreML6"];
    block.add_outputs("z");
    auto *add = block.add_operations();
    add->set_type("add");
    (*add->mutable_inputs())["x"].add_arguments()->set_name("x");
    (*add->mutable_inputs())["y"].add_arguments()->set_name("x");
    auto *output = add->add_outputs();
    output->set_name("z");
    setTensorType(output->mutable_type());
    return model.SerializeAsString();
}

void writeJSON(NSDictionary *object, NSURL *url) {
    [[NSJSONSerialization dataWithJSONObject:object options:0 error:nil] writeToURL:url atomically:YES];
}

/// Writes the model package and its metadata to `directory` and returns them in the delegate's AOT format.
std::string makeModelBlob(NSURL *directory, bool useFloat16 = true) {
    NSURL *packageURL = [directory URLByAppendingPathComponent:@"model.mlpackage"];
    NSURL *specDirectoryURL = [packageURL URLByAppendingPathComponent:@"Data/com.apple.CoreML"];
    [NSFileManager.defaultManager createDirectoryAtURL:specDirectoryURL
                           withIntermediateDirectories:YES
                                            attributes:nil
                                                 error:nil];
    const std::string spec = makeModelSpec(useFloat16);
    NSURL *specURL = [specDirectoryURL URLByAppendingPathComponent:@"model.mlmodel"];
    if (specURL == nil) {
        return "";
    }
    [[NSData dataWithBytes:spec.data() length:spec.size()] writeToURL:specURL atomically:YES];
    NSString *specItem = NSUUID.UUID.UUIDString;
    writeJSON(@{
        @"fileFormatVersion": @"1.0.0",
        @"itemInfoEntries": @{
            specItem: @{
                @"author": @"com.apple.CoreML",
                @"description": @"CoreML Model Specification",
                @"name": @"model.mlmodel",
                @"path": @"com.apple.CoreML/model.mlmodel",
            },
        },
        @"rootModelIdentifier": specItem,
    }, [packageURL URLByAppendingPathComponent:@"Manifest.json"]);
    writeJSON(@{
        @"identifier": NSUUID.UUID.UUIDString,
        @"inputNames": @[@"x"],
        @"outputNames": @[@"z"],
    }, [directory URLByAppendingPathComponent:@"metadata.json"]);

    std::error_code error;
    auto fileSystem = inmemoryfs::InMemoryFileSystem::make_from_directory(toString(directory.path),
                                                                          inmemoryfs::InMemoryFileSystem::FileLoadOption::Malloc,
                                                                          error);
    std::ostringstream stream;
    if (fileSystem == nullptr || !inmemoryfs::serialize(*fileSystem, {}, 1, stream, error)) {
        return "";
    }
    return stream.str();
}

namespace fb = executorch_flatbuffer;

/// Describes the serialized ExecutionPlan behind a MethodMeta fixture.
struct MethodMetaSpec {
    std::vector<std::string> backendIds;
    bool extraFreeCall = false;
    bool plannedInput = false;
    bool plannedOutput = false;
    bool constantInput = false;
    bool constantOutput = false;
    bool nonTensorInput = false;
    bool nonTensorOutput = false;
};

MethodMetaSpec makeSingleCoreMLSpec() {
    MethodMetaSpec spec;
    spec.backendIds = {toString(ETCoreMLStrings.delegateIdentifier)};
    return spec;
}

flatbuffers::Offset<fb::EValue> makeMetaTensorValue(
    flatbuffers::FlatBufferBuilder& builder, bool planned, bool constant) {
    flatbuffers::Offset<fb::AllocationDetails> allocation;
    if (planned) {
        allocation = fb::CreateAllocationDetails(builder, /*memory_id=*/1, /*memory_offset_low=*/0);
    }
    auto sizes = builder.CreateVector(std::vector<int32_t>(kShape.begin(), kShape.end()));
    auto dimOrder = builder.CreateVector(std::vector<uint8_t>{0, 1, 2, 3});
    auto tensor = fb::CreateTensor(
        builder,
        fb::ScalarType::HALF,
        /*storage_offset=*/0,
        sizes,
        dimOrder,
        /*requires_grad=*/false,
        constant ? 1 : 0,
        allocation);
    return fb::CreateEValue(builder, fb::KernelTypes::Tensor, tensor.Union());
}

flatbuffers::Offset<fb::EValue> makeMetaIntValue(flatbuffers::FlatBufferBuilder& builder) {
    auto intValue = fb::CreateInt(builder, /*int_val=*/0);
    return fb::CreateEValue(builder, fb::KernelTypes::Int, intValue.Union());
}

/// Serializes an ExecutionPlan for MethodMeta::from_serialized_execution_plan.
/// The returned bytes own the plan; the MethodMeta view borrows them.
std::vector<uint8_t> makeMethodMetaBytes(const MethodMetaSpec& spec) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<fb::EValue>> values;
    values.push_back(spec.nonTensorInput ? makeMetaIntValue(builder)
                                          : makeMetaTensorValue(builder, spec.plannedInput, spec.constantInput));
    values.push_back(spec.nonTensorOutput ? makeMetaIntValue(builder)
                                           : makeMetaTensorValue(builder, spec.plannedOutput, spec.constantOutput));
    const std::vector<int32_t> inputs = {0};
    const std::vector<int32_t> outputs = {1};

    std::vector<flatbuffers::Offset<fb::Instruction>> instructions;
    auto call = fb::CreateDelegateCall(
        builder, /*delegate_index=*/0, builder.CreateVector(std::vector<int32_t>{0, 1}));
    instructions.push_back(
        fb::CreateInstruction(builder, fb::InstructionArguments::DelegateCall, call.Union()));
    if (spec.extraFreeCall) {
        auto freeCall = fb::CreateFreeCall(builder, /*value_index=*/0);
        instructions.push_back(
            fb::CreateInstruction(builder, fb::InstructionArguments::FreeCall, freeCall.Union()));
    }

    std::vector<flatbuffers::Offset<fb::BackendDelegate>> delegates;
    delegates.reserve(spec.backendIds.size());
    for (const std::string& backendId : spec.backendIds) {
        delegates.push_back(fb::CreateBackendDelegate(
            builder,
            builder.CreateString(backendId),
            fb::CreateBackendDelegateDataReference(builder, fb::DataLocation::INLINE, /*index=*/0)));
    }

    auto chain = fb::CreateChain(
        builder, builder.CreateVector(inputs), builder.CreateVector(outputs), builder.CreateVector(instructions));
    auto plan = fb::CreateExecutionPlan(
        builder,
        builder.CreateString("forward"),
        /*container_meta_type=*/0,
        builder.CreateVector(values),
        builder.CreateVector(inputs),
        builder.CreateVector(outputs),
        builder.CreateVector(std::vector<flatbuffers::Offset<fb::Chain>>{chain}),
        builder.CreateVector(std::vector<flatbuffers::Offset<fb::Operator>>{}),
        builder.CreateVector(delegates),
        // Two entries on purpose: proves the guard does not restrict num_buffers.
        builder.CreateVector(std::vector<int64_t>{0, 0}));
    builder.Finish(plan);
    return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

class PixelBufferBindingTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        executorch::runtime::runtime_init();
    }

    void SetUp() override {
        directory_ = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:NSUUID.UUID.UUIDString]];
        blob_ = makeModelBlob(directory_);
        ASSERT_FALSE(blob_.empty());
        delegate_ = BackendDelegate::make(BackendDelegate::Config());
    }

    void TearDown() override {
        delegate_.reset();
        [NSFileManager.defaultManager removeItemAtURL:directory_ error:nil];
    }

    BackendDelegate::Handle *load(Span<BackendOption> runtimeSpecs = {}) {
        // In the iOS simulator, Core ML reads pixel buffer inputs as zeros unless it runs on the CPU.
        const std::string computeUnits = toString(ETCoreMLStrings.cpuComputeUnitName);
        const std::unordered_map<std::string, Buffer> specs = {
            {toString(ETCoreMLStrings.computeUnitsKeyName), Buffer(computeUnits.data(), computeUnits.size())},
        };
        return delegate_->init(Buffer(blob_.data(), blob_.size()),
                               specs,
                               nullptr,
                               nullptr,
                               Span<const BackendOption>(runtimeSpecs.data(), runtimeSpecs.size()));
    }

    // Loads the model the way ExecuTorch's runtime does, through the registered backend's init.
    Result<DelegateHandle*> loadThroughBackend(
        Span<BackendOption> runtimeSpecs, const MethodMeta* methodMeta = nullptr) {
        BackendInitContext context(&allocator_,
                                   /*event_tracer=*/nullptr,
                                   /*method_name=*/"forward",
                                   /*named_data_map=*/nullptr,
                                   Span<const BackendOption>(runtimeSpecs.data(), runtimeSpecs.size()),
                                   methodMeta);
        std::string computeUnits = toString(ETCoreMLStrings.cpuComputeUnitName);
        const std::string computeUnitsKey = toString(ETCoreMLStrings.computeUnitsKeyName);
        CompileSpec computeUnitsSpec{computeUnitsKey.c_str(), {computeUnits.data(), computeUnits.size()}};
        FreeableBuffer processed(blob_.data(), blob_.size(), nullptr);
        return backend_.init(context, &processed, {&computeUnitsSpec, 1});
    }

    NSURL *directory_ = nil;
    std::string blob_;
    std::shared_ptr<BackendDelegate> delegate_;
    CoreMLBackendDelegate backend_;
    std::vector<uint8_t> allocatorPool_ = std::vector<uint8_t>(4096);
    MemoryAllocator allocator_{static_cast<uint32_t>(allocatorPool_.size()), allocatorPool_.data()};
};

TEST_F(PixelBufferBindingTest, BindsOnlyIOSurfaceBackedFp16PixelBuffers) {
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)load();
    ASSERT_NE(model, nil);

    EXPECT_FALSE([model bindPixelBuffer:makePixelBuffer(kCVPixelFormatType_OneComponent16Half, false).get() error:nil]);
    EXPECT_FALSE([model bindPixelBuffer:makePixelBuffer(kCVPixelFormatType_OneComponent32Float).get() error:nil]);
    EXPECT_TRUE([model bindPixelBuffer:makePixelBuffer(kCVPixelFormatType_OneComponent16Half).get() error:nil]);
}

TEST_F(PixelBufferBindingTest, ReusesTheBoundBufferArrayWhileTheArgShapeStaysTheSame) {
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)load();
    ASSERT_NE(model, nil);
    PixelBufferPtr pixelBuffer = makePixelBuffer(kCVPixelFormatType_OneComponent16Half);
    ASSERT_TRUE([model bindPixelBuffer:pixelBuffer.get() error:nil]);
    void *data = baseAddress(pixelBuffer.get());

    MLMultiArray *array = nil;
    ASSERT_TRUE([model getPixelBufferMultiArray:&array forArg:makeArg(data, kShape) error:nil]);
    ASSERT_NE(array, nil);
    EXPECT_EQ(array.pixelBuffer, pixelBuffer.get());

    MLMultiArray *sameShapeArray = nil;
    ASSERT_TRUE([model getPixelBufferMultiArray:&sameShapeArray forArg:makeArg(data, kShape) error:nil]);
    EXPECT_EQ(sameShapeArray, array);

    MLMultiArray *otherShapeArray = nil;
    ASSERT_TRUE([model getPixelBufferMultiArray:&otherShapeArray forArg:makeArg(data, {kHeight, kWidth}) error:nil]);
    ASSERT_NE(otherShapeArray, nil);
    EXPECT_NE(otherShapeArray, array);
    EXPECT_EQ(otherShapeArray.pixelBuffer, pixelBuffer.get());
}

TEST_F(PixelBufferBindingTest, FindsNoArrayForAnUnboundDataPointer) {
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)load();
    ASSERT_NE(model, nil);
    ASSERT_TRUE([model bindPixelBuffer:makePixelBuffer(kCVPixelFormatType_OneComponent16Half).get() error:nil]);

    std::vector<_Float16> data(kHeight * kWidth);
    MLMultiArray *array = nil;
    EXPECT_TRUE([model getPixelBufferMultiArray:&array forArg:makeArg(data.data(), kShape) error:nil]);
    EXPECT_EQ(array, nil);
}

TEST_F(PixelBufferBindingTest, RejectsAnArgThatDoesntMatchTheBoundBuffer) {
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)load();
    ASSERT_NE(model, nil);
    PixelBufferPtr pixelBuffer = makePixelBuffer(kCVPixelFormatType_OneComponent16Half);
    ASSERT_TRUE([model bindPixelBuffer:pixelBuffer.get() error:nil]);
    void *data = baseAddress(pixelBuffer.get());

    MLMultiArray *array = nil;
    NSError *error = nil;
    EXPECT_FALSE([model getPixelBufferMultiArray:&array forArg:makeArg(data, {1, 1, kHeight, 2 * kWidth}) error:&error]);
    EXPECT_NE(error, nil);
    error = nil;
    EXPECT_FALSE([model getPixelBufferMultiArray:&array forArg:makeArg(data, {1, 2, kHeight, kWidth}) error:&error]);
    EXPECT_NE(error, nil);
    error = nil;
    EXPECT_FALSE([model getPixelBufferMultiArray:&array
                                          forArg:makeArg(data, kShape, MultiArray::DataType::Float32)
                                           error:&error]);
    EXPECT_NE(error, nil);
}

TEST_F(PixelBufferBindingTest, RejectsBoundBuffersWithMismatchedModelFeatureTypes) {
    blob_ = makeModelBlob(directory_, /*useFloat16=*/false);
    ASSERT_FALSE(blob_.empty());
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)load();
    ASSERT_NE(model, nil);
    PixelBufferPtr pixelBuffer = makePixelBuffer(kCVPixelFormatType_OneComponent16Half);
    ASSERT_NE(pixelBuffer.get(), nullptr);
    ASSERT_TRUE([model bindPixelBuffer:pixelBuffer.get() error:nil]);
    const std::vector<MultiArray> args = {makeArg(baseAddress(pixelBuffer.get()), kShape)};

    NSError *error = nil;
    EXPECT_EQ([model prepareInputs:args error:&error], nil);
    EXPECT_NE(error, nil);
    error = nil;
    EXPECT_EQ([model prepareOutputBackings:args error:&error], nil);
    EXPECT_NE(error, nil);
}

TEST_F(PixelBufferBindingTest, ConvertsUnboundBuffersWithMismatchedModelFeatureTypes) {
    blob_ = makeModelBlob(directory_, /*useFloat16=*/false);
    ASSERT_FALSE(blob_.empty());
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)load();
    ASSERT_NE(model, nil);
    std::vector<_Float16> data(kHeight * kWidth, 2);
    const std::vector<MultiArray> args = {makeArg(data.data(), kShape)};

    NSError *error = nil;
    NSArray<MLMultiArray *> *inputs = [model prepareInputs:args error:&error];
    ASSERT_NE(inputs, nil);
    EXPECT_EQ(error, nil);
    EXPECT_EQ(inputs.firstObject.dataType, MLMultiArrayDataTypeFloat32);
    EXPECT_EQ([inputs.firstObject[0] floatValue], 2.0f);
    NSArray<MLMultiArray *> *outputs = [model prepareOutputBackings:args error:&error];
    ASSERT_NE(outputs, nil);
    EXPECT_EQ(error, nil);
    EXPECT_EQ(outputs.firstObject.dataType, MLMultiArrayDataTypeFloat32);
}

TEST_F(PixelBufferBindingTest, CopiesIntoBoundOutputWhenCoreMLIgnoresBackings) {
    // A model manager's executor can withhold output backings. Core ML then allocates the output itself, and the
    // result reaches the bound buffer only through the fallback copy.
    NSURL *assetsURL = [directory_ URLByAppendingPathComponent:@"assets"];
    NSURL *trashURL = [directory_ URLByAppendingPathComponent:@"trash"];
    NSURL *databaseURL = [directory_ URLByAppendingPathComponent:@"assets.db"];
    ASSERT_TRUE([NSFileManager.defaultManager createDirectoryAtURL:assetsURL
                                       withIntermediateDirectories:YES
                                                        attributes:nil
                                                             error:nil]);
    ASSERT_TRUE([NSFileManager.defaultManager createDirectoryAtURL:trashURL
                                       withIntermediateDirectories:YES
                                                        attributes:nil
                                                             error:nil]);
    ETCoreMLAssetManager *assetManager = [[ETCoreMLAssetManager alloc] initWithDatabaseURL:databaseURL
                                                                        assetsDirectoryURL:assetsURL
                                                                         trashDirectoryURL:trashURL
                                                                      maxAssetsSizeInBytes:2 * 1024 * 1024
                                                                                     error:nil];
    ASSERT_NE(assetManager, nil);
    ETCoreMLModelManager *manager = [[ETCoreMLModelManager alloc] initWithAssetManager:assetManager];
    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = MLComputeUnitsCPUOnly;
    NSData *data = [NSData dataWithBytesNoCopy:blob_.data() length:blob_.size() freeWhenDone:NO];
    ModelHandle *handle = [manager loadModelFromAOTData:data configuration:configuration error:nil];
    ASSERT_NE(handle, nullptr);
    id<ETCoreMLModelExecutor> executor = [manager executorWithHandle:handle];
    ASSERT_NE(executor, nil);
    executor.ignoreOutputBackings = YES;
    ETCoreMLModel *model = [manager modelWithHandle:handle];
    ASSERT_NE(model, nil);

    PixelBufferPtr output = makePixelBuffer(kCVPixelFormatType_OneComponent16Half);
    ASSERT_TRUE([model bindPixelBuffer:output.get() error:nil]);
    // Padded rows make a copy that ignores the row stride fail the checks below.
    const size_t rowLength = CVPixelBufferGetBytesPerRow(output.get()) / sizeof(_Float16);
    ASSERT_GT(rowLength, kWidth);
    // The model doubles values in [0, 8), so its output never holds this.
    constexpr _Float16 kUnwritten = -1;
    std::vector<_Float16> input(kHeight * kWidth);

    for (int frame = 0; frame < 3; ++frame) {
        for (size_t y = 0; y < kHeight; ++y) {
            for (size_t x = 0; x < kWidth; ++x) {
                input[y * kWidth + x] = inputValue(x, y, frame);
            }
        }
        CVPixelBufferLockBaseAddress(output.get(), 0);
        for (size_t y = 0; y < kHeight; ++y) {
            for (size_t x = 0; x < rowLength; ++x) {
                row(output.get(), y)[x] = kUnwritten;
            }
        }
        CVPixelBufferUnlockBaseAddress(output.get(), 0);

        std::vector<MultiArray> args = {
            makeArg(input.data(), kShape),
            makeArg(baseAddress(output.get()), kShape),
        };
        NSError *error = nil;
        ASSERT_TRUE([manager executeModelWithHandle:handle
                                            argsVec:args
                                     loggingOptions:ModelLoggingOptions()
                                        eventLogger:nullptr
                                              error:&error])
            << "frame " << frame;

        CVPixelBufferLockBaseAddress(output.get(), kCVPixelBufferLock_ReadOnly);
        for (size_t y = 0; y < kHeight; ++y) {
            for (size_t x = 0; x < rowLength; ++x) {
                const float expected = x < kWidth ? 2 * inputValue(x, y, frame) : float(kUnwritten);
                EXPECT_EQ(float(row(output.get(), y)[x]), expected)
                    << "frame " << frame << " at (" << x << ", " << y << ")";
            }
        }
        CVPixelBufferUnlockBaseAddress(output.get(), kCVPixelBufferLock_ReadOnly);
    }
    EXPECT_TRUE(executor.ignoreOutputBackings);
    [manager unloadModelWithHandle:handle];
}

TEST_F(PixelBufferBindingTest, PredictsInPlaceThroughBoundBuffers) {
    BackendDelegate::Handle *handle = load();
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)handle;
    ASSERT_NE(model, nil);
    PixelBufferPtr input = makePixelBuffer(kCVPixelFormatType_OneComponent16Half);
    PixelBufferPtr output = makePixelBuffer(kCVPixelFormatType_OneComponent16Half);
    ASSERT_TRUE([model bindPixelBuffer:input.get() error:nil]);
    ASSERT_TRUE([model bindPixelBuffer:output.get() error:nil]);
    // Padded rows make a read or write that ignores the row stride fail the checks below.
    ASSERT_GT(CVPixelBufferGetBytesPerRow(input.get()), kWidth * sizeof(_Float16));
    ASSERT_GT(CVPixelBufferGetBytesPerRow(output.get()), kWidth * sizeof(_Float16));

    for (int frame = 0; frame < 3; ++frame) {
        CVPixelBufferLockBaseAddress(input.get(), 0);
        for (size_t y = 0; y < kHeight; ++y) {
            for (size_t x = 0; x < kWidth; ++x) {
                row(input.get(), y)[x] = inputValue(x, y, frame);
            }
        }
        CVPixelBufferUnlockBaseAddress(input.get(), 0);

        std::vector<MultiArray> args = {
            makeArg(baseAddress(input.get()), kShape),
            makeArg(baseAddress(output.get()), kShape),
        };
        std::error_code error;
        ASSERT_TRUE(delegate_->execute(handle, args, ModelLoggingOptions(), nullptr, error)) << "frame " << frame;

        CVPixelBufferLockBaseAddress(output.get(), kCVPixelBufferLock_ReadOnly);
        for (size_t y = 0; y < kHeight; ++y) {
            for (size_t x = 0; x < kWidth; ++x) {
                EXPECT_EQ(float(row(output.get(), y)[x]), 2 * inputValue(x, y, frame))
                    << "frame " << frame << " at (" << x << ", " << y << ")";
            }
        }
        CVPixelBufferUnlockBaseAddress(output.get(), kCVPixelBufferLock_ReadOnly);
    }
}

TEST_F(PixelBufferBindingTest, BindsIOSurfacesNamedByRuntimeSpecs) {
    PixelBufferPtr pixelBuffer = makePixelBuffer(kCVPixelFormatType_OneComponent16Half);
    IOSurfaceRef surface = IOSurfaceLookup(surfaceID(pixelBuffer.get()));
    if (surface == nullptr) {
        GTEST_SKIP() << "IOSurfaceLookup can't find this process's surfaces here (iOS simulator).";
    }
    CFRelease(surface);

    BackendOptions<1> options;
    options.set_option("iosurface_0", static_cast<int>(surfaceID(pixelBuffer.get())));
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)load(options.view());
    ASSERT_NE(model, nil);
    MLMultiArray *array = nil;
    ASSERT_TRUE([model getPixelBufferMultiArray:&array forArg:makeArg(baseAddress(pixelBuffer.get()), kShape) error:nil]);
    EXPECT_NE(array, nil);

    PixelBufferPtr fp32PixelBuffer = makePixelBuffer(kCVPixelFormatType_OneComponent32Float);
    BackendOptions<1> fp32Options;
    fp32Options.set_option("iosurface_0", static_cast<int>(surfaceID(fp32PixelBuffer.get())));
    EXPECT_EQ(load(fp32Options.view()), nullptr);
}

TEST_F(PixelBufferBindingTest, FailsToLoadWhenARuntimeSpecDoesntNameAnIOSurface) {
    BackendOptions<1> unknownSurface;
    unknownSurface.set_option("iosurface_0", 0);
    EXPECT_EQ(load(unknownSurface.view()), nullptr);

    BackendOptions<1> notAnID;
    notAnID.set_option("iosurface_0", true);
    EXPECT_EQ(load(notAnID.view()), nullptr);
}

TEST_F(PixelBufferBindingTest, BackendRejectsIOSurfaceSpecsWithoutMethodMeta) {
    PixelBufferPtr pixelBuffer = makePixelBuffer(kCVPixelFormatType_OneComponent16Half);
    BackendOptions<1> options;
    options.set_option("iosurface_0", static_cast<int>(surfaceID(pixelBuffer.get())));
    EXPECT_EQ(loadThroughBackend(options.view(), /*methodMeta=*/nullptr).error(), Error::NotSupported);
}

TEST_F(PixelBufferBindingTest, BackendLoadsWithoutIOSurfaceSpecsWithoutMethodMeta) {
    Result<DelegateHandle*> handle = loadThroughBackend({}, /*methodMeta=*/nullptr);
    ASSERT_EQ(handle.error(), Error::Ok);
    backend_.destroy(handle.get());
}

TEST_F(PixelBufferBindingTest, BackendBindsIOSurfaceSpecsWithSingleDelegateMethodMeta) {
    PixelBufferPtr pixelBuffer = makePixelBuffer(kCVPixelFormatType_OneComponent16Half);
    IOSurfaceRef surface = IOSurfaceLookup(surfaceID(pixelBuffer.get()));
    if (surface == nullptr) {
        GTEST_SKIP() << "IOSurfaceLookup can't find this process's surfaces here (iOS simulator).";
    }
    CFRelease(surface);

    BackendOptions<1> options;
    options.set_option("iosurface_0", static_cast<int>(surfaceID(pixelBuffer.get())));
    // Backing bytes own the plan; the MethodMeta view borrows them through init.
    std::vector<uint8_t> bytes = makeMethodMetaBytes(makeSingleCoreMLSpec());
    Result<MethodMeta> meta = MethodMeta::from_serialized_execution_plan(bytes.data(), bytes.size());
    ASSERT_TRUE(meta.ok());
    Result<DelegateHandle*> handle = loadThroughBackend(options.view(), &meta.get());
    ASSERT_EQ(handle.error(), Error::Ok);
    ETCoreMLModel *model = (__bridge ETCoreMLModel *)handle.get();
    MLMultiArray *array = nil;
    EXPECT_TRUE([model getPixelBufferMultiArray:&array forArg:makeArg(baseAddress(pixelBuffer.get()), kShape) error:nil]);
    EXPECT_NE(array, nil);
    backend_.destroy(handle.get());
}

TEST_F(PixelBufferBindingTest, BackendRejectsIOSurfaceSpecsWithInvalidMethodMeta) {
    PixelBufferPtr pixelBuffer = makePixelBuffer(kCVPixelFormatType_OneComponent16Half);
    BackendOptions<1> options;
    options.set_option("iosurface_0", static_cast<int>(surfaceID(pixelBuffer.get())));

    const std::string coreMLId = toString(ETCoreMLStrings.delegateIdentifier);
    const MethodMetaSpec base = makeSingleCoreMLSpec();
    MethodMetaSpec wrongBackend = base;
    wrongBackend.backendIds = {"WrongBackend"};
    MethodMetaSpec multipleDelegates = base;
    multipleDelegates.backendIds = {coreMLId, coreMLId};
    MethodMetaSpec extraInstruction = base;
    extraInstruction.extraFreeCall = true;
    MethodMetaSpec plannedInput = base;
    plannedInput.plannedInput = true;
    MethodMetaSpec plannedOutput = base;
    plannedOutput.plannedOutput = true;
    MethodMetaSpec nonTensorInput = base;
    nonTensorInput.nonTensorInput = true;
    MethodMetaSpec nonTensorOutput = base;
    nonTensorOutput.nonTensorOutput = true;
    MethodMetaSpec constantInput = base;
    constantInput.constantInput = true;
    MethodMetaSpec constantOutput = base;
    constantOutput.constantOutput = true;
    const struct {
        const char* name;
        MethodMetaSpec spec;
    } cases[] = {
        {"wrongBackend", wrongBackend},
        {"multipleDelegates", multipleDelegates},
        {"extraInstruction", extraInstruction},
        {"plannedInput", plannedInput},
        {"plannedOutput", plannedOutput},
        {"nonTensorInput", nonTensorInput},
        {"nonTensorOutput", nonTensorOutput},
        {"constantInput", constantInput},
        {"constantOutput", constantOutput},
    };
    for (const auto& testCase : cases) {
        std::vector<uint8_t> bytes = makeMethodMetaBytes(testCase.spec);
        Result<MethodMeta> meta = MethodMeta::from_serialized_execution_plan(bytes.data(), bytes.size());
        ASSERT_TRUE(meta.ok()) << testCase.name;
        EXPECT_EQ(loadThroughBackend(options.view(), &meta.get()).error(), Error::NotSupported)
            << testCase.name;
    }
}

} // namespace
