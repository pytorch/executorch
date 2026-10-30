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
#import <backend_delegate.h>
#import <inmemory_filesystem_utils.hpp>
#import <model_logging_options.h>
#import <multiarray.h>

#include <executorch/runtime/platform/runtime.h>
#include <format/Model.pb.h>
#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

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

    BackendDelegate::Handle *load() {
        // In the iOS simulator, Core ML reads pixel buffer inputs as zeros unless it runs on the CPU.
        const std::string computeUnits = toString(ETCoreMLStrings.cpuComputeUnitName);
        const std::unordered_map<std::string, Buffer> specs = {
            {toString(ETCoreMLStrings.computeUnitsKeyName), Buffer(computeUnits.data(), computeUnits.size())},
        };
        return delegate_->init(Buffer(blob_.data(), blob_.size()), specs);
    }

    NSURL *directory_ = nil;
    std::string blob_;
    std::shared_ptr<BackendDelegate> delegate_;
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

} // namespace
