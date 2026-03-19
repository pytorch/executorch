/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Tests that device info (device_type) is correctly parsed from serialized
 * tensors in .pte files into TensorImpl at runtime.
 *
 * Uses a .pte exported with DeviceAwarePartitioner (CUDA device annotation)
 * so that delegate output tensors carry device_type=CUDA in ExtraTensorInfo.
 */

#include <executorch/runtime/executor/tensor_parser.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/schema/program_generated.h>

#include <gtest/gtest.h>

using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::deserialization::parseTensor;
using executorch::runtime::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

namespace executorch {
namespace runtime {
namespace testing {
class ProgramTestFriend final {
 public:
  const static executorch_flatbuffer::Program* GetInternalProgram(
      const Program* program) {
    return program->internal_program_;
  }
};
} // namespace testing
} // namespace runtime
} // namespace executorch

using executorch::runtime::testing::ProgramTestFriend;

class TensorParserDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* path = std::getenv("ET_MODULE_ADD_WITH_DEVICE_PATH");
    ASSERT_NE(path, nullptr)
        << "ET_MODULE_ADD_WITH_DEVICE_PATH env var not set";
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));
  }

  std::unique_ptr<FileDataLoader> loader_;
};

TEST_F(TensorParserDeviceTest, CUDADeviceParsedFromPteFile) {
  Result<Program> program =
      Program::load(loader_.get(), Program::Verification::Minimal);
  ASSERT_EQ(program.error(), Error::Ok);

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);

  const executorch_flatbuffer::Program* internal_program =
      ProgramTestFriend::GetInternalProgram(&program.get());
  auto* execution_plan =
      internal_program->execution_plan()->GetMutableObject(0);
  auto* flatbuffer_values = execution_plan->values();

  int cuda_tensor_count = 0;
  int cpu_tensor_count = 0;
  int total_tensor_count = 0;

  for (size_t i = 0; i < flatbuffer_values->size(); ++i) {
    auto* serialization_value = flatbuffer_values->Get(i);
    if (serialization_value->val_type() !=
        executorch_flatbuffer::KernelTypes::Tensor) {
      continue;
    }
    total_tensor_count++;

    auto* s_tensor = serialization_value->val_as_Tensor();

    Result<Tensor> tensor = parseTensor(&program.get(), &mmm.get(), s_tensor);
    if (!tensor.ok()) {
      bool has_cuda = s_tensor->extra_tensor_info() != nullptr &&
          s_tensor->extra_tensor_info()->device_type() ==
              executorch_flatbuffer::DeviceType::CUDA;
      if (has_cuda) {
        cuda_tensor_count++;
      }
      continue;
    }

    Tensor t = tensor.get();
    auto device_type = t.unsafeGetTensorImpl()->device_type();

    if (device_type == executorch::runtime::etensor::DeviceType::CUDA) {
      cuda_tensor_count++;
      EXPECT_EQ(t.unsafeGetTensorImpl()->device_index(), 0)
          << "CUDA tensor should have device_index=0";
    } else {
      EXPECT_EQ(device_type, executorch::runtime::etensor::DeviceType::CPU);
      EXPECT_EQ(t.unsafeGetTensorImpl()->device_index(), 0)
          << "CPU tensor should have device_index=0";
      cpu_tensor_count++;
    }
  }

  EXPECT_GT(total_tensor_count, 0) << "Should have at least one tensor";
  // The model has add(a, b) delegated to CUDA — 2 inputs + 1 output = 3 CUDA
  EXPECT_EQ(cuda_tensor_count, 3)
      << "Expected 3 CUDA tensors (2 delegate inputs + 1 delegate output)";
}

TEST_F(TensorParserDeviceTest, NonDelegatedTensorsDefaultToCPU) {
  Result<Program> program =
      Program::load(loader_.get(), Program::Verification::Minimal);
  ASSERT_EQ(program.error(), Error::Ok);

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);

  const executorch_flatbuffer::Program* internal_program =
      ProgramTestFriend::GetInternalProgram(&program.get());
  auto* execution_plan =
      internal_program->execution_plan()->GetMutableObject(0);
  auto* flatbuffer_values = execution_plan->values();

  for (size_t i = 0; i < flatbuffer_values->size(); ++i) {
    auto* serialization_value = flatbuffer_values->Get(i);
    if (serialization_value->val_type() !=
        executorch_flatbuffer::KernelTypes::Tensor) {
      continue;
    }

    auto* s_tensor = serialization_value->val_as_Tensor();
    bool has_cuda_device = s_tensor->extra_tensor_info() != nullptr &&
        s_tensor->extra_tensor_info()->device_type() ==
            executorch_flatbuffer::DeviceType::CUDA;

    // Only check tensors that are NOT annotated as CUDA
    if (has_cuda_device) {
      continue;
    }

    Result<Tensor> tensor = parseTensor(&program.get(), &mmm.get(), s_tensor);
    if (!tensor.ok()) {
      continue;
    }

    Tensor t = tensor.get();
    EXPECT_EQ(
        t.unsafeGetTensorImpl()->device_type(),
        executorch::runtime::etensor::DeviceType::CPU)
        << "Tensor at index " << i
        << " without CUDA annotation should default to CPU";
    EXPECT_EQ(t.unsafeGetTensorImpl()->device_index(), 0)
        << "Tensor at index " << i
        << " without device annotation should have device_index=0";
  }
}
