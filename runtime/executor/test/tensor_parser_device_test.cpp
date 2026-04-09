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
#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/device_memory_buffer.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/test/mock_cuda_allocator.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/program_generated.h>

#include <gtest/gtest.h>

using executorch::aten::Tensor;
using executorch::runtime::DeviceAllocator;
using executorch::runtime::DeviceMemoryBuffer;
using executorch::runtime::Error;
using executorch::runtime::get_device_allocator;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::register_device_allocator;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::deserialization::parseTensor;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;
using executorch::runtime::testing::ManagedMemoryManager;
using executorch::runtime::testing::MockCudaAllocator;
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

static MockCudaAllocator g_mock_cuda;

class TensorParserDeviceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
    register_device_allocator(DeviceType::CUDA, &g_mock_cuda);
  }

  void SetUp() override {
    const char* path = std::getenv("ET_MODULE_ADD_WITH_DEVICE_PATH");
    ASSERT_NE(path, nullptr)
        << "ET_MODULE_ADD_WITH_DEVICE_PATH env var not set";
    Result<FileDataLoader> loader = FileDataLoader::from(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));

    g_mock_cuda.allocate_count_ = 0;
    g_mock_cuda.deallocate_count_ = 0;
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

  for (uint32_t i = 0; i < flatbuffer_values->size(); ++i) {
    auto* serialization_value = flatbuffer_values->Get(i);
    if (serialization_value->val_type() !=
        executorch_flatbuffer::KernelTypes::Tensor) {
      continue;
    }

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

  EXPECT_EQ(cuda_tensor_count, 3)
      << "Expected 3 CUDA tensors (2 delegate inputs + 1 delegate output)";
  // Device-aware memory planning may introduce CPU-side tensors
  // (e.g. original inputs before H2D copies), so we no longer
  // require cpu_tensor_count == 0.
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

  for (uint32_t i = 0; i < flatbuffer_values->size(); ++i) {
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
TEST_F(TensorParserDeviceTest, CudaTensorDataPtrPointsToDeviceMemory) {
  Result<Program> program =
      Program::load(loader_.get(), Program::Verification::Minimal);
  ASSERT_EQ(program.error(), Error::Ok);

  Result<MethodMeta> method_meta = program->method_meta("forward");
  ASSERT_EQ(method_meta.error(), Error::Ok);

  // ModuleAddWithDevice has planned buffers that may include both CPU and CUDA
  // entries when device-aware memory planning creates separate buffers per
  // device type.
  const size_t num_buffers = method_meta->num_memory_planned_buffers();
  ASSERT_GE(num_buffers, 1);

  // Set up device-aware planned memory.
  std::vector<Span<uint8_t>> planned_spans;
  std::vector<std::vector<uint8_t>> cpu_buffers;
  std::vector<DeviceMemoryBuffer> device_buffers;

  for (size_t i = 0; i < num_buffers; ++i) {
    auto size = method_meta->memory_planned_buffer_size(i);
    ASSERT_TRUE(size.ok());
    auto device = method_meta->memory_planned_buffer_device(i);
    ASSERT_TRUE(device.ok());

    if (device->is_cpu()) {
      cpu_buffers.emplace_back(size.get());
      planned_spans.emplace_back(
          cpu_buffers.back().data(), cpu_buffers.back().size());
    } else {
      cpu_buffers.emplace_back(); // empty placeholder
      auto dmb = DeviceMemoryBuffer::create(
          size.get(), device->type(), device->index());
      ASSERT_TRUE(dmb.ok())
          << "DeviceMemoryBuffer::create failed for buffer " << i;
      planned_spans.emplace_back(dmb->as_span());
      device_buffers.push_back(std::move(dmb.get()));
    }
  }

  ASSERT_EQ(g_mock_cuda.allocate_count_, 1);

  // Build HierarchicalAllocator with mixed CPU/device spans.
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  constexpr size_t kMethodAllocBytes = 32 * 1024U;
  auto method_alloc_pool = std::make_unique<uint8_t[]>(kMethodAllocBytes);
  MemoryAllocator method_allocator(kMethodAllocBytes, method_alloc_pool.get());
  MemoryManager memory_manager(&method_allocator, &planned_memory);

  // Parse tensors and verify CUDA tensors have device memory.
  const executorch_flatbuffer::Program* internal_program =
      ProgramTestFriend::GetInternalProgram(&program.get());
  auto* execution_plan =
      internal_program->execution_plan()->GetMutableObject(0);
  auto* flatbuffer_values = execution_plan->values();

  int cuda_with_device_memory = 0;

  for (size_t i = 0; i < flatbuffer_values->size(); ++i) {
    auto* serialization_value = flatbuffer_values->Get(i);
    if (serialization_value->val_type() !=
        executorch_flatbuffer::KernelTypes::Tensor) {
      continue;
    }

    auto* s_tensor = serialization_value->val_as_Tensor();
    bool is_cuda = s_tensor->extra_tensor_info() != nullptr &&
        s_tensor->extra_tensor_info()->device_type() ==
            executorch_flatbuffer::DeviceType::CUDA;

    Result<Tensor> tensor =
        parseTensor(&program.get(), &memory_manager, s_tensor);
    ASSERT_TRUE(tensor.ok())
        << "parseTensor failed at index " << i << " with error 0x" << std::hex
        << static_cast<uint32_t>(tensor.error());

    Tensor t = tensor.get();

    if (is_cuda && t.unsafeGetTensorImpl()->device_type() == DeviceType::CUDA) {
      EXPECT_TRUE(g_mock_cuda.is_device_ptr(t.const_data_ptr()))
          << "CUDA tensor at index " << i
          << " should have data_ptr in device memory, but got CPU memory";
      cuda_with_device_memory++;
    }
  }

  // All 3 CUDA tensors (2 inputs + 1 output of the delegate) should have
  // their data_ptr pointing to the mock device memory buffer.
  EXPECT_EQ(cuda_with_device_memory, 3)
      << "All 3 CUDA tensors should have data_ptr in device memory";
}
