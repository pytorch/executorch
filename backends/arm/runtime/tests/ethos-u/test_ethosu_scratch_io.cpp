/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstring>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::runtime::Error;

static int io_copy_count = 0;
static uint8_t* executed_scratch = nullptr;

extern "C" void
arm_ethos_io_memcpy(void* dest, const void* source, size_t size) {
  EXPECT_NE(dest, source);
  ++io_copy_count;
  std::memcpy(dest, source, size);
}

namespace executorch::backends::arm {

Error platform_init(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>,
    executorch::runtime::MemoryAllocator*,
    ExecutionHandle*) {
  return Error::Ok;
}

void platform_destroy(PlatformState*) {}

bool needs_scratch_allocation() {
  return true;
}

Error platform_execute(
    executorch::runtime::BackendExecutionContext&,
    const ExecutionHandle*,
    IOPlan& plan) {
  executed_scratch = plan.scratch.data();
  if (plan.needs_input_copy) {
    plan.input_region_base = reinterpret_cast<char*>(plan.scratch.data());
  }
  if (plan.needs_output_copy) {
    plan.contiguous_output_base = reinterpret_cast<char*>(plan.scratch.data());
  }
  const Error status = copy_inputs(plan);
  if (status != Error::Ok) {
    return status;
  }
  const auto& handles = *plan.handles;
  for (size_t output = 0; output < plan.arguments.outputs.size(); ++output) {
    for (size_t i = 0; i < plan.arguments.outputs[output]->toTensor().nbytes();
         ++i) {
      plan.scratch[handles.outputs->io[output].offset + i] =
          plan.scratch[handles.inputs->io[0].offset + i] + 42;
    }
  }
  return copy_outputs(plan);
}

} // namespace executorch::backends::arm

namespace {

using namespace executorch::backends::arm;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Span;

class EthosUScratchIOTest : public ::testing::Test {
 protected:
  struct SingleIO {
    int count = 1;
    VelaIO io = {{1, 1, 1, 1, 1, 16}, 1, 0, 1};
  } inputs_, outputs_;
  alignas(16) uint8_t scratch_[64]{};
  executorch::runtime::etensor::TensorImpl::SizesType sizes_[1]{16};
  executorch::runtime::etensor::TensorImpl::DimOrderType order_[1]{0};
  executorch::runtime::etensor::TensorImpl input_impl_{
      executorch::aten::ScalarType::Byte,
      1,
      sizes_,
      scratch_,
      order_};
  executorch::runtime::etensor::TensorImpl output_impl_{
      executorch::aten::ScalarType::Byte,
      1,
      sizes_,
      scratch_ + 16,
      order_};
  EValue input_{executorch::aten::Tensor(&input_impl_)};
  EValue output_{executorch::aten::Tensor(&output_impl_)};
  EValue* args_[2]{&input_, &output_};
  ExecutionHandle handle_;

  void SetUp() override {
    executorch::runtime::runtime_init();
    io_copy_count = 0;
    executed_scratch = nullptr;
    static constexpr char command[] = "COP1";
    outputs_.io.offset = 16;
    handle_.handles.cmd_data = command;
    handle_.handles.cmd_data_size = 4;
    handle_.handles.scratch_data_size = sizeof(scratch_);
    handle_.handles.inputs = reinterpret_cast<VelaIOs*>(&inputs_);
    handle_.handles.outputs = reinterpret_cast<VelaIOs*>(&outputs_);
  }

  Error execute(
      Span<const Span<uint8_t>> scratch,
      MemoryAllocator* temp_allocator = nullptr) {
    MemoryAllocator empty_allocator(0, nullptr);
    BackendExecutionContext context(
        nullptr,
        temp_allocator == nullptr ? &empty_allocator : temp_allocator,
        nullptr,
        scratch);
    return executorch::runtime::get_backend_class("EthosUBackend")
        ->execute(context, &handle_, Span<EValue*>(args_));
  }
};

// cppcheck-suppress syntaxError
TEST_F(EthosUScratchIOTest, UsesContextScratchWithNoTemporaryMemory) {
  Span<uint8_t> buffers[]{Span<uint8_t>(scratch_)};
  EXPECT_EQ(execute(Span<const Span<uint8_t>>(buffers)), Error::Ok);
  EXPECT_EQ(output_.toTensor().const_data_ptr<uint8_t>()[0], 42);
  EXPECT_EQ(io_copy_count, 0);
  EXPECT_EQ(executed_scratch, scratch_);
}

TEST_F(EthosUScratchIOTest, CopiesUnaliasedIOWithContextScratch) {
  uint8_t input_data[16];
  uint8_t output_data[16]{};
  input_impl_.set_data(input_data);
  output_impl_.set_data(output_data);
  Span<uint8_t> buffers[]{Span<uint8_t>(scratch_)};
  MemoryAllocator empty_allocator(0, nullptr);

  for (int invocation = 0; invocation < 2; ++invocation) {
    for (size_t i = 0; i < sizeof(input_data); ++i) {
      input_data[i] = i + invocation;
    }
    ASSERT_EQ(
        execute(Span<const Span<uint8_t>>(buffers), &empty_allocator),
        Error::Ok);
    for (size_t i = 0; i < sizeof(output_data); ++i) {
      EXPECT_EQ(output_data[i], input_data[i] + 42);
    }
    EXPECT_EQ(executed_scratch, scratch_);
    EXPECT_EQ(io_copy_count, 2 * (invocation + 1));
    EXPECT_EQ(empty_allocator.used_size(), 0);
  }
}

TEST_F(EthosUScratchIOTest, CopiesLegacyIOWithTemporaryScratch) {
  uint8_t input_data[16];
  uint8_t output_data[16]{};
  input_impl_.set_data(input_data);
  output_impl_.set_data(output_data);
  alignas(16) uint8_t temporary_memory[sizeof(scratch_)]{};
  MemoryAllocator temp_allocator(sizeof(temporary_memory), temporary_memory);

  for (int invocation = 0; invocation < 2; ++invocation) {
    temp_allocator.reset();
    for (size_t i = 0; i < sizeof(input_data); ++i) {
      input_data[i] = i + invocation;
    }
    ASSERT_EQ(execute({}, &temp_allocator), Error::Ok);
    for (size_t i = 0; i < sizeof(output_data); ++i) {
      EXPECT_EQ(output_data[i], input_data[i] + 42);
    }
    EXPECT_EQ(executed_scratch, temporary_memory);
    EXPECT_EQ(io_copy_count, 2 * (invocation + 1));
    EXPECT_EQ(temp_allocator.used_size(), sizeof(scratch_));
  }
}

TEST_F(EthosUScratchIOTest, CopiesOnlyInputsOutsideScratch) {
  uint8_t external_data[16]{7};
  executorch::runtime::etensor::TensorImpl external_impl(
      executorch::aten::ScalarType::Byte, 1, sizes_, external_data, order_);
  EValue external{executorch::aten::Tensor(&external_impl)};
  EValue* args[]{&input_, &external};
  struct {
    int count;
    VelaIO io[2];
  } inputs{2, {inputs_.io, outputs_.io}};
  handle_.handles.inputs = reinterpret_cast<VelaIOs*>(&inputs);
  handle_.handles.outputs = nullptr;
  Span<uint8_t> buffers[]{Span<uint8_t>(scratch_)};
  MemoryAllocator empty_allocator(0, nullptr);
  BackendExecutionContext context(
      nullptr, &empty_allocator, nullptr, Span<const Span<uint8_t>>(buffers));
  EXPECT_EQ(
      executorch::runtime::get_backend_class("EthosUBackend")
          ->execute(context, &handle_, Span<EValue*>(args)),
      Error::Ok);

  EXPECT_EQ(io_copy_count, 1);
  EXPECT_EQ(scratch_[16], 7);
}

} // namespace
