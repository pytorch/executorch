/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::runtime::Error;

static uint8_t* executed_scratch = nullptr;
static bool platform_needs_scratch = true;

namespace executorch::backends::arm {

Error platform_init(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>,
    executorch::runtime::MemoryAllocator*,
    ExecutionHandle*) {
  return Error::Ok;
}

void platform_destroy(PlatformState*) {}

bool needs_scratch_allocation() {
  return platform_needs_scratch;
}

Error platform_execute(
    executorch::runtime::BackendExecutionContext&,
    const ExecutionHandle*,
    const VelaHandles&,
    int,
    int,
    executorch::runtime::Span<executorch::runtime::EValue*>,
    char* ethosu_scratch) {
  executed_scratch = reinterpret_cast<uint8_t*>(ethosu_scratch);
  return Error::Ok;
}

} // namespace executorch::backends::arm

namespace {

using namespace executorch::backends::arm;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Span;

class EthosUPlannedScratchTest : public ::testing::Test {
 protected:
  alignas(16) uint8_t scratch_[64]{};
  ExecutionHandle handle_;

  void SetUp() override {
    executorch::runtime::runtime_init();
    platform_needs_scratch = true;
    executed_scratch = nullptr;
    static constexpr char command[] = "COP1";
    handle_.handles.cmd_data = command;
    handle_.handles.cmd_data_size = 4;
    handle_.handles.scratch_data_size = sizeof(scratch_);
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
        ->execute(context, &handle_, {});
  }
};

// cppcheck-suppress syntaxError
TEST_F(EthosUPlannedScratchTest, UsesContextScratchWithNoTemporaryMemory) {
  Span<uint8_t> buffers[]{Span<uint8_t>(scratch_)};
  MemoryAllocator empty_allocator(0, nullptr);
  for (int invocation = 0; invocation < 2; ++invocation) {
    ASSERT_EQ(
        execute(Span<const Span<uint8_t>>(buffers), &empty_allocator),
        Error::Ok);
    EXPECT_EQ(executed_scratch, scratch_);
    EXPECT_EQ(empty_allocator.used_size(), 0);
  }
}

TEST_F(
    EthosUPlannedScratchTest,
    AllocatesTemporaryScratchWithoutContextScratch) {
  alignas(16) uint8_t temporary_memory[sizeof(scratch_)]{};
  MemoryAllocator temp_allocator(sizeof(temporary_memory), temporary_memory);
  for (int invocation = 0; invocation < 2; ++invocation) {
    temp_allocator.reset();
    ASSERT_EQ(execute({}, &temp_allocator), Error::Ok);
    EXPECT_EQ(executed_scratch, temporary_memory);
    EXPECT_EQ(temp_allocator.used_size(), sizeof(scratch_));
  }
}

TEST_F(
    EthosUPlannedScratchTest,
    DriverManagedScratchHandlesOptionalContextReservation) {
  platform_needs_scratch = false;
  Span<uint8_t> buffers[]{Span<uint8_t>(scratch_)};
  MemoryAllocator empty_allocator(0, nullptr);
  for (bool has_planned_scratch : {false, true}) {
    const auto planned = has_planned_scratch
        ? Span<const Span<uint8_t>>(buffers)
        : Span<const Span<uint8_t>>();
    for (int invocation = 0; invocation < 2; ++invocation) {
      ASSERT_EQ(execute(planned, &empty_allocator), Error::Ok);
      EXPECT_EQ(executed_scratch, nullptr);
      EXPECT_EQ(empty_allocator.used_size(), 0);
    }
  }
}

TEST_F(EthosUPlannedScratchTest, RejectsUndersizedContextScratch) {
  Span<uint8_t> buffers[]{Span<uint8_t>(scratch_, sizeof(scratch_) - 1)};
  EXPECT_EQ(execute(Span<const Span<uint8_t>>(buffers)), Error::InvalidProgram);
}

TEST_F(EthosUPlannedScratchTest, RejectsNullContextScratch) {
#ifndef NDEBUG
  GTEST_SKIP() << "A nonempty null Span violates its debug contract";
#else
  Span<uint8_t> buffers[]{Span<uint8_t>(nullptr, sizeof(scratch_))};
  EXPECT_EQ(execute(Span<const Span<uint8_t>>(buffers)), Error::InvalidProgram);
#endif
}

TEST_F(EthosUPlannedScratchTest, RejectsUnalignedContextScratch) {
  alignas(16) uint8_t buffer[65]{};
  Span<uint8_t> buffers[]{Span<uint8_t>(buffer + 1, 64)};
  EXPECT_EQ(execute(Span<const Span<uint8_t>>(buffers)), Error::InvalidProgram);
}

TEST_F(EthosUPlannedScratchTest, RejectsUnexpectedContextScratchCount) {
  Span<uint8_t> buffers[]{Span<uint8_t>(scratch_), Span<uint8_t>(scratch_)};
  EXPECT_EQ(execute(Span<const Span<uint8_t>>(buffers)), Error::InvalidProgram);
}

TEST_F(
    EthosUPlannedScratchTest,
    ReportsInsufficientTemporaryMemoryWithoutContextScratch) {
  EXPECT_EQ(execute({}), Error::MemoryAllocationFailed);
}

TEST_F(EthosUPlannedScratchTest, ZeroScratchSizeNeedsNoTemporaryMemory) {
  handle_.handles.scratch_data_size = 0;
  MemoryAllocator empty_allocator(0, nullptr);
  ASSERT_EQ(execute({}, &empty_allocator), Error::Ok);
  EXPECT_EQ(executed_scratch, nullptr);
  EXPECT_EQ(empty_allocator.used_size(), 0);
}

} // namespace
