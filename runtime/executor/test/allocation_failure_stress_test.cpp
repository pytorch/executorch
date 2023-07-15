#include <cstdlib>
#include <filesystem>
#include <memory>

#include <executorch/core/Constants.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/executor.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/file_data_loader.h>
#include <executorch/util/util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::Scalar;
using exec_aten::Tensor;
using torch::executor::Error;
using torch::executor::Executor;
using torch::executor::kKB;
using torch::executor::MemoryAllocator;
using torch::executor::MemoryManager;
using torch::executor::Program;
using torch::executor::Result;
using torch::executor::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * kKB;
constexpr size_t kDefaultRuntimeMemBytes = 32 * kKB;

class AllocationFailureStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a loader for the serialized ModuleAdd program.
    const char* path = std::getenv("ET_MODULE_ADD_PATH");
    Result<FileDataLoader> loader = FileDataLoader::From(path);
    ASSERT_EQ(loader.error(), Error::Ok);
    loader_ = std::make_unique<FileDataLoader>(std::move(loader.get()));

    // Use it to load the program.
    Result<Program> program = Program::Load(
        loader_.get(), Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);
    program_ = std::make_unique<Program>(std::move(program.get()));
  }

 private:
  // Must outlive program_, but tests shouldn't need to touch it.
  std::unique_ptr<FileDataLoader> loader_;

 protected:
  std::unique_ptr<Program> program_;
};

/**
 * Slowly increases the amount of available runtime memory until
 * init_execution_plan() and execute() succeed. This should cause every runtime
 * allocation to fail at some point, exercising every allocation failure path
 * reachable by the test model.
 */
TEST_F(AllocationFailureStressTest, End2EndIncreaseRuntimeMemUntilSuccess) {
  size_t runtime_mem_bytes = 0;
  Error err = Error::Internal;
  size_t num_init_failures = 0;
  while (runtime_mem_bytes < kDefaultRuntimeMemBytes && err != Error::Ok) {
    ManagedMemoryManager mmm(kDefaultNonConstMemBytes, runtime_mem_bytes);
    Executor executor(program_.get(), &mmm.get());

    // Initialization should fail several times from allocation failures.
    err = executor.init_execution_plan();
    if (err != Error::Ok) {
      runtime_mem_bytes += sizeof(size_t);
      num_init_failures++;
      continue;
    }

    // Execution does not use the runtime allocator, so it should always succeed
    // once init was successful.
    exec_aten::ArrayRef<void*> inputs =
        torch::executor::util::PrepareInputTensors(executor.execution_plan());
    err = executor.execution_plan().execute();
    torch::executor::util::FreeInputs(inputs);
    ASSERT_EQ(err, Error::Ok);
  }
  EXPECT_GT(num_init_failures, 0) << "Expected at least some failures";
  EXPECT_EQ(err, Error::Ok)
      << "Did not succeed after increasing runtime_mem_bytes to "
      << runtime_mem_bytes;
}

/**
 * Slowly increases the amount of available non-constant memory until
 * init_execution_plan() and execute() succeed. This should cause every
 * non-const allocation to fail at some point, exercising every allocation
 * failure path reachable by the test model.
 */
TEST_F(AllocationFailureStressTest, End2EndNonConstantMemUntilSuccess) {
  size_t non_constant_mem_bytes = 0;
  Error err = Error::Internal;
  size_t num_init_failures = 0;
  while (non_constant_mem_bytes < kDefaultNonConstMemBytes &&
         err != Error::Ok) {
    ManagedMemoryManager mmm(non_constant_mem_bytes, kDefaultRuntimeMemBytes);
    Executor executor(program_.get(), &mmm.get());

    // Initialization should fail several times from allocation failures.
    err = executor.init_execution_plan();
    if (err != Error::Ok) {
      non_constant_mem_bytes += sizeof(size_t);
      num_init_failures++;
      continue;
    }

    // Execution does not use the runtime allocator, so it should always succeed
    // once init was successful.
    exec_aten::ArrayRef<void*> inputs =
        torch::executor::util::PrepareInputTensors(executor.execution_plan());
    err = executor.execution_plan().execute();
    torch::executor::util::FreeInputs(inputs);
    ASSERT_EQ(err, Error::Ok);
  }
  EXPECT_GT(num_init_failures, 0) << "Expected at least some failures";
  EXPECT_EQ(err, Error::Ok)
      << "Did not succeed after increasing non_constant_mem_bytes to "
      << non_constant_mem_bytes;
}
