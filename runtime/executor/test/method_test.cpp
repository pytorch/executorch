#include <cstdlib>
#include <filesystem>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/util.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using torch::executor::Error;
using torch::executor::EValue;
using torch::executor::Method;
using torch::executor::Program;
using torch::executor::Result;
using torch::executor::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

class MethodTest : public ::testing::Test {
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

TEST_F(MethodTest, MoveTest) {
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = program_->load_method("forward", &mmm.get());
  ASSERT_EQ(method.error(), Error::Ok);

  // Can execute the method.
  exec_aten::ArrayRef<void*> inputs =
      torch::executor::util::PrepareInputTensors(*method);
  Error err = method->execute();
  ASSERT_EQ(err, Error::Ok);

  // Move into a new Method.
  Method new_method(std::move(method.get()));

  // Can't execute the old method.
  err = method->execute();
  ASSERT_NE(err, Error::Ok);

  // Can execute the new method.
  err = new_method.execute();
  ASSERT_EQ(err, Error::Ok);

  torch::executor::util::FreeInputs(inputs);
}
