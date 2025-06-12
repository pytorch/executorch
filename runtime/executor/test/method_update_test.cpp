/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <filesystem>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/backend/backend_options.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/backend_update_context.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/executor/test/stub_backend.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::BackendOptionsMap;
using executorch::runtime::BackendUpdateContext;
using executorch::runtime::BoolKey;
using executorch::runtime::CompileSpec;
using executorch::runtime::DataLoader;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Entry;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::IntKey;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024U;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024U;

class MethodUpdateTest : public ::testing::Test {
 protected:
  void load_program() {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Create a loader for the serialized program.
    ASSERT_EQ(StubBackend::register_singleton(), Error::Ok);

    auto loader_res =
        FileDataLoader::from(std::getenv("ET_MODULE_ADD_MUL_DELEGATED_PATH"));
    ASSERT_EQ(loader_res.error(), Error::Ok);
    loader_ = std::make_unique<FileDataLoader>(std::move(loader_res.get()));

    // Use it to load the program.
    auto program_res = Program::load(loader_.get());
    ASSERT_EQ(program_res.error(), Error::Ok);
    program_ = std::make_unique<Program>(std::move(program_res.get()));
  }

  void SetUp() override {
    executorch::runtime::runtime_init();

    load_program();
  }

 private:
  std::unique_ptr<FileDataLoader> loader_;

 protected:
  std::unique_ptr<Program> program_;
};

TEST_F(MethodUpdateTest, MoveTest) {
  BackendInterface* backend =
      executorch::runtime::get_backend_class(StubBackend::kName);
  ASSERT_EQ(backend, &StubBackend::singleton());

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = program_->load_method("forward", &mmm.get());
  //  Check that the default number of threads is 1.
  ASSERT_EQ(StubBackend::singleton().num_threads(), 1);
  ASSERT_EQ(method.error(), Error::Ok);

  BackendOptionsMap<3> map;
  BackendOptions<1> backend_options;
  int new_num_threads = 4;
  backend_options.set_option(IntKey("NumberOfThreads"), new_num_threads);
  map.add("StubBackend", backend_options.view());
  Error update_result = method->update(map.entries());
  ASSERT_EQ(update_result, Error::Ok);
  ASSERT_EQ(StubBackend::singleton().num_threads(), new_num_threads);
}
