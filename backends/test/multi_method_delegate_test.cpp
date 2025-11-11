#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <executorch/backends/xnnpack/runtime/XNNPACKBackend.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/extension/runner_util/inputs.h>

using executorch::backends::xnnpack::workspace_sharing_mode_option_key;
using executorch::backends::xnnpack::WorkspaceSharingMode;
using executorch::backends::xnnpack::xnnpack_backend_key;

using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

using executorch::extension::FileDataLoader;
using executorch::extension::MallocMemoryAllocator;
using executorch::extension::prepare_input_tensors;

/*
 * Backend agnostic base class.
 */
class ETPTEMethodRunBaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }

  // Runs the PTE e2e without using outside resources.
  // This will run in a single thread.
  // TODO(T208989128) - Add Synchronizer based run method.
  void run(
      const int id,
      const std::string& kTestPTEPath,
      const std::string& kMethodName,
      std::atomic<size_t>& count) const {
    Result<FileDataLoader> loader = FileDataLoader::from(kTestPTEPath.c_str());
    ASSERT_EQ(loader.error(), Error::Ok);

    Result<Program> program = Program::load(
        &loader.get(), Program::Verification::InternalConsistency);
    ASSERT_EQ(program.error(), Error::Ok);

    Result<MethodMeta> method_meta = program->method_meta(kMethodName.c_str());
    ASSERT_EQ(method_meta.error(), Error::Ok);

    const size_t num_memory_planned_buffers =
        method_meta->num_memory_planned_buffers();

    std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
    std::vector<Span<uint8_t>> planned_spans;
    for (size_t i = 0; i < num_memory_planned_buffers; ++i) {
      const size_t buffer_size =
          static_cast<size_t>(method_meta->memory_planned_buffer_size(i).get());
      planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
      planned_spans.push_back({planned_buffers.back().get(), buffer_size});
    }

    auto method_allocator = std::make_unique<MallocMemoryAllocator>();
    auto memory_planned_allocator = std::make_unique<HierarchicalAllocator>(
        Span(planned_spans.data(), planned_spans.size()));
    auto temp_allocator = std::make_unique<MallocMemoryAllocator>();

    auto memory_manager = std::make_unique<MemoryManager>(
        method_allocator.get(),
        memory_planned_allocator.get(),
        temp_allocator.get());

    Result<Method> method =
        program->load_method(kMethodName.c_str(), memory_manager.get());
    ASSERT_EQ(method.error(), Error::Ok);

    auto inputs = prepare_input_tensors(*method);
    ASSERT_EQ(inputs.error(), Error::Ok);

    Error err = method->execute();
    for (int i = 0; i < id % 7; i++) {
      err = method->execute();
      ASSERT_EQ(err, Error::Ok);
    }

    std::vector<EValue> outputs(method->outputs_size());
    err = method->get_outputs(outputs.data(), outputs.size());
    ET_CHECK(err == Error::Ok);
    // TODO(T208989129) - Add validation of outputs using bundled
    // inputs/outputs.
    count++;
  }
};

class XNNPACKMultiDelegateTest : public ETPTEMethodRunBaseTest {
 protected:
  std::string kTestPTE1Path, kTestPTE2Path;
  std::string kMethodName;
  int num_threads;

  void SetUp() override {
    ETPTEMethodRunBaseTest::SetUp();
    const char* pte1_path =
        std::getenv("ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH");
    if (pte1_path == nullptr) {
      std::cerr << "ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH is not set"
                << std::endl;
      FAIL();
    }
    kTestPTE1Path = std::string(pte1_path);

    const char* pte2_path =
        std::getenv("ET_XNNPACK_GENERATED_SUB_LARGE_PTE_PATH");
    if (pte1_path == nullptr) {
      std::cerr << "ET_XNNPACK_GENERATED_SUB_LARGE_PTE_PATH is not set"
                << std::endl;
      FAIL();
    }
    kTestPTE2Path = std::string(pte2_path);

    num_threads = 40;
    kMethodName = "forward";
  }

  // This test is to validate the assumption that the delegate is thread safe.
  // That includes the following:
  // 1. The delegate can be initilized by multiple threads in parallel.
  // 2. The delegate can be executed by multiple threads in parallel.
  // 3. The delegate can be destroyed by multiple threads in parallel.
  // Regardless of the underlying implementation of the delegate.
  // This is particularly important when we have shared resources across
  // delegate instances through a singleton backend instance.
  void runStressTest() {
    ASSERT_NE(kTestPTE1Path.size(), 0);
    ASSERT_NE(kTestPTE2Path.size(), 0);
    ASSERT_NE(num_threads, 0);
    ASSERT_NE(kMethodName.size(), 0);

    std::vector<std::thread> threads(num_threads);
    std::atomic<size_t> count{0};

    for (int i = 0; i < num_threads; i++) {
      threads[i] = std::thread([&, i]() {
        run(i, i % 7 ? kTestPTE1Path : kTestPTE2Path, kMethodName, count);
      });
    }
    for (int i = 0; i < num_threads; i++) {
      threads[i].join();
    }
    ASSERT_EQ(count, num_threads);
  }

  void setWorkspaceSharingMode(WorkspaceSharingMode mode) {
    executorch::runtime::runtime_init();

    BackendOptions<1> backend_options;
    backend_options.set_option(
        workspace_sharing_mode_option_key, static_cast<int>(mode));

    auto status = executorch::runtime::set_option(
        xnnpack_backend_key, backend_options.view());
    ASSERT_EQ(status, Error::Ok);
  }
};

TEST_F(XNNPACKMultiDelegateTest, MultipleThreadsSharingDisabled) {
  setWorkspaceSharingMode(WorkspaceSharingMode::Disabled);
  runStressTest();
}

TEST_F(XNNPACKMultiDelegateTest, MultipleThreadsPerModelSharing) {
  setWorkspaceSharingMode(WorkspaceSharingMode::PerModel);
  runStressTest();
}

TEST_F(XNNPACKMultiDelegateTest, MultipleThreadsGlobalSharing) {
  setWorkspaceSharingMode(WorkspaceSharingMode::Global);
  runStressTest();
}

// TODO(T208989291): Add more tests here. For example,
// - PTEs with multiple methods
// - PTEs with proucer and consumer relationships in different threads
// - PTEs with more than 1 delegate instances
// - PTEs with different type of delegate instances
// - Add more patterns of delegate initialization and execution
