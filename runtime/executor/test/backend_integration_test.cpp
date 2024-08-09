/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>
#include <executorch/test/utils/alignment.h>
#include <executorch/util/util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DataLoader;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::PyTorchBackendInterface;
using executorch::runtime::Result;
using executorch::runtime::testing::ManagedMemoryManager;
using torch::executor::util::FileDataLoader;

/**
 * A backend class whose methods can be overridden individually.
 */
class StubBackend final : public PyTorchBackendInterface {
 public:
  // Function signature types that match the PyTorchBackendInterface methods.
  using IsAvailableFn = std::function<bool()>;
  using InitFn = std::function<Result<DelegateHandle*>(
      FreeableBuffer*,
      ArrayRef<CompileSpec>,
      MemoryAllocator*)>;
  using ExecuteFn = std::function<Error(DelegateHandle*, EValue**)>;
  using DestroyFn = std::function<void(DelegateHandle*)>;

  // Default name that this backend is registered as.
  static constexpr char kName[] = "StubBackend";

  void install_is_available(IsAvailableFn fn) {
    is_available_fn_ = fn;
  }

  bool is_available() const override {
    if (is_available_fn_) {
      return is_available_fn_.value()();
    }
    // Return a benign value otherwise.
    return true;
  }

  void install_init(InitFn fn) {
    init_fn_ = fn;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    if (init_fn_) {
      return init_fn_.value()(
          processed, compile_specs, context.get_runtime_allocator());
    }
    // Return a benign value otherwise.
    return nullptr;
  }

  void install_execute(ExecuteFn fn) {
    execute_fn_ = fn;
  }

  Error execute(
      __ET_UNUSED BackendExecutionContext& context,
      DelegateHandle* handle,
      EValue** args) const override {
    if (execute_fn_) {
      return execute_fn_.value()(handle, args);
    }
    // Return a benign value otherwise.
    return Error::Ok;
  }

  void install_destroy(DestroyFn fn) {
    destroy_fn_ = fn;
  }

  void destroy(DelegateHandle* handle) const override {
    if (destroy_fn_) {
      destroy_fn_.value()(handle);
    }
  }

  /**
   * Resets to the original constructed state.
   */
  void reset() {
    is_available_fn_.reset();
    init_fn_.reset();
    execute_fn_.reset();
    destroy_fn_.reset();
  }

  /**
   * Registers the singleton instance if not already registered.
   *
   * Note that this can be used to install the stub as the implementation for
   * any export-time backend by passing in the right name, as long as no other
   * backend with that name has been registered yet.
   */
  static Error register_singleton(const char* name = kName) {
    if (!registered_) {
      registered_ = true;
      return executorch::runtime::register_backend({name, &singleton_});
    }
    return Error::Ok;
  }

  /**
   * Returns the instance that was added to the backend registry.
   */
  static StubBackend& singleton() {
    return singleton_;
  }

 private:
  static bool registered_;
  static StubBackend singleton_;

  std::optional<IsAvailableFn> is_available_fn_;
  std::optional<InitFn> init_fn_;
  std::optional<ExecuteFn> execute_fn_;
  std::optional<DestroyFn> destroy_fn_;
};

bool StubBackend::registered_ = false;
StubBackend StubBackend::singleton_;

/**
 * A DataLoader that wraps a real DataLoader and records the operations
 * performed on it and the FreeableBuffers it loads.
 */
class DataLoaderSpy : public DataLoader {
 public:
  /// A record of an operation performed on this DataLoader.
  struct Operation {
    enum { Load, Free } op;
    size_t offset; // Set for Load; zero for Free.
    void* data; // Set for Free; nullptr for Load.
    size_t size; // Set for Load and Free.
    std::unique_ptr<const DataLoader::SegmentInfo>
        segment_info; // Set for Load; nullptr for Free.
  };

  explicit DataLoaderSpy(DataLoader* delegate) : delegate_(delegate) {}

  Result<FreeableBuffer>
  load(size_t offset, size_t size, const SegmentInfo& segment_info) override {
    Result<FreeableBuffer> buf = delegate_->load(offset, size, segment_info);
    if (!buf.ok()) {
      return buf.error();
    }

    auto segment_info_cpy =
        std::make_unique<const DataLoader::SegmentInfo>(segment_info);
    operations_.push_back(
        {Operation::Load,
         offset,
         /*data=*/nullptr,
         size,
         /*segment_info=*/std::move(segment_info_cpy)});
    auto* context = new SpyContext(&operations_, std::move(buf.get()));
    // Use context->buffer since buf has been moved.
    return FreeableBuffer(
        context->buffer.data(), context->buffer.size(), FreeBuffer, context);
  }

  Result<size_t> size() const override {
    return delegate_->size();
  }

  /**
   * Returns records of the operations performed on this DataLoader and the
   * FreeableBuffers it returned, in order they were performed.
   */
  const std::vector<Operation>& operations() const {
    return operations_;
  }

  /**
   * Returns true if the DataLoader::load() method was called with the correct
   * segment info.
   */
  bool UsedLoad(
      DataLoader::SegmentInfo::Type segment_type,
      const char* descriptor = nullptr) const {
    for (const auto& op : operations_) {
      if (op.op != Operation::Load) {
        continue;
      }
      // We have a load op.
      if (op.segment_info->segment_type == segment_type) {
        if (segment_type != DataLoader::SegmentInfo::Type::Backend) {
          // For non-backend segments, the descriptor is irrelevant / a nullptr.
          return true;
        } else {
          if (strcmp(op.segment_info->descriptor, descriptor) == 0) {
            return true;
          }
        }
      }
    }
    return false;
  }

  /**
   * Returns true if the operations list shows that the provided data pointer
   * was freed.
   */
  bool WasFreed(const void* data) const {
    for (const auto& op : operations_) {
      if (op.op == Operation::Free && op.data == data) {
        return true;
      }
    }
    return false;
  }

 private:
  struct SpyContext {
    SpyContext(std::vector<Operation>* operations, FreeableBuffer&& buffer)
        : operations(operations), buffer(std::move(buffer)) {}
    std::vector<Operation>* operations;
    FreeableBuffer buffer;
  };

  static void FreeBuffer(void* context, void* data, size_t size) {
    auto* sc = reinterpret_cast<SpyContext*>(context);
    sc->operations->push_back(
        {Operation::Free, /*offset=*/0, data, size, /*segment_info=*/nullptr});
    delete sc;
  }

  /// The real loader to delegate to.
  DataLoader* delegate_;

  std::vector<Operation> operations_;
};

constexpr size_t kDefaultNonConstMemBytes = 32 * 1024;
constexpr size_t kDefaultRuntimeMemBytes = 32 * 1024;

class BackendIntegrationTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Make sure that the backend has been registered. Safe to call multiple
    // times. Doing this at runtime ensures that it's only registered if these
    // tests are run.
    ASSERT_EQ(StubBackend::register_singleton(), Error::Ok);

    // Paths to the test program files.
    program_path_ = std::getenv("ET_MODULE_ADD_MUL_PATH");
    ASSERT_FALSE(program_path_.empty());
    program_nosegments_path_ = std::getenv("ET_MODULE_ADD_MUL_NOSEGMENTS_PATH");
    ASSERT_FALSE(program_nosegments_path_.empty());
  }

  void TearDown() override {
    // Clean up any modifications to the singleton.
    StubBackend::singleton().reset();
  }

  /**
   * Returns true if program_path() returns a file with extracted segments.
   */
  bool using_segments() const {
    return GetParam();
  }

  /**
   * Returns tha path to the program to load. May or may not have extracted
   * segments, depending on the return value of using_segments().
   */
  const char* program_path() const {
    if (using_segments()) {
      return program_path_.c_str();
    } else {
      return program_nosegments_path_.c_str();
    }
  }

 private:
  std::string program_path_;
  std::string program_nosegments_path_;
};

TEST_P(BackendIntegrationTest, BackendIsPresent) {
  PyTorchBackendInterface* backend =
      executorch::runtime::get_backend_class(StubBackend::kName);
  ASSERT_EQ(backend, &StubBackend::singleton());
}

// Demonstrate that installed StubBackend initializes successfully by default.
TEST_P(BackendIntegrationTest, BasicInitSucceeds) {
  Result<FileDataLoader> loader = FileDataLoader::from(program_path());
  ASSERT_EQ(loader.error(), Error::Ok);

  Result<Program> program = Program::load(&loader.get());
  ASSERT_EQ(program.error(), Error::Ok);

  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method_res = program->load_method("forward", &mmm.get());
  EXPECT_EQ(method_res.error(), Error::Ok);
}

TEST_P(BackendIntegrationTest, FreeingProcessedBufferSucceeds) {
  // Install an init() implementation that frees its processed buffer, and lets
  // us know that it was actually called by setting init_called.
  bool init_called = false;
  const void* processed_data = nullptr;
  StubBackend::singleton().install_init(
      [&](FreeableBuffer* processed,
          __ET_UNUSED ArrayRef<CompileSpec> compile_specs,
          __ET_UNUSED MemoryAllocator* runtime_allocator)
          -> Result<DelegateHandle*> {
        init_called = true;
        processed_data = processed->data();
        processed->Free();
        return nullptr;
      });

  // Wrap the real loader in a spy so we can see which operations were
  // performed.
  Result<FileDataLoader> loader = FileDataLoader::from(program_path());
  ASSERT_EQ(loader.error(), Error::Ok);
  DataLoaderSpy spy_loader(&loader.get());

  // Load the program.
  Result<Program> program = Program::load(&spy_loader);
  ASSERT_EQ(program.error(), Error::Ok);
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method_res = program->load_method("forward", &mmm.get());
  EXPECT_EQ(method_res.error(), Error::Ok);

  // Demonstrate that our installed init was called.
  EXPECT_TRUE(init_called);

  // See if the processed data was freed.
  bool processed_was_freed = spy_loader.WasFreed(processed_data);
  if (using_segments()) {
    // Used the loader to create the FreeableBuffer that was passed to the
    // backend, so we can see its Free() call.
    EXPECT_TRUE(processed_was_freed);
  } else {
    // Didn't use the loader to create the FreeableBuffer that was passed to the
    // backend, so we can't see its Free() call.
    EXPECT_FALSE(processed_was_freed);
  }
}

TEST_P(BackendIntegrationTest, EndToEndTestWithProcessedAsHandle) {
  // Install an init() implementation that does not free its processed buffer,
  // and returns the FreeableBuffer as the delegate handle.
  FreeableBuffer* init_processed = nullptr;
  StubBackend::singleton().install_init(
      [&](FreeableBuffer* processed,
          __ET_UNUSED ArrayRef<CompileSpec> compile_specs,
          __ET_UNUSED MemoryAllocator* runtime_allocator)
          -> Result<DelegateHandle*> {
        init_processed = processed;
        return processed;
      });

  // Install an execute() that expects the handle to be the processed
  // FreeableBuffer.
  DelegateHandle* execute_handle = nullptr;
  StubBackend::singleton().install_execute(
      [&](DelegateHandle* handle, __ET_UNUSED EValue** args) -> Error {
        execute_handle = handle;
        auto* processed = reinterpret_cast<FreeableBuffer*>(handle);

        // Read the data, which will tend to cause an ASAN error if it's not
        // valid.
        auto copy = std::make_unique<char[]>(processed->size());
        std::memcpy(copy.get(), processed->data(), processed->size());

        return Error::Ok;
      });

  // Install a destroy() that expects the handle to be the processed
  // FreeableBuffer.
  DelegateHandle* destroy_handle = nullptr;
  StubBackend::singleton().install_destroy(
      [&](DelegateHandle* handle) -> void { destroy_handle = handle; });

  // Wrap the real loader in a spy so we can see which operations were
  // performed.
  Result<FileDataLoader> loader = FileDataLoader::from(program_path());
  ASSERT_EQ(loader.error(), Error::Ok);
  DataLoaderSpy spy_loader(&loader.get());

  // Load the program.
  Result<Program> program = Program::load(&spy_loader);
  ASSERT_EQ(program.error(), Error::Ok);

  // Hold onto the address of the processed buffer so we can compare against
  // it after the FreeableBuffer has been destroyed.
  const void* processed_data;

  // Add a scope so we can watch executor be destroyed.
  {
    ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
    Result<Method> method_res = program->load_method("forward", &mmm.get());
    EXPECT_TRUE(method_res.ok());

    // Demonstrate that our installed init was called.
    EXPECT_NE(init_processed, nullptr);
    // Not freed yet.
    EXPECT_GT(init_processed->size(), 0);
    EXPECT_NE(init_processed->data(), nullptr);
    processed_data = init_processed->data();

    // The processed data should not have been freed during init.
    EXPECT_FALSE(spy_loader.WasFreed(init_processed->data()));
    auto method(std::move(method_res.get()));
    // Execute the model.
    exec_aten::ArrayRef<void*> inputs =
        torch::executor::util::PrepareInputTensors(method);
    auto err = method.execute();
    torch::executor::util::FreeInputs(inputs);
    EXPECT_EQ(err, Error::Ok);

    // Check that the processed buffer was passed to execute() as the handle.
    EXPECT_EQ(init_processed, execute_handle);

    // The processed data should not have been freed during execution.
    EXPECT_FALSE(spy_loader.WasFreed(init_processed->data()));
  }

  // `executor` has now been destroyed, which should have freed the processed
  // data.
  bool processed_was_freed = spy_loader.WasFreed(processed_data);
  if (using_segments()) {
    // Used the loader to create the FreeableBuffer that was passed to the
    // backend, so we can see its Free() call.
    EXPECT_TRUE(processed_was_freed);
  } else {
    // Didn't use the loader to create the FreeableBuffer that was passed to the
    // backend, so we can't see its Free() call.
    EXPECT_FALSE(processed_was_freed);
  }

  // And it should have destroyed the backend handle.
  EXPECT_EQ(execute_handle, destroy_handle);
}

/**
 * Tests that the DataLoader's load is receiving the correct segment info for
 * different types of segments.
 */
TEST_P(BackendIntegrationTest, SegmentInfoIsPassedIntoDataLoader) {
  const void* processed_data = nullptr;
  StubBackend::singleton().install_init(
      [&](FreeableBuffer* processed,
          __ET_UNUSED ArrayRef<CompileSpec> compile_specs,
          __ET_UNUSED MemoryAllocator* runtime_allocator)
          -> Result<DelegateHandle*> {
        processed_data = processed->data();
        processed->Free();
        return nullptr;
      });

  // Wrap the real loader in a spy so we can see which operations were
  // performed.
  Result<FileDataLoader> loader = FileDataLoader::from(program_path());
  ASSERT_EQ(loader.error(), Error::Ok);
  DataLoaderSpy spy_loader(&loader.get());

  // Load the program.
  Result<Program> program = Program::load(&spy_loader);
  ASSERT_EQ(program.error(), Error::Ok);
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);

  // Expect that load was called correctly on program segments.
  bool program_load_was_called =
      spy_loader.UsedLoad(DataLoader::SegmentInfo::Type::Program, nullptr);

  // Load a method.
  Result<Method> method_res = program->load_method("forward", &mmm.get());
  EXPECT_EQ(method_res.error(), Error::Ok);

  // Expect that load was called correctly on a backend segment.
  bool backend_load_was_called = spy_loader.UsedLoad(
      DataLoader::SegmentInfo::Type::Backend,
      "StubBackend"); // This backend id is taken from the StubBackend defined
                      // in export_delegated_program.py.

  EXPECT_TRUE(program_load_was_called);
  EXPECT_EQ(backend_load_was_called, using_segments());
}

// TODO: Add more tests for the runtime-to-backend interface. E.g.:
// - Errors during init() or execute() result in runtime init/execution failures
// - Correct values are passed to init()/execute()
// - Demonstrate use of the runtime allocator
// - ...

// Run all BackendIntegrationTests multiple times, varying the return value of
// `GetParam()` based on the `testing::Values` list. The tests will interpret
// the boolean as "using segments".
INSTANTIATE_TEST_SUITE_P(
    VariedSegments,
    BackendIntegrationTest,
    testing::Values(false, true));

class DelegateDataAlignmentTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();

    // Make sure that the backend has been registered. Safe to call multiple
    // times. Doing this at runtime ensures that it's only registered if these
    // tests are run.
    ASSERT_EQ(StubBackend::register_singleton(), Error::Ok);

    // Paths to the test program files.
    default_alignment_program_path_ =
        std::getenv("ET_MODULE_ADD_MUL_NOSEGMENTS_PATH");
    ASSERT_FALSE(default_alignment_program_path_.empty());
    override_alignment_program_path_ =
        std::getenv("ET_MODULE_ADD_MUL_NOSEGMENTS_DA1024_PATH");
    ASSERT_FALSE(override_alignment_program_path_.empty());
  }

  void TearDown() override {
    // Clean up any modifications to the singleton.
    StubBackend::singleton().reset();
  }

  /**
   * Returns the expected minimum alignment of inline tensor data, given
   * the testing parameter.
   */
  size_t expected_alignment() const {
    if (GetParam()) {
      // The delegate data inline alignment used by the -da1024 file.
      return 1024;
    } else {
      // A small alignment that's compatible with any realistic alignment.
      return 4;
    }
  }

  /**
   * Returns tha path to the program to load. May or may not have an alignment
   * override, depending on the return value of expected_alignment().
   */
  const char* program_path() const {
    if (GetParam()) {
      return override_alignment_program_path_.c_str();
    } else {
      return default_alignment_program_path_.c_str();
    }
  }

 private:
  std::string default_alignment_program_path_;
  std::string override_alignment_program_path_;
};

TEST_P(DelegateDataAlignmentTest, ExpectedDataAlignment) {
  // Install an init() implementation that records the pointer to the delegate
  // data blob so we can check its alignment.
  const void* processed_data = nullptr;
  StubBackend::singleton().install_init(
      [&](FreeableBuffer* processed,
          __ET_UNUSED ArrayRef<CompileSpec> compile_specs,
          __ET_UNUSED MemoryAllocator* runtime_allocator)
          -> Result<DelegateHandle*> {
        processed_data = processed->data();
        return nullptr;
      });

  // Create a loader that can satisfy the alignment required by this program.
  Result<FileDataLoader> loader =
      FileDataLoader::from(program_path(), /*alignment=*/expected_alignment());
  ASSERT_EQ(loader.error(), Error::Ok);

  // Wrap the real loader in a spy so we can see which operations were
  // performed.
  DataLoaderSpy spy_loader(&loader.get());

  // Load the program.
  Result<Program> program = Program::load(&spy_loader);
  ASSERT_EQ(program.error(), Error::Ok);
  ManagedMemoryManager mmm(kDefaultNonConstMemBytes, kDefaultRuntimeMemBytes);
  Result<Method> method = program->load_method("forward", &mmm.get());
  EXPECT_TRUE(method.ok());

  // Demonstrate that our installed init was called.
  EXPECT_NE(processed_data, nullptr);

  // Check that it had the required alignment. The alignment of 1024 is larger
  // than the test file with default alignment, so the default alignment cannot
  // accidentally satisfy it.
  EXPECT_ALIGNED(processed_data, expected_alignment());
}

// Run all DelegateDataAlignmentTests multiple times, varying the return value
// of `GetParam()` based on the `testing::Values` list. The tests will interpret
// the boolean as "was inline delegate data alignment overridden to 1024".
INSTANTIATE_TEST_SUITE_P(
    VariedAlignment,
    DelegateDataAlignmentTest,
    testing::Values(false, true));
