/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * Creates multiple Executor instances at the same time, demonstrating that the
 * same process can handle multiple runtimes at once.
 *
 * Usage:
 *   multi_runner --models=<model.pte>[,<m2.pte>[,...]] [--num_instances=<num>]
 */

#include <gflags/gflags.h>

#include <sys/stat.h>

#include <cassert>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <memory>
#include <sstream>
#include <thread>
#include <tuple>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/test/managed_memory_manager.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/util/read_file.h>
#include <executorch/util/util.h>

DEFINE_string(
    models,
    "",
    "Comma-separated list of paths to serialized ExecuTorch model files");
DEFINE_int32(
    num_instances,
    10,
    "Number of Executor instances to create in parallel, for each model");

static bool validate_path_list(
    const char* flagname,
    const std::string& path_list);
DEFINE_validator(models, &validate_path_list);

static bool validate_positive_int32(const char* flagname, int32_t val);
DEFINE_validator(num_instances, &validate_positive_int32);

namespace {
using torch::executor::DataLoader;
using torch::executor::Error;
using torch::executor::FreeableBuffer;
using torch::executor::MemoryAllocator;
using torch::executor::MemoryManager;
using torch::executor::Method;
using torch::executor::Program;
using torch::executor::Result;
using torch::executor::testing::ManagedMemoryManager;
using torch::executor::util::BufferDataLoader;

/**
 * A model that has been loaded and has had its execution plan and inputs
 * prepared. Can be run once.
 *
 * Creates and owns the underyling state, making things easier to manage.
 */
class PreparedModel final {
 public:
  PreparedModel(
      const std::string& name,
      const void* model_data,
      size_t model_data_size,
      size_t non_const_mem_bytes,
      size_t runtime_mem_bytes)
      : name_(name),
        loader_(model_data, model_data_size),
        program_(load_program_or_die(loader_)),
        memory_manager_(non_const_mem_bytes, runtime_mem_bytes),
        method_(load_method_or_die(program_, &memory_manager_.get())),
        has_run_(false) {
    inputs_ = torch::executor::util::PrepareInputTensors(method_);
  }

  void run() {
    ET_CHECK_MSG(!has_run_, "A PreparedModel may only be run once");
    has_run_ = true;

    Error status = method_.execute();
    ET_CHECK_MSG(
        status == Error::Ok,
        "plan.execute() failed with status 0x%" PRIx32,
        status);

    // TODO(T131578656): Do something with the outputs.
  }

  const std::string& name() const {
    return name_;
  }

  ~PreparedModel() {
    torch::executor::util::FreeInputs(inputs_);
  }

 private:
  static Program load_program_or_die(DataLoader& loader) {
    Result<Program> program = Program::load(&loader);
    ET_CHECK(program.ok());
    return std::move(program.get());
  }

  static Method load_method_or_die(
      const Program& program,
      MemoryManager* memory_manager) {
    Result<Method> method = program.load_method("forward", memory_manager);
    ET_CHECK(method.ok());
    return std::move(method.get());
  }

  const std::string name_;
  BufferDataLoader loader_; // Needs to outlive program_
  Program program_; // Needs to outlive executor_
  ManagedMemoryManager memory_manager_; // Needs to outlive executor_
  Method method_;
  exec_aten::ArrayRef<void*> inputs_;

  bool has_run_;
};

/**
 * Creates PreparedModels based on the provided serialized data and memory
 * parameters.
 */
class ModelFactory {
 public:
  ModelFactory(
      const std::string& name, // For debugging
      std::shared_ptr<const char> model_data,
      size_t model_data_size,
      size_t non_const_mem_bytes = 40 * 1024U * 1024U, // 40 MB
      size_t runtime_mem_bytes = 2 * 1024U * 1024U) // 2 MB
      : name_(name),
        model_data_(model_data),
        model_data_size_(model_data_size),
        non_const_mem_bytes_(non_const_mem_bytes),
        runtime_mem_bytes_(runtime_mem_bytes) {}

  std::unique_ptr<PreparedModel> prepare(
      std::string_view name_affix = "") const {
    return std::make_unique<PreparedModel>(
        name_affix.empty() ? name_ : std::string(name_affix) + ":" + name_,
        model_data_.get(),
        model_data_size_,
        non_const_mem_bytes_,
        runtime_mem_bytes_);
  }

  const std::string& name() const {
    return name_;
  }

 private:
  const std::string name_;
  std::shared_ptr<const char> model_data_;

  const size_t model_data_size_;
  const size_t non_const_mem_bytes_;
  const size_t runtime_mem_bytes_;
};

/// Synchronizes a set of model threads as they walk through prepare/run states.
class Synchronizer {
 public:
  explicit Synchronizer(size_t total_threads)
      : total_threads_(total_threads), state_(State::INIT_THREAD) {}

  /// The states for threads to move through. Must advance in order.
  enum class State {
    /// Initial state.
    INIT_THREAD,

    /// Thread is ready to prepare its model instance.
    PREPARE_MODEL,

    /// Thread is ready to run its model instance.
    RUN_MODEL,
  };

  /// Wait until all threads have requested to advance to this state, then
  /// advance all of them.
  void advance_to(State new_state) {
    std::unique_lock<std::mutex> lock(lock_);

    // Enforce valid state machine transitions.
    assert(
        (new_state == State::PREPARE_MODEL && state_ == State::INIT_THREAD) ||
        (new_state == State::RUN_MODEL && state_ == State::PREPARE_MODEL));

    // Indicate that this thread is ready to move to the new state.
    num_ready_++;
    if (num_ready_ == total_threads_) {
      // We were the last thread to become ready. Tell all threads to
      // move to the next state.
      state_ = new_state;
      num_ready_ = 0;
      cv_.notify_all();
    } else {
      // Wait until all other threads are ready.
      cv_.wait(lock, [=] { return this->state_ == new_state; });
    }
  }

 private:
  /// The total number of threads to wait for.
  const size_t total_threads_;

  /// Locks all mutable fields in this class.
  std::mutex lock_;

  /// The number of threads that are ready to move to the next state.
  size_t num_ready_ = 0;

  /// The state that all threads should be in.
  State state_;

  /// Signals threads to check for state updates.
  std::condition_variable cv_;
};

/**
 * Waits for all threads to begin running; prepares a model and waits for all
 * threads to finish preparation; runs the model and exits.
 */
void model_thread(ModelFactory& factory, Synchronizer& sync, size_t thread_id) {
  ET_LOG(
      Info,
      "[%zu] Thread has started for %s.",
      thread_id,
      factory.name().c_str());

  sync.advance_to(Synchronizer::State::PREPARE_MODEL);

  // Create and prepare our model instance.
  ET_LOG(Info, "[%zu] Preparing %s...", thread_id, factory.name().c_str());
  std::unique_ptr<PreparedModel> model =
      factory.prepare(/*name_affix=*/std::to_string(thread_id));
  ET_LOG(Info, "[%zu] Prepared %s.", thread_id, model->name().c_str());

  sync.advance_to(Synchronizer::State::RUN_MODEL);

  // Run our model.
  ET_LOG(Info, "[%zu] Running %s...", thread_id, model->name().c_str());
  model->run();
  ET_LOG(
      Info, "[%zu] Finished running %s...", thread_id, model->name().c_str());

  // TODO(T131578656): Check the model output.
}

/**
 * Splits the provided string on `,` and returns a vector of the non-empty
 * elements. Does not string whitespace.
 */
std::vector<std::string> split_string_list(const std::string& list) {
  std::vector<std::string> items;
  std::stringstream sstream(list);
  while (sstream.good()) {
    std::string item;
    getline(sstream, item, ',');
    if (!item.empty()) {
      items.push_back(item);
    }
  }
  return items;
}

} // namespace

int main(int argc, char** argv) {
  torch::executor::runtime_init();

  // Parse and extract flags.
  gflags::SetUsageMessage(
      "Creates multiple Executor instances at the same time, demonstrating "
      "that the same process can handle multiple runtimes at once.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<std::string> model_paths = split_string_list(FLAGS_models);
  size_t num_instances = FLAGS_num_instances;

  // Create a factory for each model provided on the commandline.
  std::vector<std::unique_ptr<ModelFactory>> factories;
  for (const auto& model_path : model_paths) {
    std::shared_ptr<char> file_data;
    size_t file_size;
    Error err = torch::executor::util::read_file_content(
        model_path.c_str(), &file_data, &file_size);
    ET_CHECK(err == Error::Ok);
    factories.push_back(std::make_unique<ModelFactory>(
        /*name=*/model_path, file_data, file_size));
  }

  // Spawn threads to prepare and run separate instances of the models in
  // parallel.
  const size_t num_threads = factories.size() * num_instances;
  Synchronizer state(num_threads);
  std::vector<std::thread> threads;
  size_t thread_id = 0; // Unique ID for every thread.
  ET_LOG(Info, "Creating %zu threads...", num_threads);
  for (const auto& factory : factories) {
    for (size_t i = 0; i < num_instances; ++i) {
      threads.push_back(std::thread(
          model_thread, std::ref(*factory), std::ref(state), thread_id++));
    }
  }

  // Wait for all threads to finish.
  ET_LOG(Info, "Waiting for %zu threads to exit...", threads.size());
  for (auto& thread : threads) {
    thread.join();
  }
  ET_LOG(Info, "All %zu threads exited.", threads.size());
}

//
// Flag validation
//

/// Returns true if the specified path exists in the filesystem.
static bool path_exists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

/// Returns true if `path_list` contains a comma-separated list of at least one
/// path that exists in the filesystem.
static bool validate_path_list(
    const char* flagname,
    const std::string& path_list) {
  const std::vector<std::string> paths = split_string_list(path_list);
  if (paths.empty()) {
    fprintf(
        stderr, "Must specify at least one valid path with --%s\n", flagname);
    return false;
  }
  for (const auto& path : split_string_list(path_list)) {
    if (!path_exists(path)) {
      fprintf(
          stderr,
          "Path '%s' does not exist in --%s='%s'\n",
          path.c_str(),
          flagname,
          path_list.c_str());
      return false;
    }
  }
  return true;
}

/// Returns true if `val` is positive.
static bool validate_positive_int32(const char* flagname, int32_t val) {
  if (val <= 0) {
    fprintf(
        stderr, "Value must be positive for --%s=%" PRId32 "\n", flagname, val);
    return false;
  }
  return true;
}
