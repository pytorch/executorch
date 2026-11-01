/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/platform/runtime.h>

#include <mutex>
#include <numeric>
#include <random>

#include <executorch/extension/threadpool/threadpool_guard.h>

#include <gtest/gtest.h>

#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#endif

using namespace ::testing;

namespace {

size_t div_round_up(const size_t divident, const size_t divisor) {
  return (divident + divisor - 1) / divisor;
}

void resize_and_fill_vector(std::vector<int32_t>& a, const size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(1, size * 2);
  a.resize(size);
  auto generator = [&distrib, &gen]() { return distrib(gen); };
  std::generate(a.begin(), a.end(), generator);
}

void generate_add_test_inputs(
    std::vector<int32_t>& a,
    std::vector<int32_t>& b,
    std::vector<int32_t>& c_ref,
    std::vector<int32_t>& c,
    size_t vector_size) {
  resize_and_fill_vector(a, vector_size);
  resize_and_fill_vector(b, vector_size);
  resize_and_fill_vector(c, vector_size);
  resize_and_fill_vector(c_ref, vector_size);
  for (size_t i = 0, size = a.size(); i < size; ++i) {
    c_ref[i] = a[i] + b[i];
  }
}

void generate_reduce_test_inputs(
    std::vector<int32_t>& a,
    int32_t& c_ref,
    size_t vector_size) {
  resize_and_fill_vector(a, vector_size);
  c_ref = 0;
  for (size_t i = 0, size = a.size(); i < size; ++i) {
    c_ref += a[i];
  }
}

void run_lambda_with_size(
    std::function<void(size_t)> f,
    size_t range,
    size_t grain_size) {
  size_t num_grains = div_round_up(range, grain_size);

  auto threadpool = ::executorch::extension::threadpool::get_threadpool();
  threadpool->run(f, range);
}

// The pool has no more threads than the CPUs this process may run on.
bool may_use_more_than_one_cpu() {
#if defined(__linux__) && !defined(__ANDROID__)
  // Grows the mask as get_num_allowed_cpus() does, for kernels with more
  // possible CPUs than a cpu_set_t holds.
  for (int num_cpus = CPU_SETSIZE; num_cpus <= (1 << 20); num_cpus *= 2) {
    std::unique_ptr<cpu_set_t, void (*)(cpu_set_t*)> cpus(
        CPU_ALLOC(num_cpus), [](cpu_set_t* set) { CPU_FREE(set); });
    if (!cpus) {
      break;
    }
    const size_t size = CPU_ALLOC_SIZE(num_cpus);
    if (sched_getaffinity(0, size, cpus.get()) == 0) {
      return CPU_COUNT_S(size, cpus.get()) > 1;
    }
    if (errno != EINVAL) {
      break;
    }
  }
#endif
  return true;
}

#if defined(__linux__) && !defined(__ANDROID__)
// Limits this thread to the CPU it runs on, then prints the default pool size
// and the performance core count for the test to match. Failing to set the
// limit aborts instead.
[[noreturn]] void print_counts_on_one_cpu() {
  const int cpu = sched_getcpu();
  if (cpu < 0) {
    std::abort();
  }
  std::unique_ptr<cpu_set_t, void (*)(cpu_set_t*)> cpus(
      CPU_ALLOC(cpu + 1), [](cpu_set_t* set) { CPU_FREE(set); });
  if (!cpus) {
    std::abort();
  }
  const size_t size = CPU_ALLOC_SIZE(cpu + 1);
  CPU_ZERO_S(size, cpus.get());
  CPU_SET_S(cpu, size, cpus.get());
  if (sched_setaffinity(0, size, cpus.get()) != 0) {
    std::abort();
  }
  std::fprintf(
      stderr,
      "pool %zu perf %u end\n",
      ::executorch::extension::threadpool::get_threadpool()->get_thread_count(),
      static_cast<unsigned>(
          ::executorch::extension::cpuinfo::get_num_performant_cores()));
  std::exit(EXIT_SUCCESS);
}
#endif
} // namespace

TEST(ThreadPoolTest, ParallelAdd) {
  executorch::runtime::runtime_init();

  std::vector<int32_t> a, b, c, c_ref;
  size_t vector_size = 100;
  size_t grain_size = 10;

  auto add_lambda = [&](size_t i) {
    size_t start_index = i * grain_size;
    size_t end_index = start_index + grain_size;
    end_index = std::min(end_index, vector_size);
    for (size_t j = start_index; j < end_index; ++j) {
      c[j] = a[j] + b[j];
    }
  };

  auto threadpool = ::executorch::extension::threadpool::get_threadpool();
  if (may_use_more_than_one_cpu()) {
    EXPECT_GT(threadpool->get_thread_count(), 1);
  }

  generate_add_test_inputs(a, b, c_ref, c, vector_size);
  run_lambda_with_size(add_lambda, vector_size, grain_size);
  EXPECT_EQ(c, c_ref);

  // Try smaller grain size
  grain_size = 5;
  generate_add_test_inputs(a, b, c_ref, c, vector_size);
  run_lambda_with_size(add_lambda, vector_size, grain_size);
  EXPECT_EQ(c, c_ref);

  vector_size = 7;
  generate_add_test_inputs(a, b, c_ref, c, vector_size);
  run_lambda_with_size(add_lambda, vector_size, grain_size);
  EXPECT_EQ(c, c_ref);

  vector_size = 7;
  grain_size = 5;
  generate_add_test_inputs(a, b, c_ref, c, vector_size);
  run_lambda_with_size(add_lambda, vector_size, grain_size);
  EXPECT_EQ(c, c_ref);
}

// Test parallel reduction where we acquire lock within lambda
TEST(ThreadPoolTest, ParallelReduce) {
  executorch::runtime::runtime_init();

  std::vector<int32_t> a;
  int32_t c = 0, c_ref = 0;
  size_t vector_size = 100;
  size_t grain_size = 11;
  std::mutex m;

  auto reduce_lambda = [&](size_t i) {
    size_t start_index = i * grain_size;
    size_t end_index = start_index + grain_size;
    end_index = std::min(end_index, vector_size);
    std::lock_guard<std::mutex> lock(m);
    for (size_t j = start_index; j < end_index; ++j) {
      c += a[j];
    }
  };

  auto threadpool = ::executorch::extension::threadpool::get_threadpool();
  if (may_use_more_than_one_cpu()) {
    EXPECT_GT(threadpool->get_thread_count(), 1);
  }

  generate_reduce_test_inputs(a, c_ref, vector_size);
  run_lambda_with_size(reduce_lambda, vector_size, grain_size);
  EXPECT_EQ(c, c_ref);

  vector_size = 7;
  c = c_ref = 0;
  generate_reduce_test_inputs(a, c_ref, vector_size);
  run_lambda_with_size(reduce_lambda, vector_size, grain_size);
  EXPECT_EQ(c, c_ref);
}

// Copied from
// caffe2/aten/src/ATen/test/test_thread_pool_guard.cp
TEST(TestNoThreadPoolGuard, TestThreadPoolGuard) {
  executorch::runtime::runtime_init();

  auto threadpool_ptr = ::executorch::extension::threadpool::get_pthreadpool();

  ASSERT_NE(threadpool_ptr, nullptr);
  {
    ::executorch::extension::threadpool::NoThreadPoolGuard g1;
    auto threadpool_ptr1 =
        ::executorch::extension::threadpool::get_pthreadpool();
    ASSERT_EQ(threadpool_ptr1, nullptr);

    {
      ::executorch::extension::threadpool::NoThreadPoolGuard g2;
      auto threadpool_ptr2 =
          ::executorch::extension::threadpool::get_pthreadpool();
      ASSERT_EQ(threadpool_ptr2, nullptr);
    }

    // Guard should restore prev value (nullptr)
    auto threadpool_ptr3 =
        ::executorch::extension::threadpool::get_pthreadpool();
    ASSERT_EQ(threadpool_ptr3, nullptr);
  }

  // Guard should restore prev value (pthreadpool_)
  auto threadpool_ptr4 = ::executorch::extension::threadpool::get_pthreadpool();
  ASSERT_NE(threadpool_ptr4, nullptr);
  ASSERT_EQ(threadpool_ptr4, threadpool_ptr);
}

TEST(TestNoThreadPoolGuard, TestRunWithGuard) {
  executorch::runtime::runtime_init();

  const std::vector<int64_t> array = {1, 2, 3};

  auto pool = ::executorch::extension::threadpool::get_threadpool();
  int64_t inner = 0;
  {
    // Run on same thread
    ::executorch::extension::threadpool::NoThreadPoolGuard g1;
    auto fn = [&array, &inner](const size_t task_id) {
      inner += array[task_id];
    };
    pool->run(fn, 3);

    // confirm the guard is on
    auto threadpool_ptr =
        ::executorch::extension::threadpool::get_pthreadpool();
    ASSERT_EQ(threadpool_ptr, nullptr);
  }
  ASSERT_EQ(inner, 6);
}

#if defined(__linux__) && !defined(__ANDROID__)
// gtest runs death tests before the other tests, so the forked child creates
// the pool itself, and the CPU limit it sets does not reach the other tests.
TEST(ThreadPoolDeathTest, DefaultSizeHonorsCpuAffinity) {
  // A later run, as with --gtest_repeat, forks after the other tests created
  // the pool, and the child keeps that pool's size.
  static int runs = 0;
  if (runs++ > 0) {
    GTEST_SKIP() << "The pool may already exist in this process";
  }
  executorch::runtime::runtime_init();

  // A known performance core count drops to 1, and an unknown one stays 0.
  const char* expected =
      ::executorch::extension::cpuinfo::get_num_performant_cores() > 0
      ? "pool 1 perf 1 end"
      : "pool 1 perf 0 end";
  EXPECT_EXIT(
      print_counts_on_one_cpu(), ExitedWithCode(EXIT_SUCCESS), expected);
}
#endif
