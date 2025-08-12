/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <array>
#include <mutex>

#include <executorch/runtime/kernel/thread_parallel_interface.h>
#include <executorch/runtime/platform/platform.h>

using namespace ::testing;
using ::executorch::extension::parallel_for;

class ParallelTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    data_.fill(0);
    sum_of_all_elements_ = 0;
  }

  void RunTask(int64_t begin, int64_t end) {
    for (int64_t j = begin; j < end; ++j) {
      // Check that we haven't written to this index before
      EXPECT_EQ(data_[j], 0);
      data_[j] = j;
    }
  }

  void RunExclusiveTask(int64_t begin, int64_t end) {
    for (int64_t j = begin; j < end; ++j) {
      // Check that we haven't written to this index before
      EXPECT_EQ(data_[j], 0);
      std::lock_guard<std::mutex> lock(mutex_);
      data_[j] = j;
      sum_of_all_elements_ += data_[j];
    }
  }

  template <typename Func>
  bool parallel_for(
      const int64_t begin,
      const int64_t end,
      const int64_t grain_size,
      const Func& func) {
    if (GetParam()) {
      return executorch::extension::parallel_for(begin, end, grain_size, func);
    }
    return executorch::extension::internal::parallel_for_no_threadpool(
        begin, end, grain_size, func);
  }

  std::array<int, 10> data_;
  std::mutex mutex_;
  int sum_of_all_elements_;
};

TEST_P(ParallelTest, TestAllInvoked) {
  EXPECT_TRUE(parallel_for(0, 10, 1, [this](int64_t begin, int64_t end) {
    this->RunTask(begin, end);
  }));

  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(data_[i], i);
  }
}

TEST_P(ParallelTest, TestAllInvokedWithMutex) {
  EXPECT_TRUE(parallel_for(0, 10, 1, [this](int64_t begin, int64_t end) {
    this->RunExclusiveTask(begin, end);
  }));

  int expected_sum = 0;
  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(data_[i], i);
    expected_sum += i;
  }
  EXPECT_EQ(sum_of_all_elements_, expected_sum);
}

TEST_P(ParallelTest, TestInvalidRange) {
  et_pal_init();
  EXPECT_FALSE(parallel_for(10, 0, 1, [this](int64_t begin, int64_t end) {
    this->RunExclusiveTask(begin, end);
  }));

  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(data_[i], 0);
  }
  EXPECT_EQ(sum_of_all_elements_, 0);
}

TEST_P(ParallelTest, TestInvalidRange2) {
  et_pal_init();
  EXPECT_FALSE(parallel_for(6, 5, 1, [this](int64_t begin, int64_t end) {
    this->RunExclusiveTask(begin, end);
  }));

  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(data_[i], 0);
  }
  EXPECT_EQ(sum_of_all_elements_, 0);
}

TEST_P(ParallelTest, TestInvokePartialFromBeginning) {
  EXPECT_TRUE(parallel_for(0, 5, 1, [this](int64_t begin, int64_t end) {
    this->RunTask(begin, end);
  }));

  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_EQ(data_[i], i);
  }
  for (int64_t i = 5; i < 10; ++i) {
    EXPECT_EQ(data_[i], 0);
  }
}

TEST_P(ParallelTest, TestInvokePartialToEnd) {
  EXPECT_TRUE(parallel_for(5, 10, 1, [this](int64_t begin, int64_t end) {
    this->RunTask(begin, end);
  }));

  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_EQ(data_[i], 0);
  }
  for (int64_t i = 5; i < 10; ++i) {
    EXPECT_EQ(data_[i], i);
  }
}

TEST_P(ParallelTest, TestInvokePartialMiddle) {
  EXPECT_TRUE(parallel_for(2, 8, 1, [this](int64_t begin, int64_t end) {
    this->RunTask(begin, end);
  }));

  for (int64_t i = 0; i < 2; ++i) {
    EXPECT_EQ(data_[i], 0);
  }
  for (int64_t i = 2; i < 8; ++i) {
    EXPECT_EQ(data_[i], i);
  }
  for (int64_t i = 8; i < 10; ++i) {
    EXPECT_EQ(data_[i], 0);
  }
}

TEST_P(ParallelTest, TestChunkSize2) {
  EXPECT_TRUE(parallel_for(0, 10, 2, [this](int64_t begin, int64_t end) {
    this->RunTask(begin, end);
  }));

  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(data_[i], i);
  }
}

TEST_P(ParallelTest, TestChunkSize2Middle) {
  EXPECT_TRUE(parallel_for(3, 8, 2, [this](int64_t begin, int64_t end) {
    this->RunTask(begin, end);
  }));

  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(data_[i], 0);
  }
  for (int64_t i = 3; i < 8; ++i) {
    EXPECT_EQ(data_[i], i);
  }
  for (int64_t i = 8; i < 10; ++i) {
    EXPECT_EQ(data_[i], 0);
  }
}

TEST_P(ParallelTest, TestChunkSize3) {
  EXPECT_TRUE(parallel_for(0, 10, 3, [this](int64_t begin, int64_t end) {
    this->RunTask(begin, end);
  }));

  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(data_[i], i);
  }
}

TEST_P(ParallelTest, TestChunkSize6) {
  EXPECT_TRUE(parallel_for(0, 10, 6, [this](int64_t begin, int64_t end) {
    this->RunTask(begin, end);
  }));

  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(data_[i], i);
  }
}

TEST_P(ParallelTest, TestChunkSizeTooLarge) {
  EXPECT_TRUE(parallel_for(0, 10, 11, [this](int64_t begin, int64_t end) {
    this->RunTask(begin, end);
  }));

  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_EQ(data_[i], i);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ParallelTestWithOrWithoutThreadpool,
    ParallelTest,
    ::testing::Values(true, false));
