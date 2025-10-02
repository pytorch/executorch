/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/testing_util/error_matchers.h>

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::runtime::testing {
namespace {

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;
using ::executorch::runtime::testing::ErrorIs;
using ::executorch::runtime::testing::IsOk;
using ::executorch::runtime::testing::IsOkAndHolds;
using ::testing::AnyOf;
using ::testing::DescribeMatcher;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Matcher;
using ::testing::Not;

TEST(ResultMatchersTest, IsOkMatchesOkResult) {
  Result<int> ok_result(42);
  EXPECT_THAT(ok_result, IsOk());
}

TEST(ResultMatchersTest, IsOkDoesNotMatchErrorResult) {
  Result<int> error_result(Error::InvalidArgument);
  EXPECT_THAT(error_result, Not(IsOk()));
}

TEST(ResultMatchersTest, IsOkAndHoldsMatchesOkResultWithMatchingValue) {
  Result<int> ok_result(42);
  EXPECT_THAT(ok_result, IsOkAndHolds(42));
  EXPECT_THAT(ok_result, IsOkAndHolds(Eq(42)));
}

TEST(ResultMatchersTest, IsOkAndHoldsDoesNotMatchErrorResult) {
  Result<int> error_result(Error::InvalidArgument);
  EXPECT_THAT(error_result, Not(IsOkAndHolds(42)));
}

TEST(ResultMatchersTest, ErrorIsMatchesSpecificError) {
  Error error = Error::InvalidArgument;
  Result<int> invalid_arg_result(Error::InvalidArgument);
  Result<int> ok_result(42);

  EXPECT_THAT(error, ErrorIs(Error::InvalidArgument));
  EXPECT_THAT(invalid_arg_result, ErrorIs(Error::InvalidArgument));
  EXPECT_THAT(invalid_arg_result, Not(ErrorIs(Error::NotFound)));
  EXPECT_THAT(ok_result, Not(ErrorIs(Error::InvalidArgument)));
}

TEST(ResultMatchersTest, ErrorIsWorksWithMatchers) {
  Result<int> invalid_arg_result(Error::InvalidArgument);
  Result<int> ok_result(42);

  EXPECT_THAT(invalid_arg_result, ErrorIs(Eq(Error::InvalidArgument)));
  EXPECT_THAT(
      invalid_arg_result,
      ErrorIs(AnyOf(Error::InvalidArgument, Error::NotFound)));
  EXPECT_THAT(
      ok_result, Not(ErrorIs(AnyOf(Error::InvalidArgument, Error::NotFound))));
}

TEST(ResultMatchersTest, ErrorIsWorksWithDifferentResultTypes) {
  Result<std::string> string_error_result(Error::InvalidType);
  Result<double> double_error_result(Error::MemoryAllocationFailed);

  EXPECT_THAT(string_error_result, ErrorIs(Error::InvalidType));
  EXPECT_THAT(double_error_result, ErrorIs(Error::MemoryAllocationFailed));
  EXPECT_THAT(string_error_result, Not(ErrorIs(Error::MemoryAllocationFailed)));
}

TEST(ResultMatchersTest, ErrorIsDoesNotMatchOkResult) {
  Result<int> ok_result(42);

  EXPECT_THAT(ok_result, Not(ErrorIs(Error::InvalidArgument)));
  EXPECT_THAT(ok_result, Not(ErrorIs(Error::NotFound)));
  EXPECT_THAT(ok_result, ErrorIs(Error::Ok));
}

TEST(ResultMatchersTest, AssertOkAndUnwrapWorksWithOkResult) {
  Result<int> ok_result(42);
  int value = ASSERT_OK_AND_UNWRAP(Result<int>(42));
  EXPECT_EQ(42, value);
}

TEST(ResultMatchersTest, AssertOkAndUnwrapWorksWithStringResult) {
  std::string value = ASSERT_OK_AND_UNWRAP(Result<std::string>("hello world"));
  EXPECT_EQ("hello world", value);
}

TEST(ResultMatchersTest, AssertOkAndUnwrapWorksWithMoveOnlyTypes) {
  Result<std::unique_ptr<int>> ok_result(std::make_unique<int>(42));
  std::unique_ptr<int> value = ASSERT_OK_AND_UNWRAP(std::move(ok_result));
  EXPECT_EQ(42, *value);
}

TEST(ResultMatchersTest, MatcherDescriptions) {
  Matcher<Result<int>> is_ok_matcher = IsOk();
  Matcher<Result<int>> is_ok_and_holds_matcher = IsOkAndHolds(42);
  Matcher<Result<int>> error_is_matcher = ErrorIs(Error::InvalidArgument);

  EXPECT_EQ("is OK", DescribeMatcher<Result<int>>(is_ok_matcher));
  EXPECT_EQ("is not OK", DescribeMatcher<Result<int>>(is_ok_matcher, true));
  EXPECT_THAT(
      DescribeMatcher<Result<int>>(is_ok_and_holds_matcher),
      HasSubstr("is OK and has a value that"));
  EXPECT_THAT(
      DescribeMatcher<Result<int>>(is_ok_and_holds_matcher, true),
      HasSubstr("isn't OK or has a value that"));
  EXPECT_THAT(
      DescribeMatcher<Result<int>>(error_is_matcher),
      HasSubstr("has an error that"));
  EXPECT_THAT(
      DescribeMatcher<Result<int>>(error_is_matcher, true),
      HasSubstr("does not have an error that"));
}

} // namespace
} // namespace executorch::runtime::testing
