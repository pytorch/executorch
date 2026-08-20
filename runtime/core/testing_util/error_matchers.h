/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Testing utilities for working with `executorch::runtime::Result<T>` and
 * `executorch::runtime::Error`. Provides matchers similar to `absl::StatusOr`
 * and `absl::Status`.
 *
 * Defines the following utilities:
 *
 *   ===============
 *   `IsOkAndHolds(m)`
 *   ===============
 *
 *   This gMock matcher matches a Result<T> value whose error is Ok
 *   and whose inner value matches matcher m.  Example:
 *
 *   ```
 *   using ::testing::MatchesRegex;
 *   using ::executorch::runtime::testing::IsOkAndHolds;
 *   ...
 *   executorch::runtime::Result<string> maybe_name = ...;
 *   EXPECT_THAT(maybe_name, IsOkAndHolds(MatchesRegex("John .*")));
 *   ```
 *
 *   ===============
 *   `ErrorIs(Error::error_code)`
 *   ===============
 *
 *   This gMock matcher matches a Result<T> value whose error matches
 *   the given error matcher. Example:
 *
 *   ```
 *   using ::executorch::runtime::testing::ErrorIs;
 *   ...
 *   executorch::runtime::Result<string> maybe_name = ...;
 *   EXPECT_THAT(maybe_name, ErrorIs(Error::InvalidArgument));
 *   ```
 *
 *   ===============
 *   `IsOk()`
 *   ===============
 *
 *   Matches an `executorch::runtime::Result<T>` value whose error value
 *   is `executorch::runtime::Error::Ok`.
 *
 *   Example:
 *   ```
 *   using ::executorch::runtime::testing::IsOk;
 *   ...
 *   executorch::runtime::Result<string> maybe_name = ...;
 *   EXPECT_THAT(maybe_name, IsOk());
 *   ```
 */

#pragma once

#include <ostream>
#include <utility>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest-matchers.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

/**
 * Unwrap a Result to obtain its value. If the Result contains an error,
 * fail the test with ASSERT_TRUE.
 *
 * This macro is useful for test code where you want to extract the value
 * from a Result and fail the test if the Result contains an error.
 *
 * Example usage:
 * ```
 *   Result<int> maybe_value = GetSomeValue();
 *   int value = ASSERT_OK_AND_UNWRAP(maybe_value);
 *   // Use value...
 * ```
 *
 * @param[in] result__ Expression yielding the Result to unwrap.
 */
#define ASSERT_OK_AND_UNWRAP(result__) \
  ({                                   \
    auto&& et_result__ = (result__);   \
    ASSERT_TRUE(et_result__.ok());     \
    std::move(*et_result__);           \
  })

namespace executorch {
namespace runtime {
namespace testing {
namespace internal {

// Helper function to get the error from a Result
template <typename T>
inline Error GetError(const Result<T>& result) {
  return result.error();
}

// Helper function to get the error from a raw Error (identity function)
inline Error GetError(const Error& error) {
  return error;
}

////////////////////////////////////////////////////////////
// Implementation of IsOkAndHolds().

// Monomorphic implementation of matcher IsOkAndHolds(m). ResultType is a
// reference to Result<T>.
template <typename ResultType>
class IsOkAndHoldsMatcherImpl : public ::testing::MatcherInterface<ResultType> {
 public:
  typedef
      typename std::remove_reference<ResultType>::type::value_type value_type;

  template <typename InnerMatcher>
  explicit IsOkAndHoldsMatcherImpl(InnerMatcher&& inner_matcher)
      : inner_matcher_(::testing::SafeMatcherCast<const value_type&>(
            std::forward<InnerMatcher>(inner_matcher))) {}

  void DescribeTo(std::ostream* os) const override {
    *os << "is OK and has a value that ";
    inner_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "isn't OK or has a value that ";
    inner_matcher_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      ResultType actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    if (!actual_value.ok()) {
      *result_listener << "which has error "
                       << ::executorch::runtime::to_string(
                              GetError(actual_value));
      return false;
    }

    // Call through to the inner matcher.
    return inner_matcher_.MatchAndExplain(*actual_value, result_listener);
  }

 private:
  const ::testing::Matcher<const value_type&> inner_matcher_;
};

// Implements IsOkAndHolds(m) as a polymorphic matcher.
template <typename InnerMatcher>
class IsOkAndHoldsMatcher {
 public:
  explicit IsOkAndHoldsMatcher(InnerMatcher inner_matcher)
      : inner_matcher_(std::forward<InnerMatcher>(inner_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the
  // given type. ResultType can be either Result<T> or a
  // reference to Result<T>.
  template <typename ResultType>
  operator ::testing::Matcher<ResultType>() const { // NOLINT
    return ::testing::Matcher<ResultType>(
        new IsOkAndHoldsMatcherImpl<const ResultType&>(inner_matcher_));
  }

 private:
  const InnerMatcher inner_matcher_;
};

////////////////////////////////////////////////////////////
// Implementation of IsOk().

// Monomorphic implementation of matcher IsOk() for a given type T.
// T can be Result<U>, Error, or references to either.
template <typename T>
class MonoIsOkMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  void DescribeTo(std::ostream* os) const override {
    *os << "is OK";
  }
  void DescribeNegationTo(std::ostream* os) const override {
    *os << "is not OK";
  }
  bool MatchAndExplain(
      T actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    const Error error = GetError(actual_value);
    if (error != Error::Ok) {
      *result_listener << "which has error "
                       << ::executorch::runtime::to_string(error);
      return false;
    }
    return true;
  }
};

// Implements IsOk() as a polymorphic matcher.
class IsOkMatcher {
 public:
  template <typename T>
  operator ::testing::Matcher<T>() const { // NOLINT
    return ::testing::Matcher<T>(new MonoIsOkMatcherImpl<const T&>());
  }
};

////////////////////////////////////////////////////////////
// Implementation of ErrorIs().

// Monomorphic implementation of matcher ErrorIs() for a given type T.
// T can be Result<U> or a reference to Result<U>.
template <typename T>
class MonoErrorIsMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  explicit MonoErrorIsMatcherImpl(::testing::Matcher<Error> error_matcher)
      : error_matcher_(std::move(error_matcher)) {}

  void DescribeTo(std::ostream* os) const override {
    *os << "has an error that ";
    error_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "does not have an error that ";
    error_matcher_.DescribeNegationTo(os);
  }

  bool MatchAndExplain(
      T actual_value,
      ::testing::MatchResultListener* result_listener) const override {
    Error actual_error = GetError(actual_value);
    *result_listener << "which has error "
                     << ::executorch::runtime::to_string(actual_error);
    return error_matcher_.MatchAndExplain(actual_error, result_listener);
  }

 private:
  const ::testing::Matcher<Error> error_matcher_;
};

// Implements ErrorIs() as a polymorphic matcher.
template <typename ErrorMatcher>
class ErrorIsMatcher {
 public:
  explicit ErrorIsMatcher(ErrorMatcher error_matcher)
      : error_matcher_(std::forward<ErrorMatcher>(error_matcher)) {}

  // Converts this polymorphic matcher to a monomorphic matcher of the
  // given type. T can be Result<U> or a reference to Result<U>.
  template <typename T>
  operator ::testing::Matcher<T>() const { // NOLINT
    return ::testing::Matcher<T>(new MonoErrorIsMatcherImpl<const T&>(
        ::testing::MatcherCast<Error>(error_matcher_)));
  }

 private:
  const ErrorMatcher error_matcher_;
};

} // namespace internal

// Returns a gMock matcher that matches a Result<> whose error is
// OK and whose value matches the inner matcher.
template <typename InnerMatcherT>
internal::IsOkAndHoldsMatcher<typename std::decay<InnerMatcherT>::type>
IsOkAndHolds(InnerMatcherT&& inner_matcher) {
  return internal::IsOkAndHoldsMatcher<
      typename std::decay<InnerMatcherT>::type>(
      std::forward<InnerMatcherT>(inner_matcher));
}

// Returns a gMock matcher that matches a Result<> whose error matches
// the given error matcher.
template <typename ErrorMatcherT>
internal::ErrorIsMatcher<typename std::decay<ErrorMatcherT>::type> ErrorIs(
    ErrorMatcherT&& error_matcher) {
  return internal::ErrorIsMatcher<typename std::decay<ErrorMatcherT>::type>(
      std::forward<ErrorMatcherT>(error_matcher));
}

// Returns a gMock matcher that matches a Result<> which is OK.
inline internal::IsOkMatcher IsOk() {
  return internal::IsOkMatcher();
}

} // namespace testing
} // namespace runtime
} // namespace executorch

namespace executorch {
namespace runtime {

// This needs to be defined in the SAME namespace that defines Error.
// C++'s look-up rules rely on that.
void PrintTo(const Error& error, std::ostream* os);

} // namespace runtime
} // namespace executorch
