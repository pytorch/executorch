/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::runtime::Error;
using executorch::runtime::Result;

static void* test_ptr = static_cast<void*>((uintptr_t*)0xDEADBEEF);

Result<uint64_t> get_abs(int64_t num) {
  if (num >= 0) {
    return static_cast<uint64_t>(num);
  } else {
    return Error::InvalidArgument;
  }
}

Result<std::string> get_op_name(int64_t op) {
  auto unsigned_op = ET_UNWRAP(get_abs(op));

  switch (unsigned_op) {
    case 0:
      return std::string("Zero");
    case 1:
      return std::string("One");
    default:
      return Error::Internal;
  }
}

Result<const void*> get_ptr(int64_t value) {
  switch (value) {
    case 0:
      return nullptr;
    case 1:
      return test_ptr;
    default:
      return Error::InvalidArgument;
  }
}

class Uncopiable {
 public:
  explicit Uncopiable(uint32_t value) : value_(value) {}
  Uncopiable(Uncopiable&& rhs) noexcept = default;
  ~Uncopiable() = default;

  uint32_t getValue() const {
    return value_;
  }

 private:
  // Delete other rule-of-five methods.
  Uncopiable(const Uncopiable& rhs) = delete;
  Uncopiable& operator=(Uncopiable&& rhs) noexcept = delete;
  Uncopiable& operator=(const Uncopiable& rhs) = delete;

  uint32_t value_;
};

Result<Uncopiable> get_no_copy(uint32_t value) {
  return Uncopiable(value);
}

// A non-trivially-movable type.
class Movable {
 public:
  explicit Movable(size_t nbytes) : buffer_(malloc(nbytes)) {}
  Movable(Movable&& rhs) noexcept : buffer_(rhs.buffer_) {
    rhs.buffer_ = nullptr;
  }

  ~Movable() {
    // This will fail with a double-free if it wasn't moved properly.
    free(buffer_);
  }

  const void* buffer() const {
    return buffer_;
  }

 private:
  // Delete other rule-of-five methods.
  Movable(const Movable& rhs) = delete;
  Movable& operator=(Movable&& rhs) noexcept = delete;
  Movable& operator=(const Movable& rhs) = delete;

  mutable void* buffer_;
};

TEST(ErrorHandlingTest, ResultBasic) {
  Result<uint32_t> r(1);
  ASSERT_TRUE(r.ok());
  ASSERT_EQ(r.error(), Error::Ok);
  ASSERT_EQ(r.get(), 1);
  ASSERT_EQ(*r, 1);
}

TEST(ErrorHandlingTest, OkErrorNotPossible) {
  Result<uint32_t> r(Error::Ok);
  ASSERT_FALSE(r.ok());
  ASSERT_NE(r.error(), Error::Ok);
}

TEST(ErrorHandlingTest, ResultWithPrimitive) {
  auto res = get_abs(100);
  ASSERT_TRUE(res.ok());
  ASSERT_EQ(res.error(), Error::Ok);

  uint64_t unsigned_result = res.get();
  ASSERT_EQ(unsigned_result, 100);
  unsigned_result = *res;
  ASSERT_EQ(unsigned_result, 100);

  auto res2 = get_abs(-3);
  ASSERT_FALSE(res2.ok());
  ASSERT_EQ(res2.error(), Error::InvalidArgument);
}

TEST(ErrorHandlingTest, ResultWithCompound) {
  auto res = get_op_name(0);
  ASSERT_TRUE(res.ok());
  ASSERT_EQ(res.error(), Error::Ok);
  ASSERT_EQ(res.get(), "Zero");
  ASSERT_EQ(*res, "Zero");

  auto res2 = get_op_name(1);
  ASSERT_TRUE(res2.ok());
  ASSERT_EQ(res2.error(), Error::Ok);
  ASSERT_EQ(res2.get(), "One");
  ASSERT_EQ(*res2, "One");

  auto res3 = get_op_name(2);
  ASSERT_FALSE(res3.ok());
  ASSERT_EQ(res3.error(), Error::Internal);
}

TEST(ErrorHandlingTest, ResultWithPointer) {
  auto res = get_ptr(0);
  ASSERT_TRUE(res.ok());
  ASSERT_EQ(res.error(), Error::Ok);
  ASSERT_EQ(res.get(), nullptr);
  ASSERT_EQ(*res, nullptr);

  auto res2 = get_ptr(1);
  ASSERT_TRUE(res2.ok());
  ASSERT_EQ(res2.error(), Error::Ok);
  ASSERT_EQ(res2.get(), test_ptr);
  ASSERT_EQ(*res2, test_ptr);

  auto res3 = get_ptr(2);
  ASSERT_FALSE(res3.ok());
  ASSERT_EQ(res3.error(), Error::InvalidArgument);
}

TEST(ErrorHandlingTest, ResultUnwrap) {
  auto res = get_op_name(-1);
  ASSERT_FALSE(res.ok());
  ASSERT_EQ(res.error(), Error::InvalidArgument);
}

TEST(ErrorHandlingTest, ResultNoCopy) {
  auto res = get_no_copy(2);
  ASSERT_TRUE(res.ok());
  ASSERT_EQ(res.error(), Error::Ok);
  ASSERT_EQ(res.get().getValue(), 2);
  ASSERT_EQ(res->getValue(), 2);

  auto res2 = std::move(res);
  ASSERT_TRUE(res2.ok());
  ASSERT_EQ(res2.error(), Error::Ok);
  ASSERT_EQ(res2.get().getValue(), 2);
  ASSERT_EQ(res2->getValue(), 2);

  Uncopiable& uc = *res2;
  ASSERT_EQ(uc.getValue(), 2);
}

TEST(ErrorHandlingTest, ResultMove) {
  executorch::runtime::runtime_init();

  Result<Movable> res = Movable(2);
  ASSERT_TRUE(res.ok());
  ASSERT_EQ(res.error(), Error::Ok);
  ASSERT_NE(res.get().buffer(), nullptr);
  ASSERT_NE(res->buffer(), nullptr);

  const void* buffer = res->buffer();

  // Move the value.
  Movable m = std::move(*res);
  // The target should point to the same buffer as the source originally did.
  ASSERT_EQ(m.buffer(), buffer);
  // The source inside the Result should no longer point to the buffer.
  ASSERT_EQ(res->buffer(), nullptr);
}
