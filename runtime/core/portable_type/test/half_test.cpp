/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/portable_type/half.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>
#include <cmath>

using executorch::runtime::etensor::Half;

namespace {

/**
 * According to the precision limitations listed here:
 * https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations
 * The max precision error for a half in the range [2^n, 2^(n+1)] is 2^(n-10)
 */
float toleranceFloat16(float f) {
  return pow(2, static_cast<int>(log2(fabs(f))) - 10);
}

bool closeEnoughFloat16(float a, float b) {
  return fabs(a - b) <= toleranceFloat16(fmax(fabs(a), fabs(b)));
}

} // namespace

/// Arithmetic with Halfs

TEST(HalfTest, ArithmeticHalfAdd) {
  float af = 104.35;
  Half ah(af);
  float bf = 72.5;
  Half bh(bf);
  EXPECT_TRUE(closeEnoughFloat16(ah + bh, af + bf));
  ah += bh;
  af += bf;
  EXPECT_TRUE(closeEnoughFloat16(ah, af));
}

TEST(HalfTest, ArithmeticHalfSub) {
  float af = 31.4;
  Half ah(af);
  float bf = 20.5;
  Half bh(bf);
  EXPECT_TRUE(closeEnoughFloat16(ah - bh, af - bf));
  ah -= bh;
  af -= bf;
  EXPECT_TRUE(closeEnoughFloat16(ah, af));
}

TEST(HalfTest, ArithmeticHalfMul) {
  float af = 85.5;
  Half ah(af);
  float bf = 17.5;
  Half bh(bf);
  EXPECT_TRUE(closeEnoughFloat16(ah * bh, af * bf));
  ah *= bh;
  af *= bf;
  EXPECT_TRUE(closeEnoughFloat16(ah, af));
}

TEST(HalfTest, ArithmeticHalfDiv) {
  float af = 96.9;
  Half ah(af);
  float bf = 12.5;
  Half bh(bf);
  EXPECT_TRUE(closeEnoughFloat16(ah / bh, af / bf));
  ah /= bh;
  af /= bf;
  EXPECT_TRUE(closeEnoughFloat16(ah, af));
}

/// Arithmetic with floats

TEST(HalfTest, ArithmeticFloatAdd) {
  float af = 104.35;
  Half ah(af);
  float b = 72.5;
  EXPECT_TRUE(closeEnoughFloat16(ah + b, af + b));
  EXPECT_TRUE(closeEnoughFloat16(b + ah, b + af));
}

TEST(HalfTest, ArithmeticFloatSub) {
  float af = 31.4;
  Half ah(af);
  float b = 20.5;
  EXPECT_TRUE(closeEnoughFloat16(ah - b, af - b));
  EXPECT_TRUE(closeEnoughFloat16(b - ah, b - af));
}

TEST(HalfTest, ArithmeticFloatMul) {
  float af = 85.5;
  Half ah(af);
  float b = 17.5;
  EXPECT_TRUE(closeEnoughFloat16(ah * b, af * b));
  EXPECT_TRUE(closeEnoughFloat16(b * ah, b * af));
}

TEST(HalfTest, ArithmeticFloatDiv) {
  float af = 96.9;
  Half ah(af);
  float b = 12.5;
  EXPECT_TRUE(closeEnoughFloat16(ah / b, af / b));
  EXPECT_TRUE(closeEnoughFloat16(b / ah, b / af));
}

/// Arithmetic with doubles

TEST(HalfTest, ArithmeticDoubleAdd) {
  float af = 104.35;
  Half ah(af);
  double b = 72.5;
  EXPECT_TRUE(closeEnoughFloat16(ah + b, af + b));
  EXPECT_TRUE(closeEnoughFloat16(b + ah, b + af));
}

TEST(HalfTest, ArithmeticDoubleSub) {
  float af = 31.4;
  Half ah(af);
  double b = 20.5;
  EXPECT_TRUE(closeEnoughFloat16(ah - b, af - b));
  EXPECT_TRUE(closeEnoughFloat16(b - ah, b - af));
}

TEST(HalfTest, ArithmeticDoubleMul) {
  float af = 85.5;
  Half ah(af);
  double b = 17.5;
  EXPECT_TRUE(closeEnoughFloat16(ah * b, af * b));
  EXPECT_TRUE(closeEnoughFloat16(b * ah, b * af));
}

TEST(HalfTest, ArithmeticDoubleDiv) {
  float af = 96.9;
  Half ah(af);
  double b = 12.5;
  EXPECT_TRUE(closeEnoughFloat16(ah / b, af / b));
  EXPECT_TRUE(closeEnoughFloat16(b / ah, b / af));
}

/// Arithmetic with ints

TEST(HalfTest, ArithmeticInt32Add) {
  float af = 104.35;
  Half ah(af);
  int32_t b = 72;
  EXPECT_TRUE(closeEnoughFloat16(ah + b, af + b));
  EXPECT_TRUE(closeEnoughFloat16(b + ah, b + af));
}

TEST(HalfTest, ArithmeticInt32Sub) {
  float af = 31.4;
  Half ah(af);
  int32_t b = 20;
  EXPECT_TRUE(closeEnoughFloat16(ah - b, af - b));
  EXPECT_TRUE(closeEnoughFloat16(b - ah, b - af));
}

TEST(HalfTest, ArithmeticInt32Mul) {
  float af = 85.5;
  Half ah(af);
  int32_t b = 17;
  EXPECT_TRUE(closeEnoughFloat16(ah * b, af * b));
  EXPECT_TRUE(closeEnoughFloat16(b * ah, b * af));
}

TEST(HalfTest, ArithmeticInt32Div) {
  float af = 96.9;
  Half ah(af);
  int32_t b = 12;
  EXPECT_TRUE(closeEnoughFloat16(ah / b, af / b));
  EXPECT_TRUE(closeEnoughFloat16(b / ah, b / af));
}

//// Arithmetic with int64_t

TEST(HalfTest, ArithmeticInt64Add) {
  float af = 104.35;
  Half ah(af);
  int64_t b = 72;
  EXPECT_TRUE(closeEnoughFloat16(ah + b, af + b));
  EXPECT_TRUE(closeEnoughFloat16(b + ah, b + af));
}

TEST(HalfTest, ArithmeticInt64Sub) {
  float af = 31.4;
  Half ah(af);
  int64_t b = 20;
  EXPECT_TRUE(closeEnoughFloat16(ah - b, af - b));
  EXPECT_TRUE(closeEnoughFloat16(b - ah, b - af));
}

TEST(HalfTest, ArithmeticInt64Mul) {
  float af = 85.5;
  Half ah(af);
  int64_t b = 17;
  EXPECT_TRUE(closeEnoughFloat16(ah * b, af * b));
  EXPECT_TRUE(closeEnoughFloat16(b * ah, b * af));
}

TEST(HalfTest, ArithmeticInt64Div) {
  float af = 96.9;
  Half ah(af);
  int64_t b = 12;
  EXPECT_TRUE(closeEnoughFloat16(ah / b, af / b));
  EXPECT_TRUE(closeEnoughFloat16(b / ah, b / af));
}
