/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@testable import ExecuTorch

import XCTest

class ValueTest: XCTestCase {
  func testNone() {
    let value = Value()
    XCTAssertTrue(value.isNone)
  }

  func testTensor() {
    let tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    let value = Value(tensor)
    XCTAssertTrue(value.isTensor)
    XCTAssertEqual(value.tensor, tensor)
  }

  func testString() {
    let value = Value("hello")
    XCTAssertTrue(value.isString)
    XCTAssertEqual(value.string, "hello")
  }

  func testBoolean() {
    let value = Value(true)
    XCTAssertTrue(value.isBoolean)
    XCTAssertEqual(value.boolean, true)
  }

  func testInteger() {
    let value = Value(42)
    XCTAssertTrue(value.isInteger)
    XCTAssertEqual(value.integer, 42)
  }

  func testDouble() {
    let value = Value(3.14)
    XCTAssertTrue(value.isDouble)
    XCTAssertEqual(value.double, 3.14)
  }
}
