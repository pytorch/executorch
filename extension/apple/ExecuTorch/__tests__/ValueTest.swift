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
}
