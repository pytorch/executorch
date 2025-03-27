/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@testable import ExecuTorch

import XCTest

class TensorTest: XCTestCase {
  func testInitBytesNoCopy() {
    var data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let tensor = data.withUnsafeMutableBytes {
      Tensor(bytesNoCopy: $0.baseAddress!, shape: [2, 3], dataType: .float)
    }
    XCTAssertEqual(tensor.dataType, .float)
    XCTAssertEqual(tensor.shape, [2, 3])
    XCTAssertEqual(tensor.strides, [3, 1])
    XCTAssertEqual(tensor.dimensionOrder, [0, 1])
    XCTAssertEqual(tensor.shapeDynamism, .dynamicBound)
    XCTAssertEqual(tensor.count, 6)
  }
}
