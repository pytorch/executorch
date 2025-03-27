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
  func testElementCountOfShape() {
    XCTAssertEqual(elementCount(ofShape: [2, 3, 4]), 24)
    XCTAssertEqual(elementCount(ofShape: [5]), 5)
    XCTAssertEqual(elementCount(ofShape: []), 1)
  }

  func testSizeOfDataType() {
    let expectedSizes: [DataType: Int] = [
      .byte: 1,
      .char: 1,
      .short: 2,
      .int: 4,
      .long: 8,
      .half: 2,
      .float: 4,
      .double: 8,
      .complexHalf: 4,
      .complexFloat: 8,
      .complexDouble: 16,
      .bool: 1,
      .qInt8: 1,
      .quInt8: 1,
      .qInt32: 4,
      .bFloat16: 2,
      .quInt4x2: 1,
      .quInt2x4: 1,
      .bits1x8: 1,
      .bits2x4: 1,
      .bits4x2: 1,
      .bits8: 1,
      .bits16: 2,
      .float8_e5m2: 1,
      .float8_e4m3fn: 1,
      .float8_e5m2fnuz: 1,
      .float8_e4m3fnuz: 1,
      .uInt16: 2,
      .uInt32: 4,
      .uInt64: 8,
    ]
    for (dataType, expectedSize) in expectedSizes {
      XCTAssertEqual(size(ofDataType: dataType), expectedSize, "Size for \(dataType) should be \(expectedSize)")
    }
  }

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

  func testInitBytes() {
    var data: [Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let tensor = data.withUnsafeMutableBytes {
      Tensor(bytes: $0.baseAddress!, shape: [2, 3], dataType: .double)
    }
    XCTAssertEqual(tensor.dataType, .double)
    XCTAssertEqual(tensor.shape, [2, 3])
    XCTAssertEqual(tensor.strides, [3, 1])
    XCTAssertEqual(tensor.dimensionOrder, [0, 1])
    XCTAssertEqual(tensor.shapeDynamism, .dynamicBound)
    XCTAssertEqual(tensor.count, 6)
  }

  func testWithCustomStridesAndDimensionOrder() {
    let data: [Float] = [1.0, 2.0, 3.0, 4.0]
    let tensor = Tensor(
      bytes: data.withUnsafeBytes { $0.baseAddress! },
      shape: [2, 2],
      strides: [1, 2],
      dimensionOrder: [1, 0],
      dataType: .float
    )
    XCTAssertEqual(tensor.shape, [2, 2])
    XCTAssertEqual(tensor.strides, [1, 2])
    XCTAssertEqual(tensor.dimensionOrder, [1, 0])
    XCTAssertEqual(tensor.count, 4)
  }
}
