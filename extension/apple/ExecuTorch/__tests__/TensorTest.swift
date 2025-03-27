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
    // Modify the original data to make sure the tensor does not copy the data.
    data.indices.forEach { data[$0] += 1 }

    XCTAssertEqual(tensor.dataType, .float)
    XCTAssertEqual(tensor.shape, [2, 3])
    XCTAssertEqual(tensor.strides, [3, 1])
    XCTAssertEqual(tensor.dimensionOrder, [0, 1])
    XCTAssertEqual(tensor.shapeDynamism, .dynamicBound)
    XCTAssertEqual(tensor.count, 6)

    tensor.bytes { pointer, count, dataType in
      XCTAssertEqual(dataType, .float)
      XCTAssertEqual(count, 6)
      XCTAssertEqual(size(ofDataType: dataType), 4)
      XCTAssertEqual(Array(UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Float.self), count: count)), data)
    }
  }

  func testInitBytes() {
    var data: [Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let tensor = data.withUnsafeMutableBytes {
      Tensor(bytes: $0.baseAddress!, shape: [2, 3], dataType: .double)
    }
    // Modify the original data to make sure the tensor copies the data.
    data.indices.forEach { data[$0] += 1 }

    XCTAssertEqual(tensor.dataType, .double)
    XCTAssertEqual(tensor.shape, [2, 3])
    XCTAssertEqual(tensor.strides, [3, 1])
    XCTAssertEqual(tensor.dimensionOrder, [0, 1])
    XCTAssertEqual(tensor.shapeDynamism, .dynamicBound)
    XCTAssertEqual(tensor.count, 6)

    tensor.bytes { pointer, count, dataType in
      XCTAssertEqual(dataType, .double)
      XCTAssertEqual(count, 6)
      XCTAssertEqual(size(ofDataType: dataType), 8)
      XCTAssertEqual(Array(UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Double.self), count: count)).map { $0 + 1 }, data)
    }
  }

  func testInitData() {
    let dataArray: [Float] = [1.0, 2.0, 3.0, 4.0]
    let data = Data(bytes: dataArray, count: dataArray.count * MemoryLayout<Float>.size)
    let tensor = Tensor(data: data, shape: [4], dataType: .float)
    XCTAssertEqual(tensor.count, 4)
    tensor.bytes { pointer, count, dataType in
      XCTAssertEqual(Array(UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Float.self), count: count)), dataArray)
    }
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

    tensor.bytes { pointer, count, dataType in
      XCTAssertEqual(Array(UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Float.self), count: count)), data)
    }
  }

  func testMutableBytes() {
    var data: [Int32] = [1, 2, 3, 4]
    let tensor = data.withUnsafeMutableBytes {
      Tensor(bytes: $0.baseAddress!, shape: [4], dataType: .int)
    }
    tensor.mutableBytes { pointer, count, dataType in
      XCTAssertEqual(dataType, .int)
      let buffer = pointer.assumingMemoryBound(to: Int32.self)
      for i in 0..<count {
        buffer[i] *= 2
      }
    }
    tensor.bytes { pointer, count, dataType in
      let updatedData = Array(UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Int32.self), count: count))
      XCTAssertEqual(updatedData, [2, 4, 6, 8])
    }
  }

  func testInitWithTensor() {
    var data: [Int] = [10, 20, 30, 40]
    let tensor1 = data.withUnsafeMutableBytes {
      Tensor(bytesNoCopy: $0.baseAddress!, shape: [2, 2], dataType: .int)
    }
    let tensor2 = Tensor(tensor1)

    XCTAssertEqual(tensor2.dataType, tensor1.dataType)
    XCTAssertEqual(tensor2.shape, tensor1.shape)
    XCTAssertEqual(tensor2.strides, tensor1.strides)
    XCTAssertEqual(tensor2.dimensionOrder, tensor1.dimensionOrder)
    XCTAssertEqual(tensor2.count, tensor1.count)
  }

  func testCopy() {
    var data: [Double] = [10.0, 20.0, 30.0, 40.0]
    let tensor1 = data.withUnsafeMutableBytes {
      Tensor(bytesNoCopy: $0.baseAddress!, shape: [2, 2], dataType: .double)
    }
    let tensor2 = tensor1.copy()

    XCTAssertEqual(tensor1.dataType, tensor2.dataType)
    XCTAssertEqual(tensor1.shape, tensor2.shape)
    XCTAssertEqual(tensor1.strides, tensor2.strides)
    XCTAssertEqual(tensor1.dimensionOrder, tensor2.dimensionOrder)
    XCTAssertEqual(tensor1.count, tensor2.count)
  }

  func testResize() {
    var data: [Int] = [1, 2, 3, 4]
    let tensor = data.withUnsafeMutableBytes {
      Tensor(bytesNoCopy: $0.baseAddress!, shape: [4, 1], dataType: .int)
    }
    XCTAssertNoThrow(try tensor.resize(to: [2, 2]))
    XCTAssertEqual(tensor.dataType, .int)
    XCTAssertEqual(tensor.shape, [2, 2])
    XCTAssertEqual(tensor.strides, [2, 1])
    XCTAssertEqual(tensor.dimensionOrder, [0, 1])
    XCTAssertEqual(tensor.count, 4)

    tensor.bytes { pointer, count, dataType in
      XCTAssertEqual(Array(UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Int.self), count: count)), data)
    }
  }

  func testResizeError() {
    var data: [Int] = [1, 2, 3, 4]
    let tensor = data.withUnsafeMutableBytes {
      Tensor(bytesNoCopy: $0.baseAddress!, shape: [4, 1], dataType: .int)
    }
    XCTAssertThrowsError(try tensor.resize(to: [2, 3]))
  }

  func testIsEqual() {
    var data: [Float] = [1.0, 2.0, 3.0, 4.0]
    let tensor1 = data.withUnsafeMutableBytes {
      Tensor(bytesNoCopy: $0.baseAddress!, shape: [2, 2], dataType: .float)
    }
    let tensor2 = Tensor(tensor1)
    XCTAssertTrue(tensor1.isEqual(tensor2))
    XCTAssertTrue(tensor2.isEqual(tensor1))

    var dataModified: [Float] = [1.0, 2.0, 3.0, 5.0]
    let tensor3 = dataModified.withUnsafeMutableBytes {
      Tensor(bytesNoCopy: $0.baseAddress!, shape: [2, 2], dataType: .float)
    }
    XCTAssertFalse(tensor1.isEqual(tensor3))
    let tensor4 = data.withUnsafeMutableBytes {
      Tensor(bytesNoCopy: $0.baseAddress!, shape: [4, 1], dataType: .float)
    }
    XCTAssertFalse(tensor1.isEqual(tensor4))
    XCTAssertTrue(tensor1.isEqual(tensor1))
    XCTAssertFalse(tensor1.isEqual(NSString(string: "Not a tensor")))
    XCTAssertFalse(tensor4.isEqual(tensor2.copy()))
  }

  func testInitScalarsFloat() {
    let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let tensor = Tensor(data.map(NSNumber.init), shape: [2, 3], strides: [3, 1], dimensionOrder: [0, 1], dataType: .float, shapeDynamism: .dynamicBound)
    XCTAssertEqual(tensor.dataType, .float)
    XCTAssertEqual(tensor.shape, [2, 3])
    XCTAssertEqual(tensor.strides, [3, 1])
    XCTAssertEqual(tensor.dimensionOrder, [0, 1])
    XCTAssertEqual(tensor.count, 6)
    tensor.bytes { pointer, count, dataType in
      XCTAssertEqual(Array(UnsafeBufferPointer(start: pointer.assumingMemoryBound(to: Float.self), count: count)), data)
    }
  }
}
