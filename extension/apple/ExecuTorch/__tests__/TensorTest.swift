/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorch
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
      Tensor<Float>(bytesNoCopy: $0.baseAddress!, shape: [2, 3])
    }
    // Modify the original data to make sure the tensor does not copy the data.
    data.indices.forEach { data[$0] += 1 }

    XCTAssertEqual(tensor.dataType, .float)
    XCTAssertEqual(tensor.shape, [2, 3])
    XCTAssertEqual(tensor.strides, [3, 1])
    XCTAssertEqual(tensor.dimensionOrder, [0, 1])
    XCTAssertEqual(tensor.shapeDynamism, .dynamicBound)
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitBytes() {
    var data: [Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let tensor = data.withUnsafeMutableBytes {
      Tensor<Double>(bytes: $0.baseAddress!, shape: [2, 3])
    }
    // Modify the original data to make sure the tensor copies the data.
    data.indices.forEach { data[$0] += 1 }

    XCTAssertEqual(tensor.dataType, .double)
    XCTAssertEqual(tensor.shape, [2, 3])
    XCTAssertEqual(tensor.strides, [3, 1])
    XCTAssertEqual(tensor.dimensionOrder, [0, 1])
    XCTAssertEqual(tensor.shapeDynamism, .dynamicBound)
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars().map { $0 + 1 }, data)
  }

  func testInitData() {
    let dataArray: [Float] = [1.0, 2.0, 3.0, 4.0]
    let data = Data(bytes: dataArray, count: dataArray.count * MemoryLayout<Float>.size)
    let tensor = Tensor<Float>(data: data, shape: [4])
    XCTAssertEqual(tensor.count, 4)
    XCTAssertEqual(tensor.scalars(), dataArray)
  }

  func testInitDataViewSurvivesSourceScopeEnd() {
    let dataArray: [Float] = [1.0, 2.0, 3.0, 4.0]
    var view: Tensor<Float>!
    autoreleasepool {
      let data = Data(bytes: dataArray, count: dataArray.count * MemoryLayout<Float>.size)
      let tensor = Tensor<Float>(data: data, shape: [4])
      view = Tensor<Float>(tensor)
      XCTAssertEqual(view.scalars(), dataArray)
    }
    XCTAssertEqual(view.count, 4)
    XCTAssertEqual(view.scalars(), dataArray)
  }

  func testWithCustomStridesAndDimensionOrder() {
    let data: [Float] = [1.0, 2.0, 3.0, 4.0]
    let tensor = Tensor<Float>(
      bytes: data.withUnsafeBytes { $0.baseAddress! },
      shape: [2, 2],
      strides: [1, 2],
      dimensionOrder: [1, 0]
    )
    XCTAssertEqual(tensor.shape, [2, 2])
    XCTAssertEqual(tensor.strides, [1, 2])
    XCTAssertEqual(tensor.dimensionOrder, [1, 0])
    XCTAssertEqual(tensor.count, 4)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testMutableBytes() {
    var data: [Int32] = [1, 2, 3, 4]
    let tensor = data.withUnsafeMutableBytes {
      Tensor<Int32>(bytes: $0.baseAddress!, shape: [4])
    }
    tensor.withUnsafeMutableBytes { buffer in
      for i in buffer.indices {
        buffer[i] *= 2
      }
    }
    XCTAssertEqual(tensor.scalars(), data.map { $0 * 2 })
  }

  func testInitWithTensor() throws {
    var data: [Int] = [10, 20, 30, 40]
    let tensor1 = data.withUnsafeMutableBytes {
      Tensor<Int>(bytesNoCopy: $0.baseAddress!, shape: [2, 2])
    }
    let tensor2 = Tensor(tensor1)

    XCTAssertEqual(tensor2.dataType, tensor1.dataType)
    XCTAssertEqual(tensor2.shape, tensor1.shape)
    XCTAssertEqual(tensor2.strides, tensor1.strides)
    XCTAssertEqual(tensor2.dimensionOrder, tensor1.dimensionOrder)
    XCTAssertEqual(tensor2.count, tensor1.count)
    XCTAssertEqual(
      tensor1.withUnsafeMutableBytes { UnsafeMutableRawPointer($0.baseAddress!) },
      tensor2.withUnsafeMutableBytes { UnsafeMutableRawPointer($0.baseAddress!) }
    )

    // Modify the original data to make sure the tensor does not copy the data.
    data.indices.forEach { data[$0] += 1 }

    XCTAssertEqual(tensor1.scalars(), tensor2.scalars())

    try tensor2.resize(to: [4, 1])
    XCTAssertEqual(tensor2.shape, [4, 1])
    XCTAssertEqual(tensor1.shape, [2, 2])
    XCTAssertEqual(tensor2.strides, [1, 1])
    XCTAssertEqual(tensor1.strides, [2, 1])
    XCTAssertEqual(tensor2.dimensionOrder, tensor1.dimensionOrder)
    XCTAssertEqual(tensor2.count, tensor1.count)
  }

  func testInitWithTensorDerivesStridesAndSharesStorage() {
    var scalars: [Int32] = [1, 2, 3, 4, 5, 6]
    let tensor1 = scalars.withUnsafeMutableBytes {
      Tensor<Int32>(bytesNoCopy: $0.baseAddress!, shape: [2, 3])
    }
    let tensor2 = Tensor(tensor1, shape: [3, 2])

    XCTAssertEqual(tensor1.withUnsafeBytes { $0.baseAddress }, tensor2.withUnsafeBytes { $0.baseAddress })
    XCTAssertEqual(tensor2.shape, [3, 2])
    XCTAssertEqual(tensor2.strides, [2, 1])
    XCTAssertEqual(tensor2.dimensionOrder, [0, 1])

    scalars[0] = 99
    XCTAssertEqual(tensor2.withUnsafeBytes { $0[0] }, 99)
  }

  func testInitWithTensorExplicitOverridesAppliesMetadata() {
    var scalars: [Float] = [1, 2, 3, 4]
    let tensor1 = scalars.withUnsafeMutableBytes {
      Tensor<Float>(bytesNoCopy: $0.baseAddress!, shape: [2, 2])
    }
    let tensor2 = Tensor(tensor1, shape: [2, 2], dimensionOrder: [1, 0], strides: [1, 2])

    XCTAssertEqual(tensor1.withUnsafeBytes { $0.baseAddress }, tensor2.withUnsafeBytes { $0.baseAddress })
    XCTAssertEqual(tensor2.shape, [2, 2])
    XCTAssertEqual(tensor2.dimensionOrder, [1, 0])
    XCTAssertEqual(tensor2.strides, [1, 2])

    scalars[3] = 42
    XCTAssertEqual(tensor2.withUnsafeBytes { $0[3] }, 42)
  }

  func testCopy() {
    var data: [Double] = [10.0, 20.0, 30.0, 40.0]
    let tensor1 = data.withUnsafeMutableBytes {
      Tensor<Double>(bytesNoCopy: $0.baseAddress!, shape: [2, 2])
    }
    let tensor2 = tensor1.copy()

    XCTAssertEqual(tensor1.dataType, tensor2.dataType)
    XCTAssertEqual(tensor1.shape, tensor2.shape)
    XCTAssertEqual(tensor1.strides, tensor2.strides)
    XCTAssertEqual(tensor1.dimensionOrder, tensor2.dimensionOrder)
    XCTAssertEqual(tensor1.count, tensor2.count)
  }

  func testCopyToSameDataType() {
    let tensor1 = Tensor<Float>([1, 2, 3, 4], shape: [2, 2])
    let tensor2 = tensor1.copy(to: Float.self)
    XCTAssertEqual(tensor2.dataType, .float)
    XCTAssertEqual(tensor2.shape, [2, 2])
    XCTAssertEqual(tensor2.strides, tensor1.strides)
    XCTAssertEqual(tensor2.dimensionOrder, tensor1.dimensionOrder)
    XCTAssertEqual(tensor2.scalars(), [1, 2, 3, 4])
  }

  func testCopyToDifferentDataTypeKeepsSourceAlive() {
    var data = [10.0, 20.0, 30.0, 40.0]
    let tensor1 = data.withUnsafeMutableBytes {
      Tensor<Double>(bytesNoCopy: $0.baseAddress!, shape: [2, 2])
    }
    let tensor2 = tensor1.copy(to: Float.self)
    data[0] = 999.0
    XCTAssertEqual(tensor2.dataType, .float)
    XCTAssertEqual(tensor2.shape, [2, 2])
    XCTAssertEqual(tensor2.scalars(), [10.0, 20.0, 30.0, 40.0])
  }

  func testCopyToPreservesShapeAndOrderOn2D() {
    let tensor1 = Tensor<Int32>(
      [1, 2, 3, 4, 5, 6],
      shape: [2, 3],
      strides: [3, 1],
      dimensionOrder: [0, 1]
    )
    let tensor2 = tensor1.copy(to: Double.self)
    XCTAssertEqual(tensor2.shape, [2, 3])
    XCTAssertEqual(tensor2.strides, [3, 1])
    XCTAssertEqual(tensor2.dimensionOrder, [0, 1])
    XCTAssertEqual(tensor2.count, 6)
    XCTAssertEqual(tensor2.scalars(), [1, 2, 3, 4, 5, 6])
  }

  func testResize() {
    var data: [Int] = [1, 2, 3, 4]
    let tensor = data.withUnsafeMutableBytes {
      Tensor<Int>(bytesNoCopy: $0.baseAddress!, shape: [4, 1])
    }
    XCTAssertNoThrow(try tensor.resize(to: [2, 2]))
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [2, 2])
    XCTAssertEqual(tensor.strides, [2, 1])
    XCTAssertEqual(tensor.dimensionOrder, [0, 1])
    XCTAssertEqual(tensor.count, 4)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testResizeError() {
    var data: [Int] = [1, 2, 3, 4]
    let tensor = data.withUnsafeMutableBytes {
      Tensor<Int>(bytesNoCopy: $0.baseAddress!, shape: [4, 1])
    }
    XCTAssertThrowsError(try tensor.resize(to: [2, 3]))
  }

  func testIsEqual() {
    var data: [Float] = [1.0, 2.0, 3.0, 4.0]
    let tensor1 = data.withUnsafeMutableBytes {
      Tensor<Float>(bytesNoCopy: $0.baseAddress!, shape: [2, 2])
    }
    let tensor2 = Tensor(tensor1)
    XCTAssertEqual(tensor1, tensor2)
    XCTAssertEqual(tensor2, tensor1)

    var dataModified: [Float] = [1.0, 2.0, 3.0, 5.0]
    let tensor3 = dataModified.withUnsafeMutableBytes {
      Tensor<Float>(bytesNoCopy: $0.baseAddress!, shape: [2, 2])
    }
    XCTAssertNotEqual(tensor1, tensor3)
    let tensor4 = data.withUnsafeMutableBytes {
      Tensor<Float>(bytesNoCopy: $0.baseAddress!, shape: [4, 1])
    }
    XCTAssertNotEqual(tensor1, tensor4)
    XCTAssertEqual(tensor1, tensor1)
    XCTAssertNotEqual(tensor4, tensor2)
    let tensor5 = data.withUnsafeMutableBytes {
      Tensor<Float>(bytesNoCopy: $0.baseAddress!, shape: [2, 2], shapeDynamism: .static)
    }
    XCTAssertEqual(tensor1, tensor5)
  }

  func testInitScalarsNoCopyWithExplicitParams() throws {
    var data: [Int] = [10, 20, 30, 40]
    let tensor = Tensor(
      &data,
      shape: [2, 2],
      strides: [1, 2],
      dimensionOrder: [1, 0],
      shapeDynamism: .static
    )
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [2, 2])
    XCTAssertEqual(tensor.strides, [1, 2])
    XCTAssertEqual(tensor.dimensionOrder, [1, 0])
    XCTAssertEqual(tensor.shapeDynamism, .static)
    XCTAssertEqual(tensor.count, 4)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyUInt8() {
    var data: [UInt8] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .byte)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsUInt8() {
    let data: [UInt8] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .byte)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyInt8() {
    var data: [Int8] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .char)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsInt8() {
    let data: [Int8] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .char)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyInt16() {
    var data: [Int16] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .short)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsInt16() {
    let data: [Int16] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .short)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyInt32() {
    var data: [Int32] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .int)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsInt32() {
    let data: [Int32] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .int)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyInt64() {
    var data: [Int64] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsInt64() {
    let data: [Int64] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyFloat() {
    var data: [Float] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .float)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsFloat() {
    let data: [Float] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .float)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyDouble() {
    var data: [Double] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .double)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsDouble() {
    let data: [Double] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .double)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyBool() {
    var data: [Bool] = [true, false, true, false, true, false]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .bool)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = false
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsBool() {
    let data: [Bool] = [true, false, true, false, true, false]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .bool)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyUInt16() {
    var data: [UInt16] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .uInt16)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsUInt16() {
    let data: [UInt16] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .uInt16)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyUInt32() {
    var data: [UInt32] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .uInt32)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsUInt32() {
    let data: [UInt32] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .uInt32)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyUInt64() {
    var data: [UInt64] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .uInt64)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsUInt64() {
    let data: [UInt64] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .uInt64)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyInt() {
    var data: [Int] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsInt() {
    let data: [Int] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsNoCopyUInt() {
    var data: [UInt] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(&data)
    XCTAssertEqual(tensor.dataType, .uInt64)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
    data[2] = 42
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitScalarsUInt() {
    let data: [UInt] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data)
    XCTAssertEqual(tensor.dataType, .uInt64)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertEqual(tensor.scalars(), data)
  }

  func testInitInt8() {
    let tensor = Tensor(Int8(42))
    XCTAssertEqual(tensor.dataType, .char)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testInitInt16() {
    let tensor = Tensor(Int16(42))
    XCTAssertEqual(tensor.dataType, .short)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testInitInt32() {
    let tensor = Tensor(Int32(42))
    XCTAssertEqual(tensor.dataType, .int)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testInitInt64() {
    let tensor = Tensor(Int64(42))
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testInitUInt8() {
    let tensor = Tensor(UInt8(42))
    XCTAssertEqual(tensor.dataType, .byte)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testInitUInt16() {
    let tensor = Tensor(UInt16(42))
    XCTAssertEqual(tensor.dataType, .uInt16)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testInitUInt32() {
    let tensor = Tensor(UInt32(42))
    XCTAssertEqual(tensor.dataType, .uInt32)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testInitUInt64() {
    let tensor = Tensor(UInt64(42))
    XCTAssertEqual(tensor.dataType, .uInt64)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testInitBool() {
    let tensor = Tensor(true)
    XCTAssertEqual(tensor.dataType, .bool)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, true)
  }

  func testInitFloat() {
    let tensor = Tensor(Float(42.0))
    XCTAssertEqual(tensor.dataType, .float)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testInitDouble() {
    let tensor = Tensor(42.0)
    XCTAssertEqual(tensor.dataType, .double)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testInitInt() {
    let tensor = Tensor(42)
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testInitUInt() {
    let tensor = Tensor(UInt(42))
    XCTAssertEqual(tensor.dataType, .uInt64)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertEqual(tensor.scalars().first, 42)
  }

  func testExtractAnyTensorMatchesOriginalDataAndMetadata() {
    let tensor = Tensor([1, 2, 3, 4], shape: [2, 2])
    let anyTensor = tensor.anyTensor
    XCTAssertEqual(anyTensor.shape, tensor.shape)
    XCTAssertEqual(anyTensor.strides, tensor.strides)
    XCTAssertEqual(anyTensor.dimensionOrder, tensor.dimensionOrder)
    XCTAssertEqual(anyTensor.count, tensor.count)
    XCTAssertEqual(anyTensor.dataType, tensor.dataType)
    XCTAssertEqual(anyTensor.shapeDynamism, tensor.shapeDynamism)
    let newTensor = Tensor<Int>(anyTensor)
    XCTAssertEqual(newTensor, tensor)
  }

  func testReconstructGenericTensorViaInitAndAsTensor() {
    let tensor = Tensor([5, 6, 7])
    let anyTensor = tensor.anyTensor
    let tensorInit = Tensor<Int>(anyTensor)
    let tensorFromAny: Tensor<Int> = anyTensor.asTensor()!
    XCTAssertEqual(tensorInit, tensorFromAny)
  }

  func testAsTensorMismatchedTypeReturnsNil() {
    let tensor = Tensor([8, 9, 10])
    let anyTensor = tensor.anyTensor
    let wrongTypedTensor: Tensor<Float>? = anyTensor.asTensor()
    XCTAssertNil(wrongTypedTensor)
  }

  func testViewSharesDataAndResizeAltersShapeNotData() throws {
    var scalars = [11, 12, 13, 14]
    let tensor = Tensor(&scalars, shape: [2, 2])
    let viewTensor = Tensor(tensor)
    let scalarsAddress = scalars.withUnsafeBufferPointer { $0.baseAddress }
    let tensorDataAddress = tensor.withUnsafeBytes { $0.baseAddress }
    let viewTensorDataAddress = viewTensor.withUnsafeBytes { $0.baseAddress }
    XCTAssertEqual(tensorDataAddress, scalarsAddress)
    XCTAssertEqual(tensorDataAddress, viewTensorDataAddress)

    scalars[2] = 42
    XCTAssertEqual(tensor.scalars(), scalars)
    XCTAssertEqual(viewTensor.scalars(), scalars)

    XCTAssertNoThrow(try viewTensor.resize(to: [4, 1]))
    XCTAssertEqual(viewTensor.shape, [4, 1])
    XCTAssertEqual(tensor.shape, [2, 2])
    XCTAssertEqual(tensor.scalars(), scalars)
    XCTAssertEqual(viewTensor.scalars(), scalars)
  }

  func testMultipleGenericFromAnyReflectChanges() {
    let tensor = Tensor([2, 4, 6, 8], shape: [2, 2])
    let anyTensor = tensor.anyTensor
    let tensor1: Tensor<Int> = anyTensor.asTensor()!
    let tensor2: Tensor<Int> = anyTensor.asTensor()!

    XCTAssertEqual(tensor1, tensor2)
    tensor1.withUnsafeMutableBytes { $0[1] = 42 }
    XCTAssertEqual(tensor2.withUnsafeBytes { $0[1] }, 42)
  }

  func testEmpty() {
    let tensor = Tensor<Float>.empty(shape: [3, 4])
    XCTAssertEqual(tensor.shape, [3, 4])
    XCTAssertEqual(tensor.count, 12)
    tensor.withUnsafeBytes { buffer in
      XCTAssertNotNil(buffer.baseAddress)
      XCTAssertEqual(buffer.count, 12)
      XCTAssertEqual(tensor.dataType, .float)
    }
  }

  func testEmptyLike() {
    let other = Tensor<Float>.empty(shape: [2, 2])
    let tensor = Tensor<Float>.empty(like: other)
    XCTAssertEqual(tensor.shape, other.shape)
    XCTAssertEqual(tensor.strides, other.strides)
    XCTAssertEqual(tensor.dimensionOrder, other.dimensionOrder)
    XCTAssertEqual(tensor.dataType, other.dataType)
  }

  func testFull() {
    let tensor = Tensor<Int32>.full(shape: [2, 2], scalar: 7)
    XCTAssertEqual(tensor.shape, [2, 2])
    XCTAssertEqual(tensor.count, 4)
    tensor.withUnsafeBytes { buffer in
      for value in buffer {
        XCTAssertEqual(value, 7)
      }
    }
  }

  func testFullLike() {
    let other = Tensor<Float>.empty(shape: [2, 2])
    let tensor = Tensor<Float>.full(like: other, scalar: 42)
    XCTAssertEqual(tensor.shape, other.shape)
    tensor.withUnsafeBytes { buffer in
      for value in buffer {
        XCTAssertEqual(value, 42.0)
      }
    }
  }

  func testOnes() {
    let tensor = Tensor<Float>.ones(shape: [2, 3])
    XCTAssertEqual(tensor.shape, [2, 3])
    XCTAssertEqual(tensor.count, 6)
    tensor.withUnsafeBytes { buffer in
      for value in buffer {
        XCTAssertEqual(value, 1.0)
      }
    }
  }

  func testOnesLike() {
    let other = Tensor<Double>.empty(shape: [2, 4])
    let tensor = Tensor<Double>.ones(like: other)
    XCTAssertEqual(tensor.shape, other.shape)
    tensor.withUnsafeBytes { buffer in
      for value in buffer {
        XCTAssertEqual(value, 1.0)
      }
    }
  }

  func testZeros() {
    let tensor = Tensor<Double>.zeros(shape: [2, 3])
    XCTAssertEqual(tensor.shape, [2, 3])
    XCTAssertEqual(tensor.count, 6)
    tensor.withUnsafeBytes { buffer in
      for value in buffer {
        XCTAssertEqual(value, 0)
      }
    }
  }

  func testZerosLike() {
    let other = Tensor<Int32>.full(shape: [3, 2], scalar: 9)
    let tensor = Tensor<Int32>.zeros(like: other)
    XCTAssertEqual(tensor.shape, other.shape)
    tensor.withUnsafeBytes { buffer in
      for value in buffer {
        XCTAssertEqual(value, 0)
      }
    }
  }

  func testRandom() {
    let tensor = Tensor<Float>.rand(shape: [3, 3])
    XCTAssertEqual(tensor.shape, [3, 3])
    XCTAssertEqual(tensor.count, 9)
    tensor.withUnsafeBytes { buffer in
      let uniqueValues = Set(buffer)
      XCTAssertTrue(uniqueValues.count > 1)
    }
  }

  func testRandomLike() {
    let other = Tensor<Int>.full(shape: [3, 3], scalar: 9)
    let tensor = Tensor<Int>.rand(like: other)
    XCTAssertEqual(tensor.shape, other.shape)
    XCTAssertEqual(tensor.count, other.count)
  }

  func testRandomNormal() {
    let tensor = Tensor<Double>.randn(shape: [4])
    XCTAssertEqual(tensor.shape, [4])
    XCTAssertEqual(tensor.count, 4)
    tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(buffer.count, 4)
    }
  }

  func testRandomNormalLike() {
    let other = Tensor<Float>.zeros(shape: [4])
    let tensor = Tensor<Float>.randn(like: other)
    XCTAssertEqual(tensor.shape, other.shape)
    XCTAssertEqual(tensor.count, other.count)
  }

  func testRandomInteger() {
    let tensor = Tensor<Int>.randint(low: 10, high: 20, shape: [5])
    XCTAssertEqual(tensor.shape, [5])
    XCTAssertEqual(tensor.count, 5)
    tensor.withUnsafeBytes { buffer in
      for value in buffer {
        XCTAssertTrue(value >= 10 && value < 20)
      }
    }
  }

  func testRandomIntegerLike() {
    let other = Tensor<Int>.ones(shape: [5])
    let tensor = Tensor<Int>.randint(like: other, low: 100, high: 200)
    tensor.withUnsafeBytes { buffer in
      for value in buffer {
        XCTAssertTrue(value >= 100 && value < 200)
      }
    }
  }
}
