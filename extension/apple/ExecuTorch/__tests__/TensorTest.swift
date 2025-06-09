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
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
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
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer).map { $0 + 1 }, data)
    })
  }

  func testInitData() {
    let dataArray: [Float] = [1.0, 2.0, 3.0, 4.0]
    let data = Data(bytes: dataArray, count: dataArray.count * MemoryLayout<Float>.size)
    let tensor = Tensor(data: data, shape: [4], dataType: .float)
    XCTAssertEqual(tensor.count, 4)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), dataArray)
    })
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
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testMutableBytes() {
    var data: [Int32] = [1, 2, 3, 4]
    let tensor = data.withUnsafeMutableBytes {
      Tensor(bytes: $0.baseAddress!, shape: [4], dataType: .int)
    }
    XCTAssertNoThrow(try tensor.withUnsafeMutableBytes { (buffer: UnsafeMutableBufferPointer<Int32>) in
      for i in buffer.indices {
        buffer[i] *= 2
      }
    })
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Int32>) in
      XCTAssertEqual(Array(buffer), [2, 4, 6, 8])
    })
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
      Tensor(bytesNoCopy: $0.baseAddress!, shape: [4, 1], dataType: .long)
    }
    XCTAssertNoThrow(try tensor.resize(to: [2, 2]))
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [2, 2])
    XCTAssertEqual(tensor.strides, [2, 1])
    XCTAssertEqual(tensor.dimensionOrder, [0, 1])
    XCTAssertEqual(tensor.count, 4)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
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

  func testInitScalarsUInt8() {
    let data: [UInt8] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init), dataType: .byte)
    XCTAssertEqual(tensor.dataType, .byte)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsInt8() {
    let data: [Int8] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init), dataType: .char)
    XCTAssertEqual(tensor.dataType, .char)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsInt16() {
    let data: [Int16] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init))
    XCTAssertEqual(tensor.dataType, .short)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsInt32() {
    let data: [Int32] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init))
    XCTAssertEqual(tensor.dataType, .int)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsInt64() {
    let data: [Int64] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init))
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsFloat() {
    let data: [Float] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init))
    XCTAssertEqual(tensor.dataType, .float)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsDouble() {
    let data: [Double] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init))
    XCTAssertEqual(tensor.dataType, .double)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsBool() {
    let data: [Bool] = [true, false, true, false, true, false]
    let tensor = Tensor(data.map(NSNumber.init), dataType: .bool)
    XCTAssertEqual(tensor.dataType, .bool)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsUInt16() {
    let data: [UInt16] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init), dataType: .uInt16)
    XCTAssertEqual(tensor.dataType, .uInt16)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsUInt32() {
    let data: [UInt32] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init), dataType: .uInt32)
    XCTAssertEqual(tensor.dataType, .uInt32)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsUInt64() {
    let data: [UInt64] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init), dataType: .uInt64)
    XCTAssertEqual(tensor.dataType, .uInt64)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsInt() {
    let data: [Int] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init), dataType: .long)
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitScalarsUInt() {
    let data: [UInt] = [1, 2, 3, 4, 5, 6]
    let tensor = Tensor(data.map(NSNumber.init), dataType: .uInt64)
    XCTAssertEqual(tensor.dataType, .uInt64)
    XCTAssertEqual(tensor.shape, [6])
    XCTAssertEqual(tensor.strides, [1])
    XCTAssertEqual(tensor.dimensionOrder, [0])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer), data)
    })
  }

  func testInitInt8() {
    let tensor = Tensor(Int8(42))
    XCTAssertEqual(tensor.dataType, .char)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Int8>) in
      XCTAssertEqual(Array(buffer).first, 42)
    })
  }

  func testInitInt16() {
    let tensor = Tensor(Int16(42))
    XCTAssertEqual(tensor.dataType, .short)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Int16>) in
      XCTAssertEqual(Array(buffer).first, 42)
    })
  }

  func testInitInt32() {
    let tensor = Tensor(Int32(42))
    XCTAssertEqual(tensor.dataType, .int)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Int32>) in
      XCTAssertEqual(Array(buffer).first, 42)
    })
  }

  func testInitInt64() {
    let tensor = Tensor(Int64(42))
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Int64>) in
      XCTAssertEqual(Array(buffer).first, 42)
    })
  }

  func testInitUInt8() {
    let tensor = Tensor(UInt8(42))
    XCTAssertEqual(tensor.dataType, .byte)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<UInt8>) in
      XCTAssertEqual(Array(buffer).first, 42)
    })
  }

  func testInitUInt16() {
    let tensor = Tensor(UInt16(42))
    XCTAssertEqual(tensor.dataType, .uInt16)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<UInt16>) in
      XCTAssertEqual(Array(buffer).first, 42)
    })
  }

  func testInitUInt32() {
    let tensor = Tensor(UInt32(42))
    XCTAssertEqual(tensor.dataType, .uInt32)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<UInt32>) in
      XCTAssertEqual(Array(buffer).first, 42)
    })
  }

  func testInitUInt64() {
    let tensor = Tensor(UInt64(42))
    XCTAssertEqual(tensor.dataType, .uInt64)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<UInt64>) in
      XCTAssertEqual(Array(buffer).first, 42)
    })
  }

  func testInitBool() {
    let tensor = Tensor(true)
    XCTAssertEqual(tensor.dataType, .bool)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Bool>) in
      XCTAssertEqual(Array(buffer).first, true)
    })
  }

  func testInitFloat() {
    let tensor = Tensor(Float(42.0))
    XCTAssertEqual(tensor.dataType, .float)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Float>) in
      XCTAssertEqual(Array(buffer).first, 42)
    })
  }

  func testInitDouble() {
    let tensor = Tensor(42.0)
    XCTAssertEqual(tensor.dataType, .double)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Double>) in
      XCTAssertEqual(Array(buffer).first, 42.0)
    })
  }

  func testInitInt() {
    let tensor = Tensor(42)
    XCTAssertEqual(tensor.dataType, .long)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { buffer in
      XCTAssertEqual(Array(buffer).first, 42)
    })
  }

  func testInitUInt() {
    let tensor = Tensor(UInt(42))
    XCTAssertEqual(tensor.dataType, .uInt64)
    XCTAssertEqual(tensor.shape, [])
    XCTAssertEqual(tensor.strides, [])
    XCTAssertEqual(tensor.dimensionOrder, [])
    XCTAssertEqual(tensor.count, 1)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<UInt64>) in
      XCTAssertEqual(Array(buffer).first, 42)
    })
  }

  func testEmpty() {
    let tensor = Tensor.empty(shape: [3, 4], dataType: .float)
    XCTAssertEqual(tensor.shape, [3, 4])
    XCTAssertEqual(tensor.count, 12)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Float>) in
      XCTAssertNotNil(buffer.baseAddress)
      XCTAssertEqual(buffer.count, 12)
      XCTAssertEqual(tensor.dataType, .float)
    })
  }

  func testEmptyLike() {
    let other = Tensor.empty(shape: [2, 2], dataType: .int)
    let tensor = Tensor.empty(like: other)
    XCTAssertEqual(tensor.shape, other.shape)
    XCTAssertEqual(tensor.strides, other.strides)
    XCTAssertEqual(tensor.dimensionOrder, other.dimensionOrder)
    XCTAssertEqual(tensor.dataType, other.dataType)
  }

  func testFull() {
    let tensor = Tensor.full(shape: [2, 2], scalar: 7, dataType: .int)
    XCTAssertEqual(tensor.shape, [2, 2])
    XCTAssertEqual(tensor.count, 4)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Int32>) in
      for value in buffer {
        XCTAssertEqual(value, 7)
      }
    })
  }

  func testFullLike() {
    let other = Tensor.empty(shape: [2, 2], dataType: .int)
    let tensor = Tensor.full(like: other, scalar: 42, dataType: .float)
    XCTAssertEqual(tensor.shape, other.shape)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Float>) in
      for value in buffer {
        XCTAssertEqual(value, 42.0)
      }
    })
  }

  func testOnes() {
    let tensor = Tensor.ones(shape: [2, 3], dataType: .float)
    XCTAssertEqual(tensor.shape, [2, 3])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Float>) in
      for value in buffer {
        XCTAssertEqual(value, 1.0)
      }
    })
  }

  func testOnesLike() {
    let other = Tensor.empty(shape: [2, 4], dataType: .double)
    let tensor = Tensor.ones(like: other)
    XCTAssertEqual(tensor.shape, other.shape)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Double>) in
      for value in buffer {
        XCTAssertEqual(value, 1.0)
      }
    })
  }

  func testZeros() {
    let tensor = Tensor.zeros(shape: [2, 3], dataType: .double)
    XCTAssertEqual(tensor.shape, [2, 3])
    XCTAssertEqual(tensor.count, 6)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Double>) in
      for value in buffer {
        XCTAssertEqual(value, 0)
      }
    })
  }

  func testZerosLike() {
    let other = Tensor.full(shape: [3, 2], scalar: 9, dataType: .int)
    let tensor = Tensor.zeros(like: other)
    XCTAssertEqual(tensor.shape, other.shape)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Int32>) in
      for value in buffer {
        XCTAssertEqual(value, 0)
      }
    })
  }

  func testRandom() {
    let tensor = Tensor.rand(shape: [3, 3], dataType: .float)
    XCTAssertEqual(tensor.shape, [3, 3])
    XCTAssertEqual(tensor.count, 9)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Float>) in
      let uniqueValues = Set(buffer)
      XCTAssertTrue(uniqueValues.count > 1)
    })
  }

  func testRandomLike() {
    let other = Tensor.full(shape: [3, 3], scalar: 9, dataType: .int)
    let tensor = Tensor.rand(like: other)
    XCTAssertEqual(tensor.shape, other.shape)
    XCTAssertEqual(tensor.count, other.count)
  }

  func testRandomNormal() {
    let tensor = Tensor.randn(shape: [4], dataType: .double)
    XCTAssertEqual(tensor.shape, [4])
    XCTAssertEqual(tensor.count, 4)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Double>) in
      XCTAssertEqual(buffer.count, 4)
    })
  }

  func testRandomNormalLike() {
    let other = Tensor.zeros(shape: [4], dataType: .float)
    let tensor = Tensor.randn(like: other)
    XCTAssertEqual(tensor.shape, other.shape)
    XCTAssertEqual(tensor.count, other.count)
  }

  func testRandomInteger() {
    let tensor = Tensor.randint(low: 10, high: 20, shape: [5], dataType: .int)
    XCTAssertEqual(tensor.shape, [5])
    XCTAssertEqual(tensor.count, 5)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Int32>) in
      for value in buffer {
        XCTAssertTrue(value >= 10 && value < 20)
      }
    })
  }

  func testRandomIntegerLike() {
    let other = Tensor.ones(shape: [5], dataType: .int)
    let tensor = Tensor.randint(like: other, low: 100, high: 200)
    XCTAssertNoThrow(try tensor.withUnsafeBytes { (buffer: UnsafeBufferPointer<Int32>) in
      for value in buffer {
        XCTAssertTrue(value >= 100 && value < 200)
      }
    })
  }
}
