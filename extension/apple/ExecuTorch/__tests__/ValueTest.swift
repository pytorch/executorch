/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorch
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
    XCTAssertEqual(value.tensor(), tensor)
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

  func testScalarBoolean() {
    let value = Value(true as NSNumber)
    XCTAssertTrue(value.isBoolean)
    XCTAssertEqual(value.boolean, true)

    let value2 = Value(false as NSNumber)
    XCTAssertTrue(value2.isBoolean)
    XCTAssertEqual(value2.boolean, false)
  }

  func testScalarInteger() {
    let value = Value(42 as NSNumber)
    XCTAssertTrue(value.isInteger)
    XCTAssertEqual(value.integer, 42)

    let value2 = Value(Int16(7) as NSNumber)
    XCTAssertTrue(value2.isInteger)
    XCTAssertEqual(value2.integer, 7)

    let value3 = Value(Int32(13) as NSNumber)
    XCTAssertTrue(value3.isInteger)
    XCTAssertEqual(value3.integer, 13)

    let value4 = Value(Int64(64) as NSNumber)
    XCTAssertTrue(value4.isInteger)
    XCTAssertEqual(value4.integer, 64)
  }

  func testScalarDouble() {
    let value = Value(3.14 as NSNumber)
    XCTAssertTrue(value.isDouble)
    XCTAssertEqual(value.double, 3.14)

    let value2 = Value(Float(6.28) as NSNumber)
    XCTAssertTrue(value2.isFloat)
    XCTAssertEqual(value2.float, 6.28)
  }

  func testIsEqual() {
    let noneValue1 = Value()
    let noneValue2 = Value()
    XCTAssertTrue(noneValue1.isEqual(noneValue2))

    let intValue1 = Value(42)
    let intValue2 = Value(42)
    let intValueDifferent = Value(43)
    XCTAssertTrue(intValue1.isEqual(intValue2))
    XCTAssertFalse(intValue1.isEqual(intValueDifferent))

    let boolValue1 = Value(true)
    let boolValue2 = Value(true)
    let boolValueDifferent = Value(false)
    XCTAssertTrue(boolValue1.isEqual(boolValue2))
    XCTAssertFalse(boolValue1.isEqual(boolValueDifferent))

    let doubleValue1 = Value(3.14)
    let doubleValue2 = Value(3.14)
    let doubleValueDifferent = Value(2.71)
    XCTAssertTrue(doubleValue1.isEqual(doubleValue2))
    XCTAssertFalse(doubleValue1.isEqual(doubleValueDifferent))

    let stringValue1 = Value("hello")
    let stringValue2 = Value("hello")
    let stringValueDifferent = Value("world")
    XCTAssertTrue(stringValue1.isEqual(stringValue2))
    XCTAssertFalse(stringValue1.isEqual(stringValueDifferent))

    let tensor1 = Tensor([1.0, 2.0, 3.0])
    let tensor2 = Tensor([1.0, 2.0, 3.0])
    let tensorDifferent = Tensor([3.0, 2.0, 1.0])
    let tensorValue1 = Value(tensor1)
    let tensorValue2 = Value(tensor2)
    let tensorValueDifferent = Value(tensorDifferent)
    XCTAssertTrue(tensorValue1.isEqual(tensorValue2))
    XCTAssertFalse(tensorValue1.isEqual(tensorValueDifferent))
  }
}

class ValueProtocolTest: XCTestCase {
  private func encoded(_ inputs: ValueConvertible...) -> [Value] {
    inputs.map { $0.asValue() }
  }

  func testEncodeDecodeBool() throws {
    let original: Bool = true
    let value = original.asValue()
    XCTAssertTrue(value.isBoolean)
    let decoded: Bool = try Bool.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeInt() throws {
    let original: Int = 123
    let value = original.asValue()
    XCTAssertTrue(value.isInteger)
    let decoded: Int = try Int.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeInt8() throws {
    let original: Int8 = -42
    let value = original.asValue()
    XCTAssertTrue(value.isInteger)
    let decoded: Int8 = try Int8.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeInt16() throws {
    let original: Int16 = 1024
    let value = original.asValue()
    XCTAssertTrue(value.isInteger)
    let decoded: Int16 = try Int16.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeInt32() throws {
    let original: Int32 = -2048
    let value = original.asValue()
    XCTAssertTrue(value.isInteger)
    let decoded: Int32 = try Int32.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeInt64() throws {
    let original: Int64 = 1_000_000_000
    let value = original.asValue()
    XCTAssertTrue(value.isInteger)
    let decoded: Int64 = try Int64.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeUInt8() throws {
    let original: UInt8 = 255
    let value = original.asValue()
    XCTAssertTrue(value.isInteger)
    let decoded: UInt8 = try UInt8.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeUInt16() throws {
    let original: UInt16 = 65_535
    let value = original.asValue()
    XCTAssertTrue(value.isInteger)
    let decoded: UInt16 = try UInt16.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeUInt32() throws {
    let original: UInt32 = 4_294_967_295
    let value = original.asValue()
    XCTAssertTrue(value.isInteger)
    let decoded: UInt32 = try UInt32.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeUInt64() throws {
    let original: UInt64 = 18_446_744_073_709_551_615
    let value = original.asValue()
    XCTAssertTrue(value.isInteger)
    let decoded: UInt64 = try UInt64.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeUInt() throws {
    let original: UInt = 42
    let value = original.asValue()
    XCTAssertTrue(value.isInteger)
    let decoded: UInt = try UInt.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeFloat() throws {
    let original: Float = 3.1415
    let value = original.asValue()
    XCTAssertTrue(value.isFloat)
    let decoded: Float = try Float.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeDouble() throws {
    let original: Double = 2.71828
    let value = original.asValue()
    XCTAssertTrue(value.isDouble)
    let decoded: Double = try Double.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeString() throws {
    let original = "swift"
    let value = original.asValue()
    XCTAssertTrue(value.isString)
    let decoded: String = try String.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testEncodeDecodeNSNumber() throws {
    let original = NSNumber(value: 7.0)
    let value = original.asValue()
    XCTAssertTrue(value.isDouble)
    let decoded: NSNumber = try NSNumber.from(value)
    XCTAssertEqual(decoded, original)
  }

  func testSequenceDecodeSingleInt() throws {
    let values = encoded(99)
    let decoded = try Int.from(values)
    XCTAssertEqual(decoded, 99)
  }

  func testSequenceDecodeSingleBool() throws {
    let values = encoded(false)
    let decoded = try Bool.from(values)
    XCTAssertEqual(decoded, false)
  }

  func testSequenceDecodeMultipleFailure() {
    let values = encoded(1, 2)
    XCTAssertThrowsError(try Int.from(values))
  }

  func testArrayDecodeInts() throws {
    let values = encoded(1, 2, 3, 4)
    let decoded: [Int] = try [Int].from(values)
    XCTAssertEqual(decoded, [1, 2, 3, 4])
  }

  func testArrayDecodeFloats() throws {
    let values = encoded(1.5, 2.5, 3.5)
    let decoded: [Float] = try [Float].from(values)
    XCTAssertEqual(decoded, [1.5, 2.5, 3.5])
  }

  func testArrayDecodeMismatchFailure() {
    let values = encoded(1, "two", 3)
    XCTAssertThrowsError(try [Int].from(values))
  }

  func testArrayDecodeEmpty() throws {
    let values: [Value] = encoded()
    let decoded: [Int] = try [Int].from(values)
    XCTAssertEqual(decoded, [])
  }
}
