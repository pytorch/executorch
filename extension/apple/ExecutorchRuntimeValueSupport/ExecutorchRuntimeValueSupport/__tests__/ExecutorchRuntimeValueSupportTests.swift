/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@testable import ExecutorchRuntimeValueSupport
import XCTest

public extension String {

  /// Returns a random string.
  /// This useful for testing when we want to ensure that production code
  /// accidentally pass a test by using the same value as the test.
  static func random() -> String {
    UUID().uuidString
  }
}

public extension Float {
  static func randomPositive() -> Float {
    .random(in: 1...Float.greatestFiniteMagnitude)
  }
}

class ExecutorchRuntimeValueSupportTests: XCTestCase {

  func testTensorValue() throws {
    let factory = ExecutorchRuntimeValueSupport(),
        size = 100,
        data = (1...size).map { _ in Float.randomPositive() },
        shape = [size]

    let sut = try XCTUnwrap(try? factory.createTensor(value: factory.createFloatTensor(value: data, shape: shape)))

    XCTAssertEqual(try? sut.tensorValue().floatRepresentation().floatArray, data)
    XCTAssertEqual(try? sut.tensorValue().floatRepresentation().shape, shape)
  }

  func testCreateStringsThrows() {
    let factory = ExecutorchRuntimeValueSupport(),
        value: String = .random()

    XCTAssertThrowsError(try factory.createString(value: value))
  }
}
