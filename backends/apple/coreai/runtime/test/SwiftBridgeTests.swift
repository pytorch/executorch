// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import CoreAI
import Foundation
import XCTest

private final class ProtocolPreparedModel: NSObject, ETCoreAIPreparedModel, Sendable {
  let released: XCTestExpectation

  init(released: XCTestExpectation) {
    self.released = released
    super.init()
  }

  deinit { released.fulfill() }

  func copyBookmarkData() -> Data { Data([1, 2, 3]) }

  func loadFunctionNamed(
    _ functionName: String, inputNames: [String], outputNames: [String],
    completion: @escaping ((any ETCoreAISession)?, Error?) -> Void
  ) {
    completion(
      nil, NSError(domain: ETCoreAIErrorDomain, code: ETCoreAIErrorCode.invalidModel.rawValue))
  }
}

// These contract tests never invoke the SDK; the linked test bundle still requires OS 27.
final class SwiftAcquisitionTests: XCTestCase {
  func testHitPinsPreparedModelThroughCompletion() {
    let completed = expectation(description: "acquired")
    let released = expectation(description: "prepared model released")
    completed.assertForOverFulfill = true
    released.assertForOverFulfill = true
    acquireCoreAIModel(context: "test hit") { status, model, error in
      XCTAssertFalse(Thread.isMainThread)
      XCTAssertEqual(status, .hit)
      XCTAssertNil(error)
      XCTAssertEqual(model?.copyBookmarkData(), Data([1, 2, 3]))
      model?.loadFunctionNamed("missing", inputNames: [], outputNames: []) { session, error in
        XCTAssertNil(session)
        XCTAssertEqual((error as NSError?)?.code, ETCoreAIErrorCode.invalidModel.rawValue)
      }
      completed.fulfill()
    } operation: {
      ProtocolPreparedModel(released: released)
    }
    wait(for: [completed, released], timeout: 5)
  }

  func testOnlyNilIsMiss() {
    let completed = expectation(description: "miss")
    completed.assertForOverFulfill = true
    acquireCoreAIModel(context: "test miss") { status, model, error in
      XCTAssertFalse(Thread.isMainThread)
      XCTAssertEqual(status, .miss)
      XCTAssertNil(model)
      XCTAssertNil(error)
      completed.fulfill()
    } operation: {
      nil
    }
    wait(for: [completed], timeout: 5)
  }

  func testErrorsAreWrappedOnceAndPreserveTheUnderlyingError() {
    let sdk = NSError(domain: "SDK.test", code: 91, userInfo: ["detail": "opaque"])
    let bridge = NSError(
      domain: ETCoreAIErrorDomain, code: ETCoreAIErrorCode.invalidArgument.rawValue)
    for underlying in [sdk, bridge] {
      let completed = expectation(description: underlying.domain)
      completed.assertForOverFulfill = true
      acquireCoreAIModel(context: "test restoration") { status, model, error in
        XCTAssertFalse(Thread.isMainThread)
        XCTAssertEqual(status, .error)
        XCTAssertNil(model)
        let error = error as NSError?
        if underlying === bridge {
          XCTAssertTrue(error === bridge)
        } else {
          XCTAssertEqual(error?.domain, ETCoreAIErrorDomain)
          XCTAssertEqual(error?.code, ETCoreAIErrorCode.invalidModel.rawValue)
          XCTAssertTrue(error?.localizedDescription.hasPrefix("test restoration:") == true)
          XCTAssertTrue((error?.userInfo[NSUnderlyingErrorKey] as? NSError) === sdk)
        }
        completed.fulfill()
      } operation: {
        throw underlying
      }
      wait(for: [completed], timeout: 5)
    }
  }
}

@available(macOS 27.0, iOS 27.0, *)
final class SwiftBridgeTests: XCTestCase {
  func testInvalidSpecializationURLCompletesOffMainThread() {
    let completed = expectation(description: "invalid specialization URL")
    completed.assertForOverFulfill = true
    ETCoreAISwiftModelLoader().specializeModel(at: URL(string: "coreai-test:model")!) {
      status, model, error in
      XCTAssertFalse(Thread.isMainThread)
      XCTAssertEqual(status, .error)
      XCTAssertNil(model)
      XCTAssertEqual((error as NSError?)?.domain, ETCoreAIErrorDomain)
      XCTAssertEqual((error as NSError?)?.code, ETCoreAIErrorCode.invalidArgument.rawValue)
      completed.fulfill()
    }
    wait(for: [completed], timeout: 5)
  }

  func testByteCounts() throws {
    XCTAssertEqual(try CoreAITensorPayload.byteCount(shape: [], scalarType: .float32), 4)
    XCTAssertEqual(try CoreAITensorPayload.byteCount(shape: [2, 3], scalarType: .float16), 12)
    XCTAssertEqual(try CoreAITensorPayload.byteCount(shape: [2, 0, 3], scalarType: .float32), 0)
    for shape in [[-1], [Int.max / 4, 8]] {
      XCTAssertThrowsError(try CoreAITensorPayload.byteCount(shape: shape, scalarType: .float32)) {
        XCTAssertEqual(($0 as NSError).domain, ETCoreAIErrorDomain)
        XCTAssertEqual(($0 as NSError).code, ETCoreAIErrorCode.invalidArgument.rawValue)
      }
    }
  }

  func testInvalidPayloads() {
    var value: Float = 1
    withUnsafeBytes(of: &value) { bytes in
      assertInvalidInput(bytes: bytes.baseAddress, byteCount: 4, shape: [-1])
      assertInvalidInput(bytes: bytes.baseAddress, byteCount: 4, shape: [0, NSNumber(value: Int.max)])
      assertInvalidInput(bytes: bytes.baseAddress, byteCount: 3, shape: [1])
      assertInvalidInput(bytes: nil, byteCount: 4, shape: [1])
    }
  }

  private func assertInvalidInput(
    bytes: UnsafeRawPointer?, byteCount: UInt, shape: [NSNumber],
    file: StaticString = #filePath, line: UInt = #line
  ) {
    XCTAssertThrowsError(
      try CoreAITensorPayload(
        ETCoreAIInputTensor(
          bytes: bytes, byteCount: byteCount, shape: shape, scalarType: .float32)),
      file: file, line: line
    ) {
      XCTAssertEqual(($0 as NSError).domain, ETCoreAIErrorDomain, file: file, line: line)
      XCTAssertEqual(
        ($0 as NSError).code, ETCoreAIErrorCode.invalidArgument.rawValue, file: file, line: line)
    }
  }

  func testInputsBorrowStorage() throws {
    try assertBorrows(Array((0..<12).map(Float.init)), scalarType: .float32)
    try assertBorrows(Array((0..<12).map(Float16.init)), scalarType: .float16)
  }

  private func assertBorrows<T>(_ values: [T], scalarType: ETCoreAIScalarType) throws {
    try values.withUnsafeBytes { storage in
      try assertBorrowedInput(
        UnsafeRawBufferPointer(start: storage.baseAddress, count: 6 * MemoryLayout<T>.stride),
        shape: [2, 3], scalarType: scalarType)
      try assertBorrowedInput(
        UnsafeRawBufferPointer(start: storage.baseAddress, count: MemoryLayout<T>.stride),
        shape: [], scalarType: scalarType)
      try assertBorrowedInput(
        UnsafeRawBufferPointer(start: nil, count: 0), shape: [2, 0, 3], scalarType: scalarType)
      // A dynamic-shape subrange starts mid-allocation and tracks the current rows.
      for rows in [4, 0] {
        try assertBorrowedInput(
          UnsafeRawBufferPointer(
            start: storage.baseAddress!.advanced(by: 2 * MemoryLayout<T>.stride),
            count: rows * 2 * MemoryLayout<T>.stride),
          shape: [rows, 2], scalarType: scalarType)
      }
    }
  }

  private func assertBorrowedInput(
    _ source: UnsafeRawBufferPointer, shape: [Int], scalarType: ETCoreAIScalarType,
    file: StaticString = #filePath, line: UInt = #line
  ) throws {
    let payload = try CoreAITensorPayload(
      ETCoreAIInputTensor(
        bytes: source.baseAddress, byteCount: UInt(source.count),
        shape: shape.map(NSNumber.init(value:)), scalarType: scalarType))
    XCTAssertEqual(payload.bytes, source.baseAddress, file: file, line: line)
    XCTAssertEqual(payload.byteCount, source.count, file: file, line: line)
    XCTAssertEqual(payload.shape, shape, file: file, line: line)
    XCTAssertEqual(payload.scalarType, scalarType, file: file, line: line)
    // The source's withUnsafeBytes scope is still active for every view access.
    let bytes = RawSpan(
      _unsafeBytes: UnsafeRawBufferPointer(
        start: payload.bytes, count: payload.byteCount))
    let expectedType: NDArray.ScalarType = scalarType == .float16 ? .float16 : .float32
    let view = NDArray.RawView(
      bytes: bytes, scalarType: expectedType, shape: payload.shape)
    view.bytes.withUnsafeBytes { borrowed in
      XCTAssertEqual(borrowed.baseAddress, source.baseAddress, file: file, line: line)
      XCTAssertEqual(borrowed.count, source.count, file: file, line: line)
    }
    let viewShape = view.shape
    XCTAssertEqual(viewShape.count, shape.count, file: file, line: line)
    for axis in shape.indices {
      XCTAssertEqual(viewShape[axis], shape[axis], file: file, line: line)
    }
    XCTAssertEqual(view.scalarType, expectedType, file: file, line: line)
  }

  func testOutputCopiesTrackActualShape() throws {
    for rows in [2, 0] {
      let values = (0..<(rows * 2)).map(Float.init)
      var array = NDArray(shape: [rows, 2], scalarType: .float32, strides: [3, 1])
      do {
        var view = array.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { pointer, _, strides in
          for row in 0..<rows {
            for column in 0..<2 {
              pointer[row * strides[0] + column * strides[1]] = values[row * 2 + column]
            }
          }
        }
      }
      let output = try copyOutput(array, as: Float.self)
      XCTAssertEqual(output, values.withUnsafeBytes { Data($0) })
    }
  }

  func testStridedLayout() throws {
    let padded = try CoreAITensorLayout(
      shape: [2, 3], strides: [4, 1], width: 4, storageByteCount: 28)
    XCTAssertEqual((0..<padded.count).map(padded.offset(for:)), [0, 1, 2, 4, 5, 6])
    XCTAssertFalse(padded.isDense)
    let transposed = try CoreAITensorLayout(
      shape: [2, 3], strides: [1, 2], width: 2, storageByteCount: 12)
    XCTAssertEqual((0..<transposed.count).map(transposed.offset(for:)), [0, 2, 4, 1, 3, 5])
    XCTAssertFalse(transposed.isDense)
    let dense = try CoreAITensorLayout(
      shape: [2, 1, 3], strides: [3, 7, 1], width: 4, storageByteCount: 24)
    XCTAssertTrue(dense.isDense)
    let scalar = try CoreAITensorLayout(shape: [], strides: [], width: 4, storageByteCount: 4)
    XCTAssertEqual(scalar.count, 1)
    XCTAssertEqual(scalar.offset(for: 0), 0)
    XCTAssertTrue(scalar.isDense)
  }

  func testInvalidLayouts() {
    for strides in [[-1, 1], [4, 1]] {
      XCTAssertThrowsError(
        try CoreAITensorLayout(
          shape: [2, 3], strides: strides, width: 4, storageByteCount: 24))
    }
  }

  func testOutputCopiesStridedAndTransposedLayouts() throws {
    let values: [Float] = [1, 2, 3, 4, 5, 6]
    let bytes = values.withUnsafeBytes { Data($0) }
    var strided = NDArray(shape: [2, 3], scalarType: .float32, strides: [4, 1])
    do {
      var view = strided.mutableView(as: Float.self)
      view.withUnsafeMutablePointer { pointer, _, strides in
        for row in 0..<2 {
          for column in 0..<3 {
            pointer[row * strides[0] + column * strides[1]] = values[row * 3 + column]
          }
        }
      }
    }
    let output = try copyOutput(strided, as: Float.self)
    XCTAssertEqual(output, bytes)
    do {
      var view = strided.mutableView(as: Float.self)
      view.withUnsafeMutablePointer { pointer, _, _ in pointer[0] = 99 }
    }
    XCTAssertEqual(output, bytes, "the copy must not alias the SDK array")
    XCTAssertThrowsError(try copyOutput(strided, as: Float16.self))

    var transposed = NDArray(shape: [2, 3], scalarType: .float16, strides: [1, 2])
    do {
      var view = transposed.mutableView(as: Float16.self)
      view.withUnsafeMutablePointer { pointer, _, _ in
        for index in 0..<6 { pointer[index] = Float16(index + 1) }
      }
    }
    let expected: [Float16] = [1, 3, 5, 2, 4, 6]
    XCTAssertEqual(try copyOutput(transposed, as: Float16.self), expected.withUnsafeBytes { Data($0) })
  }

  func testRejectsInterleavedOutput() {
    let array = NDArray(
      shape: [2, 4], scalarType: .float16, strides: [8, 2],
      interleaveLayout: NDArray.InterleaveLayout(dimension: 1, factor: 2))
    XCTAssertThrowsError(try copyOutput(array, as: Float16.self)) {
      XCTAssertEqual(($0 as NSError).code, ETCoreAIErrorCode.unsupported.rawValue)
    }
  }
}
