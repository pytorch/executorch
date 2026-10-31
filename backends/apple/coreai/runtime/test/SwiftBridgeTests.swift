// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import CoreAI
import Foundation
import Synchronization
import XCTest

private func copiedData(_ output: ETCoreAITensor) -> Data {
  var data = Data(count: Int(output.byteCount))
  data.withUnsafeMutableBytes { output.copyBytes(to: $0.baseAddress) }
  return data
}

@available(macOS 27.0, iOS 27.0, *)
private func copyOutput(_ array: NDArray) throws -> Data {
  copiedData(try prepareOutput(array))
}

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
      completed.fulfill()
    } operation: {
      ProtocolPreparedModel(released: released)
    }
    wait(for: [completed, released], timeout: 5, enforceOrder: true)
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
          XCTAssertEqual(error?.code, ETCoreAIErrorCode.runtime.rawValue)
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
final class SwiftSpecializationTests: XCTestCase {
  func testCachePolicyAllowsOnlyStoragePressure() {
    XCTAssertEqual(ETCoreAISwiftModelLoader.cachePolicy.purgeConditions, [.storagePressure])
  }
  func testDefaultPreservesIdentifier() throws {
    let snapshot = try CoreAISpecialization(options: .default)
    XCTAssertEqual(snapshot.options, .default)
    XCTAssertEqual(snapshot.identifier, "default")
    XCTAssertEqual(ETCoreAISwiftModelLoader().specializationIdentifier, snapshot.identifier)
  }

  func testAllOptionFieldsParticipateInIdentity() throws {
    let choices: [SpecializationOptions] = [
      .default, .cpuOnly,
      SpecializationOptions(preferredComputeUnitKind: .cpu),
      SpecializationOptions(preferredComputeUnitKind: .gpu),
      SpecializationOptions(preferredComputeUnitKind: .neuralEngine),
    ]
    let options =
      choices
      + choices.map { original in
        var changed = original
        changed.expectFrequentReshapes.toggle()
        return changed
      }
    let snapshots = try options.map { try CoreAISpecialization(options: $0) }
    for first in snapshots {
      XCTAssertEqual(first.identifier, try CoreAISpecialization(options: first.options).identifier)
      XCTAssertEqual(
        ETCoreAISwiftModelLoader(specialization: first).specializationIdentifier, first.identifier)
      for second in snapshots {
        XCTAssertEqual(first.options == second.options, first.identifier == second.identifier)
      }
    }
  }

  func testSnapshotDoesNotFollowSubsequentOptionMutation() throws {
    var options = SpecializationOptions.cpuOnly
    let original = options
    let snapshot = try CoreAISpecialization(options: options)
    let loader = ETCoreAISwiftModelLoader(specialization: snapshot)
    options.expectFrequentReshapes.toggle()
    XCTAssertEqual(snapshot.options, original)
    XCTAssertEqual(loader.specializationIdentifier, snapshot.identifier)
    XCTAssertNotEqual(
      loader.specializationIdentifier, try CoreAISpecialization(options: options).identifier)
  }

  func testDescriptorIsVersionedAndUsesStableComputeUnitNames() throws {
    var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
    options.expectFrequentReshapes = !SpecializationOptions.default.expectFrequentReshapes
    let snapshot = try CoreAISpecialization(options: options)
    let json = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(snapshot.identifier.utf8)) as? [String: Any])
    XCTAssertEqual(json["version"] as? Int, 1)
    XCTAssertEqual(json["preferredComputeUnit"] as? String, "gpu")
    XCTAssertEqual(json["expectFrequentReshapes"] as? Bool, options.expectFrequentReshapes)
    let allowed = try XCTUnwrap(json["allowedComputeUnits"] as? [String])
    XCTAssertEqual(allowed, allowed.sorted())
    XCTAssertTrue(Set(allowed).isSubset(of: ["cpu", "gpu", "neuralEngine"]))
    XCTAssertEqual(allowed.count, options.allowedComputeUnitKinds.count)
  }
}

@available(macOS 27.0, iOS 27.0, *)
final class SwiftSessionTests: XCTestCase {
  private struct Manifest: Decodable {
    let function: String
    let inputNames: [String]
    let outputNames: [String]

    enum CodingKeys: String, CodingKey {
      case function
      case inputNames = "input_names"
      case outputNames = "output_names"
    }
  }

  private struct Input: @unchecked Sendable {
    let storage: NSData
    let tensor: ETCoreAIInputTensor

    init(_ data: Data, shape: [Int], type: ETCoreAIScalarType = .float32) {
      storage = data as NSData
      tensor = ETCoreAIInputTensor(
        bytes: storage.bytes, byteCount: UInt(storage.length),
        shape: shape.map(NSNumber.init(value:)), scalarType: type)
    }

    init(_ values: [Float]) {
      self.init(values.withUnsafeBytes { Data($0) }, shape: [values.count])
    }
  }

  private var loader = ETCoreAISwiftModelLoader()
  private var manifest: Manifest!
  private var fixtureDirectory: URL!
  private var model: (any ETCoreAIPreparedModel)?
  private var session: (any ETCoreAISession)?
  private var bookmark: Data?

  override func setUpWithError() throws {
    guard let root = ProcessInfo.processInfo.environment["COREAI_BRIDGE_FIXTURE_DIR"] else {
      throw XCTSkip("COREAI_BRIDGE_FIXTURE_DIR is required for SDK session tests")
    }
    try useFixture(at: URL(fileURLWithPath: root, isDirectory: true))
  }

  private func useFixture(at directory: URL) throws {
    fixtureDirectory = directory
    manifest = try JSONDecoder().decode(
      Manifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
    XCTAssertEqual(manifest.inputNames.count, 2)
    XCTAssertEqual(manifest.outputNames.count, 2)
  }

  func testSpecializationUsesTheSnapshotOptions() throws {
    let snapshot = try CoreAISpecialization(options: .cpuOnly)
    loader = ETCoreAISwiftModelLoader(specialization: snapshot)
    try prepareModel()
    XCTAssertNotNil(
      try AIModelCache.default.model(
        for: fixtureDirectory.appendingPathComponent("model.aimodel"), options: snapshot.options))
  }

  private func prepareModel() throws {
    guard model == nil else { return }
    let completed = expectation(description: "specialized session fixture")
    completed.assertForOverFulfill = true
    let result = Mutex<(model: (any ETCoreAIPreparedModel)?, error: Error?)>((nil, nil))
    loader.specializeModel(at: fixtureDirectory.appendingPathComponent("model.aimodel")) {
      status, model, error in
      XCTAssertEqual(status, .hit)
      result.withLock { $0 = (model, error) }
      completed.fulfill()
    }
    wait(for: [completed], timeout: 30)
    let value = result.withLock { value in
      let snapshot = value
      value = (nil, nil)
      return snapshot
    }
    if let failure = value.error { throw failure }
    model = try XCTUnwrap(value.model)
    bookmark = model?.copyBookmarkData()
  }

  override func tearDownWithError() throws {
    session = nil
    model = nil
    if let bookmark {
      let completed = expectation(description: "evicted session fixture")
      completed.assertForOverFulfill = true
      loader.evictModel(withBookmark: bookmark) { error in
        XCTAssertNil(error)
        completed.fulfill()
      }
      wait(for: [completed], timeout: 10)
    }
    bookmark = nil
  }

  private func bind(inputNames: [String]? = nil, outputNames: [String]? = nil) throws {
    try prepareModel()
    let result = Mutex<(session: (any ETCoreAISession)?, error: Error?, calls: Int)>((nil, nil, 0))
    try XCTUnwrap(model).loadFunctionNamed(
      manifest.function, inputNames: inputNames ?? manifest.inputNames,
      outputNames: outputNames ?? manifest.outputNames
    ) { session, error in
      result.withLock { $0 = (session, error, $0.calls + 1) }
    }
    let value = result.withLock { value in
      let snapshot = value
      value = (nil, nil, 0)
      return snapshot
    }
    XCTAssertEqual(value.calls, 1)
    if let failure = value.error { throw failure }
    session = try XCTUnwrap(value.session)
  }

  private func execute(_ inputs: [Input]) throws -> [ETCoreAITensor] {
    let completed = expectation(description: "executed session fixture")
    completed.assertForOverFulfill = true
    let result = Mutex<(outputs: [ETCoreAITensor]?, error: Error?)>((nil, nil))
    try XCTUnwrap(session).executeInputs(inputs.map(\.tensor)) { tensors, error in
      let owned = tensors?.map {
        ETCoreAITensor(data: copiedData($0), shape: $0.shape, scalarType: $0.scalarType)
      }
      result.withLock { $0 = (owned, error) }
      // Retain the borrowed allocations even if the XCTest wait times out.
      withExtendedLifetime(inputs) { completed.fulfill() }
    }
    wait(for: [completed], timeout: 10)
    let value = result.withLock { value in
      let snapshot = value
      value = (nil, nil)
      return snapshot
    }
    if let failure = value.error { throw failure }
    return try XCTUnwrap(value.outputs)
  }

  func testExecutesWithOrderedAndReversedBindings() throws {
    for reversed in [false, true] {
      try bind(
        inputNames: reversed ? Array(manifest.inputNames.reversed()) : manifest.inputNames,
        outputNames: reversed ? Array(manifest.outputNames.reversed()) : manifest.outputNames)
      let inputs = [Input([5, 9]), Input([1, 2])]
      let outputs = try execute(reversed ? Array(inputs.reversed()) : inputs)
      let expected: [[Float]] = reversed ? [[4, 7], [6, 11]] : [[6, 11], [4, 7]]
      XCTAssertEqual(outputs.count, expected.count)
      for (output, values) in zip(outputs, expected) {
        XCTAssertEqual(output.scalarType, .float32)
        XCTAssertEqual(output.shape, [2])
        XCTAssertEqual(copiedData(output), values.withUnsafeBytes { Data($0) })
      }
    }
  }

  func testDynamicDescriptorsTrackRuntimeShapes() throws {
    try useFixture(at: fixtureDirectory.appendingPathComponent("dynamic"))
    try bind()
    let sdkModel = try XCTUnwrap(AIModel(resolvingBookmark: XCTUnwrap(bookmark)))
    let function = try XCTUnwrap(sdkModel.loadFunction(named: manifest.function))
    for name in manifest.inputNames {
      guard case .ndArray(let descriptor) = function.descriptor.inputDescriptor(of: name) else {
        return XCTFail("Expected an NDArray input descriptor")
      }
      XCTAssertEqual(descriptor.shape, [-1])
    }
    for count in [2, 4, 1] {
      let left = (0..<count).map { Float($0 + 2) }
      let right = [Float](repeating: 1, count: count)
      let outputs = try execute([Input(left), Input(right)])
      let expected = [left.map { $0 + 1 }, left.map { $0 - 1 }]
      XCTAssertEqual(outputs.count, 2)
      for (output, values) in zip(outputs, expected) {
        XCTAssertEqual(output.shape, [NSNumber(value: count)])
        XCTAssertEqual(copiedData(output), values.withUnsafeBytes { Data($0) })
      }
    }
  }

  func testRejectsDuplicateAndMismatchedBindings() throws {
    let inputs = manifest.inputNames
    let outputs = manifest.outputNames
    for (inputNames, outputNames) in [
      ([inputs[0], inputs[0]], outputs),
      (inputs, [outputs[0], outputs[0]]),
      ([inputs[0], "missing"], outputs),
      (inputs, [outputs[0], "missing"]),
    ] {
      XCTAssertThrowsError(try bind(inputNames: inputNames, outputNames: outputNames)) {
        XCTAssertEqual(($0 as NSError).domain, ETCoreAIErrorDomain)
        XCTAssertEqual(($0 as NSError).code, ETCoreAIErrorCode.invalidModel.rawValue)
      }
    }
  }

  func testInvalidInputCountCompletesSynchronouslyOnce() throws {
    try bind()
    let calls = Mutex(0)
    let caller = ObjectIdentifier(Thread.current)
    try XCTUnwrap(session).executeInputs([]) { outputs, error in
      calls.withLock { $0 += 1 }
      XCTAssertEqual(ObjectIdentifier(Thread.current), caller)
      XCTAssertNil(outputs)
      XCTAssertEqual((error as NSError?)?.domain, ETCoreAIErrorDomain)
      XCTAssertEqual((error as NSError?)?.code, ETCoreAIErrorCode.invalidArgument.rawValue)
    }
    XCTAssertEqual(calls.withLock { $0 }, 1)
  }

  func testInvalidPayloadCompletesSynchronouslyOnce() throws {
    try bind()
    let invalid = ETCoreAIInputTensor(
      bytes: nil, byteCount: 8, shape: [2], scalarType: .float32)
    let calls = Mutex(0)
    let caller = ObjectIdentifier(Thread.current)
    try XCTUnwrap(session).executeInputs([invalid, invalid]) { outputs, error in
      calls.withLock { $0 += 1 }
      XCTAssertEqual(ObjectIdentifier(Thread.current), caller)
      XCTAssertNil(outputs)
      XCTAssertEqual((error as NSError?)?.code, ETCoreAIErrorCode.invalidArgument.rawValue)
    }
    XCTAssertEqual(calls.withLock { $0 }, 1)
  }

  func testRejectsInputDescriptorDtypeAndShapeMismatches() throws {
    try bind()
    for input in [
      Input(Data(count: 4), shape: [2], type: .float16),
      Input(Data(count: 8), shape: [1, 2]),
      Input(Data(count: 12), shape: [3]),
    ] {
      XCTAssertThrowsError(try execute([input, Input([1, 2])])) {
        XCTAssertEqual(($0 as NSError).domain, ETCoreAIErrorDomain)
        XCTAssertEqual(($0 as NSError).code, ETCoreAIErrorCode.invalidArgument.rawValue)
      }
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

  func testScalarDefinitionsRoundTripAndRejectUnsupportedTypes() throws {
    let supported = CoreAIScalarInfo.supported
    XCTAssertEqual(Set(supported.map(\.sdkType)).count, supported.count)
    XCTAssertEqual(Set(supported.map { $0.bridgeType.rawValue }).count, supported.count)
    for info in supported {
      XCTAssertGreaterThan(info.byteWidth, 0)
      XCTAssertEqual(try CoreAIScalarInfo.from(sdkType: info.sdkType).bridgeType, info.bridgeType)
      XCTAssertEqual(try CoreAIScalarInfo.from(bridgeType: info.bridgeType).sdkType, info.sdkType)
      XCTAssertEqual(
        try CoreAITensorPayload.byteCount(shape: [2, 3], scalarType: info.bridgeType),
        6 * info.byteWidth)
    }
    let supportedTypes = Set(supported.map(\.sdkType))
    for type in NDArray.ScalarType.allCases where !supportedTypes.contains(type) {
      XCTAssertThrowsError(try CoreAIScalarInfo.from(sdkType: type)) {
        XCTAssertEqual(($0 as NSError).code, ETCoreAIErrorCode.unsupported.rawValue)
      }
    }
    let unsupported = try XCTUnwrap(ETCoreAIScalarType(rawValue: Int.max))
    XCTAssertThrowsError(try CoreAITensorPayload.byteCount(shape: [1], scalarType: unsupported)) {
      XCTAssertEqual(($0 as NSError).code, ETCoreAIErrorCode.unsupported.rawValue)
    }
    XCTAssertThrowsError(try prepareOutput(NDArray(shape: [2], scalarType: .int32))) {
      XCTAssertEqual(($0 as NSError).code, ETCoreAIErrorCode.unsupported.rawValue)
    }
  }

  func testRawOutputCopiesPreserveBitsForEverySupportedTypeAndLayout() throws {
    for info in CoreAIScalarInfo.supported {
      for (shape, strides) in [
        ([2, 3], [3, 1]), ([2, 3], [4, 1]), ([2, 3], [1, 2]),
        ([2, 1, 3], [3, 19, 1]), ([], []), ([2, 0, 3], [3, 3, 1]),
      ] {
        let count = shape.reduce(1, *)
        let expected = (0..<(count * info.byteWidth)).map {
          UInt8(truncatingIfNeeded: $0 * 73 + 255)
        }
        var array = NDArray(shape: shape, scalarType: info.sdkType, strides: strides)
        do {
          var view = array.mutableRawView()
          var bytes = view.mutableBytes
          bytes.withUnsafeMutableBytes { storage in
            for index in 0..<count {
              var remainder = index
              var offset = 0
              for axis in shape.indices.reversed() {
                offset += (remainder % shape[axis]) * strides[axis]
                remainder /= shape[axis]
              }
              for byte in 0..<info.byteWidth {
                storage[offset * info.byteWidth + byte] = expected[index * info.byteWidth + byte]
              }
            }
          }
        }
        let output = try prepareOutput(array)
        XCTAssertEqual(output.scalarType, info.bridgeType)
        XCTAssertEqual(output.shape, shape.map(NSNumber.init(value:)))
        XCTAssertEqual(output.byteCount, UInt(expected.count))
        var destination = [UInt8](repeating: 0xa5, count: expected.count + 2)
        destination.withUnsafeMutableBytes { bytes in
          output.copyBytes(to: bytes.baseAddress!.advanced(by: 1))
        }
        XCTAssertEqual(destination, [0xa5] + expected + [0xa5])
      }
    }
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
      assertInvalidInput(
        bytes: bytes.baseAddress, byteCount: 4, shape: [0, NSNumber(value: Int.max)])
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
    let expectedType = try CoreAIScalarInfo.from(bridgeType: scalarType).sdkType
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

  func testPreparedOutputsRetainSDKStorageAndCopyIntoCallerSubranges() throws {
    try assertOwnedOutput([Float(3), 7], scalarType: .float32)
    try assertOwnedOutput([Float16(3), 7], scalarType: .float16)
  }

  private func assertOwnedOutput<T: BinaryFloatingPoint & BitwiseCopyable>(
    _ values: [T], scalarType: NDArray.ScalarType
  ) throws {
    let output: ETCoreAITensor = try {
      var array = NDArray(shape: [values.count], scalarType: scalarType)
      do {
        var view = array.mutableView(as: T.self)
        view.withUnsafeMutablePointer { pointer, _, _ in
          for index in values.indices { pointer[index] = values[index] }
        }
      }
      return try prepareOutput(array)
    }()
    XCTAssertEqual(output.byteCount, UInt(values.count * MemoryLayout<T>.stride))
    for _ in 0..<2 {
      var destination = [T](repeating: 99, count: values.count + 2)
      destination.withUnsafeMutableBytes { bytes in
        output.copyBytes(to: bytes.baseAddress!.advanced(by: MemoryLayout<T>.stride))
      }
      XCTAssertEqual(destination, [T(99)] + values + [T(99)])
    }
  }

  func testEmptyOutputDoesNotInvokeCopier() {
    var calls = 0
    let output = ETCoreAITensor(byteCount: 0, shape: [0], scalarType: .float32) { _ in
      calls += 1
    }
    output.copyBytes(to: nil)
    XCTAssertEqual(calls, 0)
  }

  func testOutputCopierIsDeferredUntilCopyAndRetainsItsOwner() {
    final class Owner { var calls = 0 }
    weak var observed: Owner?
    var output: ETCoreAITensor? = {
      let owner = Owner()
      observed = owner
      return ETCoreAITensor(byteCount: 4, shape: [1], scalarType: .float32) { destination in
        owner.calls += 1
        var value: Float = 7
        withUnsafeBytes(of: &value) { source in
          destination.copyMemory(from: source.baseAddress!, byteCount: source.count)
        }
      }
    }()
    XCTAssertEqual(observed?.calls, 0)
    XCTAssertEqual(copiedData(output!), [Float(7)].withUnsafeBytes { Data($0) })
    XCTAssertEqual(observed?.calls, 1)
    output = nil
    XCTAssertNil(observed)
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
      let output = try copyOutput(array)
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
          shape: [2, 3], strides: strides, width: 4, storageByteCount: 24)
      ) {
        let expected: ETCoreAIErrorCode = strides[0] < 0 ? .unsupported : .runtime
        XCTAssertEqual(($0 as NSError).code, expected.rawValue)
      }
    }
    XCTAssertThrowsError(
      try CoreAITensorLayout(
        shape: [Int.max, 2], strides: [2, 1], width: 4, storageByteCount: Int.max)
    ) {
      XCTAssertEqual(($0 as NSError).code, ETCoreAIErrorCode.runtime.rawValue)
    }
    XCTAssertThrowsError(
      try CoreAITensorPayload.byteCount(
        shape: [Int.max, 2], scalarType: .float32, errorCode: .runtime)
    ) {
      XCTAssertEqual(($0 as NSError).code, ETCoreAIErrorCode.runtime.rawValue)
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
    let output = try copyOutput(strided)
    XCTAssertEqual(output, bytes)
    do {
      var view = strided.mutableView(as: Float.self)
      view.withUnsafeMutablePointer { pointer, _, _ in pointer[0] = 99 }
    }
    XCTAssertEqual(output, bytes, "the copy must not alias the SDK array")

    var transposed = NDArray(shape: [2, 3], scalarType: .float16, strides: [1, 2])
    do {
      var view = transposed.mutableView(as: Float16.self)
      view.withUnsafeMutablePointer { pointer, _, _ in
        for index in 0..<6 { pointer[index] = Float16(index + 1) }
      }
    }
    let expected: [Float16] = [1, 3, 5, 2, 4, 6]
    XCTAssertEqual(
      try copyOutput(transposed), expected.withUnsafeBytes { Data($0) })
  }

  func testRejectsInterleavedOutput() {
    let array = NDArray(
      shape: [2, 4], scalarType: .float16, strides: [8, 2],
      interleaveLayout: NDArray.InterleaveLayout(dimension: 1, factor: 2))
    XCTAssertThrowsError(try copyOutput(array)) {
      XCTAssertEqual(($0 as NSError).code, ETCoreAIErrorCode.unsupported.rawValue)
    }
  }
}
