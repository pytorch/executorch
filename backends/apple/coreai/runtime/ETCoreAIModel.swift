// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import CoreAI
import Foundation

private func bridgeError(
  _ code: ETCoreAIErrorCode, _ message: String, underlying: Error? = nil
) -> NSError {
  var info: [String: Any] = [NSLocalizedDescriptionKey: message]
  if let underlying { info[NSUnderlyingErrorKey] = underlying }
  return NSError(domain: ETCoreAIErrorDomain, code: code.rawValue, userInfo: info)
}

private func mappedError(_ error: Error, code: ETCoreAIErrorCode, context: String) -> NSError {
  let error = error as NSError
  return error.domain == ETCoreAIErrorDomain
    ? error : bridgeError(code, "\(context): \(error.localizedDescription)", underlying: error)
}

// Objective-C blocks have no Sendable annotation. Each box is invoked by one detached task only.
private final class Completion<Callback>: @unchecked Sendable {
  let call: Callback

  init(_ call: Callback) {
    self.call = call
  }
}

// Only a successful SDK nil is a miss; thrown errors retain their underlying cause.
func acquireCoreAIModel(
  context: String, completion: @escaping ETCoreAIAcquisitionCompletion,
  operation: @escaping @Sendable () async throws -> (any ETCoreAIPreparedModel)?
) {
  let callback = Completion(completion)
  Task.detached {
    do {
      if let model = try await operation() {
        callback.call(.hit, model, nil)
      } else {
        callback.call(.miss, nil, nil)
      }
    } catch {
      callback.call(.error, nil, mappedError(error, code: .runtime, context: context))
    }
  }
}

@available(macOS 27.0, iOS 27.0, *)
struct CoreAIScalarInfo: Sendable {
  let sdkType: NDArray.ScalarType
  let bridgeType: ETCoreAIScalarType
  let byteWidth: Int

  // Only byte-addressable scalar types belong here; packed types need a separate layout policy.
  static let supported: [Self] = [
    Self(sdkType: .float16, bridgeType: .float16, byteWidth: 2),
    Self(sdkType: .float32, bridgeType: .float32, byteWidth: 4),
  ]

  static func from(sdkType: NDArray.ScalarType) throws -> Self {
    guard let info = supported.first(where: { $0.sdkType == sdkType }) else {
      throw bridgeError(.unsupported, "Unsupported Core AI scalar type: \(sdkType)")
    }
    return info
  }

  static func from(bridgeType: ETCoreAIScalarType) throws -> Self {
    guard let info = supported.first(where: { $0.bridgeType == bridgeType }) else {
      throw bridgeError(.unsupported, "Unsupported bridge scalar type: \(bridgeType.rawValue)")
    }
    return info
  }

  func byteCount(shape: [Int], errorCode: ETCoreAIErrorCode = .invalidArgument) throws -> Int {
    guard shape.allSatisfy({ $0 >= 0 }) else {
      throw bridgeError(errorCode, "Negative tensor dimension")
    }
    // Check suffix products too: SDK dense-stride construction must not overflow on empty tensors.
    var count = byteWidth
    for dimension in shape.reversed() {
      let (next, overflow) = count.multipliedReportingOverflow(by: max(dimension, 1))
      guard !overflow else {
        throw bridgeError(errorCode, "Tensor byte count overflows Int")
      }
      count = next
    }
    return shape.contains(0) ? 0 : count
  }
}

// ExecuTorch waits synchronously and keeps input storage live through output copying.
// This snapshot transfers only metadata and a pointer to the task; it does not own storage.
@available(macOS 27.0, iOS 27.0, *)
struct CoreAITensorPayload: @unchecked Sendable {
  let bytes: UnsafeRawPointer?
  let byteCount: Int
  let shape: [Int]
  let scalarType: ETCoreAIScalarType

  init(_ tensor: ETCoreAIInputTensor) throws {
    shape = try tensor.shape.map { number in
      guard let dimension = number as? Int, dimension >= 0 else {
        throw bridgeError(.invalidArgument, "Tensor dimensions must be nonnegative integers")
      }
      return dimension
    }
    scalarType = tensor.scalarType
    byteCount = try Self.byteCount(shape: shape, scalarType: scalarType)
    guard tensor.byteCount == byteCount else {
      throw bridgeError(
        .invalidArgument, "Tensor payload byte count does not match shape and dtype")
    }
    bytes = tensor.bytes
    guard byteCount == 0 || bytes != nil else {
      throw bridgeError(.invalidArgument, "Nonempty tensor has null data")
    }
  }

  static func byteCount(
    shape: [Int], scalarType: ETCoreAIScalarType,
    errorCode: ETCoreAIErrorCode = .invalidArgument
  ) throws -> Int {
    try CoreAIScalarInfo.from(bridgeType: scalarType).byteCount(shape: shape, errorCode: errorCode)
  }
}

@available(macOS 27.0, iOS 27.0, *)
private func tensorDescriptor(_ value: InferenceValue.Descriptor?) throws -> NDArrayDescriptor {
  guard let value else { throw bridgeError(.invalidModel, "Missing tensor descriptor") }
  guard case .ndArray(let descriptor) = value else {
    throw bridgeError(.unsupported, "Image values are not supported")
  }
  _ = try CoreAIScalarInfo.from(sdkType: descriptor.scalarType)
  guard descriptor.interleaveLayout == nil else {
    throw bridgeError(.unsupported, "Interleaved tensor descriptors are not supported")
  }
  guard descriptor.shape.allSatisfy({ $0 >= -1 }) else {
    throw bridgeError(.invalidModel, "Invalid descriptor dimension")
  }
  return descriptor
}

@available(macOS 27.0, iOS 27.0, *)
private func validateShape(
  _ shape: [Int], descriptor: NDArrayDescriptor,
  errorCode: ETCoreAIErrorCode = .invalidArgument
) throws {
  guard shape.count == descriptor.shape.count,
    zip(shape, descriptor.shape).allSatisfy({ actual, expected in
      actual >= 0 && (expected == -1 || actual == expected)
    })
  else {
    throw bridgeError(errorCode, "Tensor shape does not match the function descriptor")
  }
}

// Offsets are in elements, including padded or transposed SDK layouts.
struct CoreAITensorLayout {
  let shape: [Int]
  let strides: [Int]
  let count: Int
  // Row-major contiguous, ignoring strides of size-1 dimensions.
  let isDense: Bool

  init(shape: [Int], strides: [Int], width: Int, storageByteCount: Int) throws {
    guard shape.count == strides.count, shape.allSatisfy({ $0 >= 0 }),
      strides.allSatisfy({ $0 >= 0 }), width > 0, storageByteCount >= 0
    else { throw bridgeError(.unsupported, "Invalid or negative NDArray strides") }
    self.shape = shape
    self.strides = strides
    var count = 1
    var maximumOffset = 0
    for (dimension, stride) in zip(shape, strides) {
      let (next, countOverflow) = count.multipliedReportingOverflow(by: dimension)
      let (extent, extentOverflow) = max(dimension - 1, 0).multipliedReportingOverflow(by: stride)
      let (offset, offsetOverflow) = maximumOffset.addingReportingOverflow(extent)
      guard !countOverflow, !extentOverflow, !offsetOverflow else {
        throw bridgeError(.runtime, "NDArray layout overflows Int")
      }
      count = next
      maximumOffset = offset
    }
    self.count = count
    var expectedStride = 1
    var isDense = true
    for axis in shape.indices.reversed() where shape[axis] != 1 {
      isDense = isDense && strides[axis] == expectedStride
      expectedStride = expectedStride &* shape[axis]
    }
    self.isDense = isDense
    if count > 0 {
      let (span, spanOverflow) = maximumOffset.addingReportingOverflow(1)
      let (bytes, byteOverflow) = span.multipliedReportingOverflow(by: width)
      guard !spanOverflow, !byteOverflow, bytes <= storageByteCount else {
        throw bridgeError(.runtime, "NDArray layout exceeds its storage")
      }
    }
  }

  func offset(for linearIndex: Int) -> Int {
    var remainder = linearIndex
    var offset = 0
    for axis in shape.indices.reversed() {
      offset += (remainder % shape[axis]) * strides[axis]
      remainder /= shape[axis]
    }
    return offset
  }
}

@available(macOS 27.0, iOS 27.0, *)
func prepareOutput(_ array: NDArray) throws -> ETCoreAITensor {
  let type = try CoreAIScalarInfo.from(sdkType: array.scalarType)
  let byteCount = try type.byteCount(shape: array.shape, errorCode: .runtime)
  guard array.interleaveLayout == nil else {
    throw bridgeError(.unsupported, "Interleaved output storage is not supported")
  }
  let view = array.rawView()
  let layout = try CoreAITensorLayout(
    shape: array.shape, strides: array.strides, width: type.byteWidth,
    storageByteCount: view.bytes.byteCount)
  let shape = view.shape
  let strides = view.strides
  guard view.scalarType == type.sdkType, view.interleaveLayout == nil,
    shape.count == layout.shape.count, strides.count == layout.strides.count
  else {
    throw bridgeError(.runtime, "Output view layout changed")
  }
  for axis in layout.shape.indices {
    guard shape[axis] == layout.shape[axis], strides[axis] == layout.strides[axis] else {
      throw bridgeError(.runtime, "Output view layout changed")
    }
  }
  // Retain the immutable array, not its scoped pointer. All fallible work is complete.
  return ETCoreAITensor(
    byteCount: UInt(byteCount), shape: layout.shape.map(NSNumber.init(value:)),
    scalarType: type.bridgeType
  ) { destination in
    let view = array.rawView()
    view.bytes.withUnsafeBytes { source in
      if layout.isDense {
        destination.copyMemory(from: source.baseAddress!, byteCount: byteCount)
        return
      }
      for index in 0..<layout.count {
        destination.advanced(by: index * type.byteWidth).copyMemory(
          from: source.baseAddress!.advanced(by: layout.offset(for: index) * type.byteWidth),
          byteCount: type.byteWidth)
      }
    }
  }
}

@available(macOS 27.0, iOS 27.0, *)
private final class CoreAISwiftSession: NSObject, ETCoreAISession, Sendable {
  let model: AIModel
  let function: InferenceFunction
  let inputNames: [String]
  let outputNames: [String]
  let inputDescriptors: [NDArrayDescriptor]
  let outputDescriptors: [NDArrayDescriptor]

  init(model: AIModel, function: InferenceFunction, inputNames: [String], outputNames: [String])
    throws
  {
    let descriptor = function.descriptor
    guard descriptor.stateNames.isEmpty else {
      throw bridgeError(.unsupported, "Stateful functions are not supported")
    }
    guard inputNames.count == descriptor.inputCount,
      outputNames.count == descriptor.outputCount,
      Set(inputNames).count == inputNames.count, Set(outputNames).count == outputNames.count,
      descriptor.inputNames.count == inputNames.count,
      descriptor.outputNames.count == outputNames.count,
      Set(inputNames) == Set(descriptor.inputNames), Set(outputNames) == Set(descriptor.outputNames)
    else {
      throw bridgeError(.invalidModel, "Ordered bindings do not match the function descriptor")
    }
    self.model = model
    self.function = function
    self.inputNames = inputNames
    self.outputNames = outputNames
    inputDescriptors = try inputNames.map {
      try tensorDescriptor(descriptor.inputDescriptor(of: $0))
    }
    outputDescriptors = try outputNames.map {
      try tensorDescriptor(descriptor.outputDescriptor(of: $0))
    }
    super.init()
  }

  // All nonescapable input views end here on both return and throw, before completion.
  private func runBorrowing(_ payloads: [CoreAITensorPayload]) async throws
    -> InferenceFunction.Outputs
  {
    var arguments = InferenceFunction.Inputs()
    for index in payloads.indices {
      let payload = payloads[index]
      let descriptor = inputDescriptors[index]
      try validateShape(payload.shape, descriptor: descriptor)
      let expectedType = try CoreAIScalarInfo.from(sdkType: descriptor.scalarType)
      guard payload.scalarType == expectedType.bridgeType else {
        throw bridgeError(.invalidArgument, "Input dtype does not match the function descriptor")
      }
      // Unsafe boundary: the ET caller, not this metadata snapshot, owns the storage.
      let bytes = RawSpan(
        _unsafeBytes: UnsafeRawBufferPointer(
          start: payload.bytes, count: payload.byteCount))
      let view = NDArray.RawView(
        bytes: bytes, scalarType: descriptor.scalarType, shape: payload.shape)
      arguments.insert(view, for: inputNames[index])
    }
    return try await function.run(inputs: arguments)
  }

  func executeInputs(
    _ inputs: [ETCoreAIInputTensor], completion: @escaping ([ETCoreAITensor]?, Error?) -> Void
  ) {
    let payloads: [CoreAITensorPayload]
    do {
      guard inputs.count == inputNames.count else {
        throw bridgeError(.invalidArgument, "Input count does not match the function")
      }
      payloads = try inputs.map(CoreAITensorPayload.init)
    } catch {
      completion(nil, mappedError(error, code: .invalidArgument, context: "Invalid inputs"))
      return
    }
    let callback = Completion(completion)
    Task.detached { [self, payloads, callback] in
      do {
        var results = try await runBorrowing(payloads)
        guard results.count == outputNames.count, Set(results.names) == Set(outputNames) else {
          throw bridgeError(.runtime, "Inference returned unexpected output names")
        }
        var tensors: [ETCoreAITensor] = []
        for index in outputNames.indices {
          guard let value = results.remove(outputNames[index]) else {
            throw bridgeError(.runtime, "Inference output is missing")
          }
          guard let array = value.ndArray else {
            throw bridgeError(.unsupported, "Inference returned an image output")
          }
          let descriptor = outputDescriptors[index]
          try validateShape(array.shape, descriptor: descriptor, errorCode: .runtime)
          guard array.scalarType == descriptor.scalarType else {
            throw bridgeError(.runtime, "Output dtype does not match the function descriptor")
          }
          tensors.append(try prepareOutput(array))
        }
        callback.call(tensors, nil)
      } catch {
        callback.call(nil, mappedError(error, code: .runtime, context: "Core AI inference failed"))
      }
    }
  }
}

@available(macOS 27.0, iOS 27.0, *)
private func bindFunction(
  model: AIModel, functionName: String, inputNames: [String], outputNames: [String]
) throws -> CoreAISwiftSession {
  guard !functionName.isEmpty else {
    throw bridgeError(.invalidArgument, "A nonempty function name is required")
  }
  guard let function = try model.loadFunction(named: functionName) else {
    throw bridgeError(.invalidModel, "Model has no function named '\(functionName)'")
  }
  return try CoreAISwiftSession(
    model: model, function: function, inputNames: inputNames, outputNames: outputNames)
}

@available(macOS 27.0, iOS 27.0, *)
private final class CoreAISwiftPreparedModel: NSObject, ETCoreAIPreparedModel, Sendable {
  let model: AIModel

  init(model: AIModel) {
    self.model = model
    super.init()
  }

  func copyBookmarkData() -> Data {
    model.bookmarkData
  }

  // Binding never awaits, so it completes on the calling thread.
  func loadFunctionNamed(
    _ functionName: String, inputNames: [String], outputNames: [String],
    completion: @escaping ((any ETCoreAISession)?, Error?) -> Void
  ) {
    do {
      completion(
        try bindFunction(
          model: model, functionName: functionName, inputNames: inputNames,
          outputNames: outputNames), nil)
    } catch {
      completion(
        nil, mappedError(error, code: .runtime, context: "Core AI function binding failed"))
    }
  }
}

@available(macOS 27.0, iOS 27.0, *)
struct CoreAISpecialization: Sendable {
  let options: SpecializationOptions
  let identifier: String

  static let `default` = Self()

  private init() {
    options = .default
    identifier = "default"
  }

  // New SDK option fields require a schema review and versioned identity changes.
  private struct Descriptor: Encodable {
    let version: Int
    let allowedComputeUnits: [String]
    let preferredComputeUnit: String?
    let expectFrequentReshapes: Bool
  }

  init(options: SpecializationOptions) throws {
    if options == .default {
      self = Self.default
      return
    }
    let descriptor = Descriptor(
      version: 1,
      allowedComputeUnits: try options.allowedComputeUnitKinds.map(Self.name).sorted(),
      preferredComputeUnit: try options.preferredComputeUnitKind.map(Self.name),
      expectFrequentReshapes: options.expectFrequentReshapes)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    identifier = String(decoding: try encoder.encode(descriptor), as: UTF8.self)
    self.options = options
  }

  private static func name(_ kind: ComputeUnitKind) throws -> String {
    switch kind {
    case .cpu: return "cpu"
    case .gpu: return "gpu"
    case .neuralEngine: return "neuralEngine"
    @unknown default:
      throw bridgeError(.unsupported, "Unknown specialization compute unit")
    }
  }
}

@available(macOS 27.0, iOS 27.0, *)
@objc(ETCoreAISwiftModelLoader)
public final class ETCoreAISwiftModelLoader: NSObject, ETCoreAIModelLoading {
  static let cachePolicy = AIModelCache.Policy(purgeConditions: [.storagePressure])
  private let specialization: CoreAISpecialization

  public override init() {
    specialization = .default
    super.init()
  }

  init(specialization: CoreAISpecialization) {
    self.specialization = specialization
    super.init()
  }

  public var specializationIdentifier: String { specialization.identifier }

  @objc public static var deviceArchitectureName: String { AIModel.deviceArchitectureName }

  public func restoreModel(
    fromBookmark bookmark: Data, completion: @escaping ETCoreAIAcquisitionCompletion
  ) {
    acquireCoreAIModel(context: "Core AI bookmark restoration failed", completion: completion) {
      try AIModel(resolvingBookmark: bookmark).map { CoreAISwiftPreparedModel(model: $0) }
    }
  }

  public func specializeModel(at url: URL, completion: @escaping ETCoreAIAcquisitionCompletion) {
    let options = specialization.options
    acquireCoreAIModel(context: "Core AI specialization failed", completion: completion) {
      guard url.isFileURL else {
        throw bridgeError(.invalidArgument, "A file URL is required")
      }
      let model = try await AIModel.specialize(
        contentsOf: url, options: options, cache: .default, cachePolicy: Self.cachePolicy)
      return CoreAISwiftPreparedModel(model: model)
    }
  }

  public func evictModel(
    withBookmark bookmark: Data, completion: @escaping (Error?) -> Void
  ) {
    let callback = Completion(completion)
    Task.detached {
      do {
        try AIModelCache.deleteEntry(referencedBy: bookmark)
        callback.call(nil)
      } catch {
        callback.call(
          mappedError(error, code: .runtime, context: "Core AI cache eviction failed"))
      }
    }
  }
}
