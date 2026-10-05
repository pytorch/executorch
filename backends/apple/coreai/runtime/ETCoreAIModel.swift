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
private final class Completion<Value>: @unchecked Sendable {
  let call: (Value?, Error?) -> Void

  init(_ call: @escaping (Value?, Error?) -> Void) {
    self.call = call
  }
}

private final class AcquisitionCompletion: @unchecked Sendable {
  let call: ETCoreAIAcquisitionCompletion

  init(_ call: @escaping ETCoreAIAcquisitionCompletion) {
    self.call = call
  }
}

// Only a successful SDK nil is a miss; thrown errors retain their underlying cause.
func acquireCoreAIModel(
  context: String, completion: @escaping ETCoreAIAcquisitionCompletion,
  operation: @escaping @Sendable () async throws -> (any ETCoreAIPreparedModel)?
) {
  let callback = AcquisitionCompletion(completion)
  Task.detached {
    do {
      if let model = try await operation() {
        callback.call(.hit, model, nil)
      } else {
        callback.call(.miss, nil, nil)
      }
    } catch {
      callback.call(.error, nil, mappedError(error, code: .invalidModel, context: context))
    }
  }
}

// ExecuTorch waits synchronously and keeps storage valid and unchanged until completion.
// This snapshot transfers only metadata and a pointer to the task; it does not own storage.
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

  static func byteCount(shape: [Int], scalarType: ETCoreAIScalarType) throws -> Int {
    let width: Int
    switch scalarType {
    case .float16: width = 2
    case .float32: width = 4
    default: throw bridgeError(.unsupported, "Only FP16 and FP32 tensors are supported")
    }
    guard shape.allSatisfy({ $0 >= 0 }) else {
      throw bridgeError(.invalidArgument, "Negative tensor dimension")
    }
    // Check suffix products too: SDK dense-stride construction must not overflow on empty tensors.
    var count = width
    for dimension in shape.reversed() {
      let (next, overflow) = count.multipliedReportingOverflow(by: max(dimension, 1))
      guard !overflow else {
        throw bridgeError(.invalidArgument, "Tensor byte count overflows Int")
      }
      count = next
    }
    return shape.contains(0) ? 0 : count
  }
}

@available(macOS 27.0, iOS 27.0, *)
private func bridgeScalarType(_ type: NDArray.ScalarType) throws -> ETCoreAIScalarType {
  switch type {
  case .float16: return .float16
  case .float32: return .float32
  default: throw bridgeError(.unsupported, "Only FP16 and FP32 NDArrays are supported")
  }
}

@available(macOS 27.0, iOS 27.0, *)
private func tensorDescriptor(_ value: InferenceValue.Descriptor?) throws -> NDArrayDescriptor {
  guard let value else { throw bridgeError(.invalidModel, "Missing tensor descriptor") }
  guard case .ndArray(let descriptor) = value else {
    throw bridgeError(.unsupported, "Image values are not supported")
  }
  _ = try bridgeScalarType(descriptor.scalarType)
  guard descriptor.interleaveLayout == nil else {
    throw bridgeError(.unsupported, "Interleaved tensor descriptors are not supported")
  }
  guard descriptor.shape.allSatisfy({ $0 >= -1 }) else {
    throw bridgeError(.invalidModel, "Invalid descriptor dimension")
  }
  return descriptor
}

@available(macOS 27.0, iOS 27.0, *)
private func validateShape(_ shape: [Int], descriptor: NDArrayDescriptor) throws {
  guard shape.count == descriptor.shape.count,
    zip(shape, descriptor.shape).allSatisfy({ actual, expected in
      actual >= 0 && (expected == -1 || actual == expected)
    })
  else {
    throw bridgeError(.invalidArgument, "Tensor shape does not match the function descriptor")
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
        throw bridgeError(.invalidArgument, "NDArray layout overflows Int")
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
        throw bridgeError(.invalidArgument, "NDArray layout exceeds its storage")
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
func copyOutput<T: BitwiseCopyable>(_ array: NDArray, as: T.Type) throws -> Data {
  let type = try bridgeScalarType(array.scalarType)
  guard (T.self == Float.self && type == .float32) || (T.self == Float16.self && type == .float16)
  else {
    throw bridgeError(.invalidArgument, "Output copy dtype mismatch")
  }
  let byteCount = try CoreAITensorPayload.byteCount(shape: array.shape, scalarType: type)
  guard array.interleaveLayout == nil else {
    throw bridgeError(.unsupported, "Interleaved output storage is not supported")
  }
  let layout = try CoreAITensorLayout(
    shape: array.shape, strides: array.strides, width: MemoryLayout<T>.stride,
    storageByteCount: array.rawView().bytes.byteCount)
  var data = Data(count: byteCount)
  let view = array.view(as: T.self)
  try view.withUnsafePointer { source, shape, strides in
    guard shape.count == layout.shape.count, strides.count == layout.strides.count else {
      throw bridgeError(.runtime, "Output view layout changed")
    }
    for axis in layout.shape.indices {
      guard shape[axis] == layout.shape[axis], strides[axis] == layout.strides[axis] else {
        throw bridgeError(.runtime, "Output view layout changed")
      }
    }
    try data.withUnsafeMutableBytes { destination in
      guard layout.count > 0 else { return }
      guard let base = destination.baseAddress else {
        throw bridgeError(.runtime, "Nonempty output has null data")
      }
      if layout.isDense {
        base.copyMemory(from: source, byteCount: layout.count * MemoryLayout<T>.stride)
        return
      }
      for index in 0..<layout.count {
        base.advanced(by: index * MemoryLayout<T>.stride).copyMemory(
          from: source.advanced(by: layout.offset(for: index)), byteCount: MemoryLayout<T>.stride)
      }
    }
  }
  return data
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
      guard payload.scalarType == (try bridgeScalarType(descriptor.scalarType)) else {
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
          try validateShape(array.shape, descriptor: descriptor)
          guard array.scalarType == descriptor.scalarType else {
            throw bridgeError(.runtime, "Output dtype does not match the function descriptor")
          }
          let type = try bridgeScalarType(array.scalarType)
          let data =
            try type == .float16
            ? copyOutput(array, as: Float16.self) : copyOutput(array, as: Float.self)
          tensors.append(
            ETCoreAITensor(
              data: data, shape: array.shape.map(NSNumber.init(value:)), scalarType: type))
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
        nil, mappedError(error, code: .invalidModel, context: "Core AI function binding failed"))
    }
  }
}

@available(macOS 27.0, iOS 27.0, *)
@objc(ETCoreAISwiftModelLoader)
public final class ETCoreAISwiftModelLoader: NSObject, ETCoreAIModelLoading {
  @objc public static var deviceArchitectureName: String { AIModel.deviceArchitectureName }

  public func restoreModel(
    fromBookmark bookmark: Data, completion: @escaping ETCoreAIAcquisitionCompletion
  ) {
    acquireCoreAIModel(context: "Core AI bookmark restoration failed", completion: completion) {
      try AIModel(resolvingBookmark: bookmark).map { CoreAISwiftPreparedModel(model: $0) }
    }
  }

  public func specializeModel(at url: URL, completion: @escaping ETCoreAIAcquisitionCompletion) {
    acquireCoreAIModel(context: "Core AI specialization failed", completion: completion) {
      guard url.isFileURL else {
        throw bridgeError(.invalidArgument, "A file URL is required")
      }
      let model = try await AIModel.specialize(
        contentsOf: url, options: .default, cache: .default, cachePolicy: .persistent)
      return CoreAISwiftPreparedModel(model: model)
    }
  }

  public func evictModel(
    withBookmark bookmark: Data, completion: @escaping (Error?) -> Void
  ) {
    let callback = Completion<Void> { _, error in completion(error) }
    Task.detached {
      do {
        try AIModelCache.deleteEntry(referencedBy: bookmark)
        callback.call((), nil)
      } catch {
        callback.call(
          nil, mappedError(error, code: .runtime, context: "Core AI cache eviction failed"))
      }
    }
  }
}
