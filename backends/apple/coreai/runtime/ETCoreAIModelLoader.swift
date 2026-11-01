// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import CoreAI
import Foundation

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
