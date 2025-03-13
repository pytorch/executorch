// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

@_implementationOnly import ExecutorchRuntimeBridge
import Foundation
import ModelRunnerDataKit

public struct ExecutorchRuntimeValueSupport {

  public init() {}
}

extension ExecutorchRuntimeValueSupport: ModelRuntimeValueFactory {

  public func createString(value: String) throws -> ModelRuntimeValue {
    throw ModelRuntimeValueError.unsupportedType(String(describing: String.self))
  }

  public func createTensor(value: ModelRuntimeTensorValue) throws -> ModelRuntimeValue {
    guard let tensorValue = value.innerValue as? ExecutorchRuntimeTensorValue else {
      throw ModelRuntimeValueError.invalidType(
        String(describing: value.innerValue.self),
        String(describing: ExecutorchRuntimeTensorValue.self)
      )
    }
    return ModelRuntimeValue(innerValue: ExecutorchRuntimeValue(tensor: tensorValue))
  }
}

extension ExecutorchRuntimeValueSupport: ModelRuntimeTensorValueFactory {

  public func createFloatTensor(value: [Float], shape: [Int]) -> ModelRuntimeTensorValue {
    ModelRuntimeTensorValue(
      innerValue: ExecutorchRuntimeTensorValue(
        floatArray: value.compactMap { NSNumber(value: $0) },
        shape: shape.compactMap { NSNumber(value: $0) }
      )
    )
  }
}
