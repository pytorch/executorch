/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@_implementationOnly import ExecutorchRuntimeBridge
@_implementationOnly import ExecutorchRuntimeValueSupport
import Foundation
import ModelRunnerDataKit

public class ExecutorchRuntime: ModelRuntime {
  private let engine: ExecutorchRuntimeEngine
  public init(modelPath: String, modelMethodName: String) throws {
    self.engine = try ExecutorchRuntimeEngine(modelPath: modelPath, modelMethodName: modelMethodName)
  }
  public func infer(input: [ModelRuntimeValue]) throws -> [ModelRuntimeValue] {
    let modelInput = input.compactMap { $0.value as? ExecutorchRuntimeValue }
    // Not all values were of type ExecutorchRuntimeValue
    guard input.count == modelInput.count else {
      throw ModelRuntimeError.unsupportedInputType
    }
    return try engine.infer(input: modelInput).compactMap { ModelRuntimeValue(innerValue: $0) }
  }

  public func getModelValueFactory() -> ModelRuntimeValueFactory {
    return ExecutorchRuntimeValueSupport()
  }
  public func getModelTensorFactory() -> ModelRuntimeTensorValueFactory {
    return ExecutorchRuntimeValueSupport()
  }
}
