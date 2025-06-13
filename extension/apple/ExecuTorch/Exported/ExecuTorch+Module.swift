/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@_exported import ExecuTorch

@available(*, deprecated, message: "This API is experimental.")
public extension Module {
  /// Executes a specific method with the provided input values.
  /// The method is loaded on demand if not already loaded.
  ///
  /// - Parameters:
  ///   - method: The name of the method to execute.
  ///   - inputs: An array of `ValueConvertible` types representing the inputs.
  /// - Returns: An array of `Value` objects representing the outputs.
  /// - Throws: An error if method execution fails.
  func execute(_ method: String, _ inputs: [ValueConvertible]) throws -> [Value] {
    try __executeMethod(method, withInputs: inputs.map { $0.objcValue() } )
  }

  /// Executes a specific method with a single input value.
  /// The method is loaded on demand if not already loaded.
  ///
  /// - Parameters:
  ///   - method: The name of the method to execute.
  ///   - input: A single `ValueConvertible` type representing the input.
  /// - Returns: An array of `Value` objects representing the outputs.
  /// - Throws: An error if method execution fails.
  func execute(_ method: String, _ input: ValueConvertible) throws -> [Value] {
    try __executeMethod(method, withInputs: [input.objcValue()])
  }

  /// Executes the "forward" method with the provided input values.
  /// The method is loaded on demand if not already loaded.
  ///
  /// - Parameter inputs: An array of `ValueConvertible` types representing the inputs.
  /// - Returns: An array of `Value` objects representing the outputs.
  /// - Throws: An error if method execution fails.
  func forward(_ inputs: [ValueConvertible]) throws -> [Value] {
    try __executeMethod("forward", withInputs: inputs.map { $0.objcValue() })
  }

  /// Executes the "forward" method with a single input value.
  /// The method is loaded on demand if not already loaded.
  ///
  /// - Parameter input: A single `ValueConvertible` type representing the input.
  /// - Returns: An array of `Value` objects representing the outputs.
  /// - Throws: An error if method execution fails.
  func forward(_ input: ValueConvertible) throws -> [Value] {
    try __executeMethod("forward", withInputs: [input.objcValue()])
  }
}
