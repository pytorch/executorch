/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@_exported import ExecuTorch

/// A protocol that provides a uniform way to convert different Swift types
/// into a `Value`.
@available(*, deprecated, message: "This API is experimental.")
public protocol ValueConvertible {
  /// Converts the instance into a `Value`.
  func asValue() -> Value
}

@available(*, deprecated, message: "This API is experimental.")
public extension Value {
  /// Creates a `Value` instance encapsulating a `Tensor`.
  ///
  /// - Parameter tensor: The `Tensor` to wrap.
  convenience init<T: Scalar>(_ tensor: Tensor<T>) {
    self.init(tensor.anyTensor)
  }

  /// Attempts to return the underlying type-erased `AnyTensor` if the `Value` contains one.
  ///
  /// - Returns: An `AnyTensor`, or `nil` if the `Value` is not a tensor.
  var anyTensor: AnyTensor? {
    __tensorValue
  }

  /// Attempts to return the underlying `Tensor` if the `Value` contains one.
  ///
  /// - Returns: A `Tensor` of the specified scalar type, or `nil` if the
  ///   `Value` is not a tensor or the data type does not match.
  func tensor<T: Scalar>() -> Tensor<T>? {
    anyTensor?.asTensor()
  }
}

// MARK: - ValueConvertible Conformances

@available(*, deprecated, message: "This API is experimental.")
extension Value: ValueConvertible {
  /// Returns the `Value` itself.
  public func asValue() -> Value { self }
}

@available(*, deprecated, message: "This API is experimental.")
extension AnyTensor: ValueConvertible {
  /// Converts the `Tensor` into a `Value`.
  public func asValue() -> Value { Value(self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Tensor: ValueConvertible {
  /// Converts the `Tensor` into a `Value`.
  public func asValue() -> Value { Value(self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension String: ValueConvertible {
  /// Converts the `String` into a `Value`.
  public func asValue() -> Value { Value(self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension NSNumber: ValueConvertible {
  /// Converts the `NSNumber` into a `Value`.
  public func asValue() -> Value { Value(self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt8: ValueConvertible {
  /// Converts the `UInt8` into a `Value`.
  public func asValue() -> Value { Value(NSNumber(value: Int(self))) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int8: ValueConvertible {
  /// Converts the `Int8` into a `Value`.
  public func asValue() -> Value { Value(NSNumber(value: Int(self))) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int16: ValueConvertible {
  /// Converts the `Int16` into a `Value`.
  public func asValue() -> Value { Value(NSNumber(value: self)) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int32: ValueConvertible {
  /// Converts the `Int32` into a `Value`.
  public func asValue() -> Value { Value(NSNumber(value: self)) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int64: ValueConvertible {
  /// Converts the `Int64` into a `Value`.
  public func asValue() -> Value { Value(NSNumber(value: self)) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int: ValueConvertible {
  /// Converts the `Int` into a `Value`.
  public func asValue() -> Value { Value(self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Float: ValueConvertible {
  /// Converts the `Float` into a `Value`.
  public func asValue() -> Value { Value(self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Double: ValueConvertible {
  /// Converts the `Double` into a `Value`.
  public func asValue() -> Value { Value(self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension Bool: ValueConvertible {
  /// Converts the `Bool` into a `Value`.
  public func asValue() -> Value { Value(self) }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt16: ValueConvertible {
  /// Converts the `UInt16` into a `Value`.
  public func asValue() -> Value { Value(NSNumber(value: self)) }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt32: ValueConvertible {
  /// Converts the `UInt32` into a `Value`.
  public func asValue() -> Value { Value(NSNumber(value: self)) }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt64: ValueConvertible {
  /// Converts the `UInt64` into a `Value`.
  public func asValue() -> Value { Value(NSNumber(value: self)) }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt: ValueConvertible {
  /// Converts the `UInt` into a `Value`.
  public func asValue() -> Value { Value(NSNumber(value: self)) }
}
