/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

@_exported import ExecuTorch

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

/// A protocol that provides a uniform way to convert different Swift types
/// into a `Value`.
@available(*, deprecated, message: "This API is experimental.")
public protocol ValueConvertible {
  /// Converts the instance into a `Value`.
  func asValue() -> Value
}

/// A protocol that provides a uniform way to create an instance from a `Value`.
@available(*, deprecated, message: "This API is experimental.")
public protocol ValueConstructible {
  /// Constructs the instance from a `Value`.
  static func from(_ value: Value) throws -> Self
}

@available(*, deprecated, message: "This API is experimental.")
public extension ValueConstructible {
  /// Sugar on top of `decode(from:)`
  init(_ value: Value) throws {
    self = try Self.from(value)
  }
}

/// A protocol that provides a uniform way to create an instance from an array of `Value`.
@available(*, deprecated, message: "This API is experimental.")
public protocol ValueSequenceConstructible {
  /// Constructs the instance from a `Value` array.
  static func from(_ values: [Value]) throws -> Self
}

@available(*, deprecated, message: "This API is experimental.")
extension ValueSequenceConstructible where Self: ValueConstructible {
  public static func from(_ values: [Value]) throws -> Self {
    guard values.count == 1 else { throw Error(code: .invalidType) }
    return try Self.from(values[0])
  }
}

@available(*, deprecated, message: "This API is experimental.")
public extension ValueSequenceConstructible {
  /// Sugar on top of `decode(from:)`
  init(_ values: [Value]) throws {
    self = try Self.from(values)
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

// MARK: - ValueConstructible Conformances

@available(*, deprecated, message: "This API is experimental.")
extension Value: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    value as! Self
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension AnyTensor: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let tensor = value.anyTensor else {
      throw Error(code: .invalidType, description: "Value is not a tensor")
    }
    return tensor as! Self
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension Tensor: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let anyTensor = value.anyTensor else {
      throw Error(code: .invalidType, description: "Value is not a tensor")
    }
    guard let tensor = Tensor<T>(anyTensor) as? Self else {
      throw Error(code: .invalidType, description: "Tensor is not of type \(Self.self)")
    }
    return tensor
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension String: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let string = value.string else {
      throw Error(code: .invalidType, description: "Value is not a string")
    }
    return string
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension NSNumber: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let scalar = value.scalar as? Self else {
      throw Error(code: .invalidType, description: "Value is not a scalar")
    }
    return scalar
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt8: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let scalar = value.scalar else {
      throw Error(code: .invalidType, description: "Value is not a scalar")
    }
    guard let integer = UInt8(exactly: scalar.uint8Value) else {
      throw Error(code: .invalidType, description: "Cannot convert scalar to \(Self.self)")
    }
    return integer
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int8: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let scalar = value.scalar else {
      throw Error(code: .invalidType, description: "Value is not a scalar")
    }
    guard let integer = Int8(exactly: scalar.int8Value) else {
      throw Error(code: .invalidType, description: "Cannot convert scalar to \(Self.self)")
    }
    return integer
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int16: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let scalar = value.scalar else {
      throw Error(code: .invalidType, description: "Value is not a scalar")
    }
    guard let integer = Int16(exactly: scalar.int16Value) else {
      throw Error(code: .invalidType, description: "Cannot convert scalar to \(Self.self)")
    }
    return integer
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int32: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let scalar = value.scalar else {
      throw Error(code: .invalidType, description: "Value is not a scalar")
    }
    guard let integer = Int32(exactly: scalar.int32Value) else {
      throw Error(code: .invalidType, description: "Cannot convert scalar to \(Self.self)")
    }
    return integer
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int64: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let scalar = value.scalar else {
      throw Error(code: .invalidType, description: "Value is not a scalar")
    }
    guard let integer = Int64(exactly: scalar.int64Value) else {
      throw Error(code: .invalidType, description: "Cannot convert scalar to \(Self.self)")
    }
    return integer
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension Int: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let scalar = value.scalar else {
      throw Error(code: .invalidType, description: "Value is not a scalar")
    }
    guard let integer = Int(exactly: scalar.intValue) else {
      throw Error(code: .invalidType, description: "Cannot convert scalar to \(Self.self)")
    }
    return integer
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension Float: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard value.isFloat else {
      throw Error(code: .invalidType, description: "Value is not a float")
    }
    return value.float as Self
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension Double: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard value.isDouble else {
      throw Error(code: .invalidType, description: "Value is not a double")
    }
    return value.double as Self
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension Bool: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard value.isBoolean else {
      throw Error(code: .invalidType, description: "Value is not a boolean")
    }
    return value.boolean as Self
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt16: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let scalar = value.scalar else {
      throw Error(code: .invalidType, description: "Value is not a scalar")
    }
    guard let integer = UInt16(exactly: scalar.uint16Value) else {
      throw Error(code: .invalidType, description: "Cannot convert scalar to \(Self.self)")
    }
    return integer
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt32: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let scalar = value.scalar else {
      throw Error(code: .invalidType, description: "Value is not a scalar")
    }
    guard let integer = UInt32(exactly: scalar.uint32Value) else {
      throw Error(code: .invalidType, description: "Cannot convert scalar to \(Self.self)")
    }
    return integer
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt64: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let scalar = value.scalar else {
      throw Error(code: .invalidType, description: "Value is not a scalar")
    }
    guard let integer = UInt64(exactly: scalar.uint64Value) else {
      throw Error(code: .invalidType, description: "Cannot convert scalar to \(Self.self)")
    }
    return integer
  }
}

@available(*, deprecated, message: "This API is experimental.")
extension UInt: ValueConstructible, ValueSequenceConstructible {
  public static func from(_ value: Value) throws -> Self {
    guard let scalar = value.scalar else {
      throw Error(code: .invalidType, description: "Value is not a scalar")
    }
    guard let integer = UInt(exactly: scalar.uintValue) else {
      throw Error(code: .invalidType, description: "Cannot convert scalar to \(Self.self)")
    }
    return integer
  }
}

// MARK: - ValueSequenceConstructible Conformances

@available(*, deprecated, message: "This API is experimental.")
extension Array: ValueSequenceConstructible where Element: ValueConstructible {
  public static func from(_ values: [Value]) throws -> [Element] {
    return try values.map { try Element.from($0) }
  }
}
