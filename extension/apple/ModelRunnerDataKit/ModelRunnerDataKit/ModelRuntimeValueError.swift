// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import Foundation

public enum ModelRuntimeValueError: Error, CustomStringConvertible {
  case unsupportedType(String)
  case invalidType(String, String)

  public var description: String {
    switch self {
    case .unsupportedType(let type):
      return "Unsupported type: \(type)"
    case .invalidType(let expectedType, let type):
      return "Invalid type: \(type), expected \(expectedType)"
    }
  }
}

@objc public class ModelRuntimeValueErrorFactory: NSObject {
  @objc public class func unsupportedType(_ type: String) -> Error {
    return ModelRuntimeValueError.unsupportedType(type)
  }

  @objc public class func invalidType(_ actualType: String, expectedType: String) -> Error {
    return ModelRuntimeValueError.invalidType(expectedType, actualType)
  }
}
