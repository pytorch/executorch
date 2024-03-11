// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import SwiftUI

final class ResourceManager: ObservableObject {
  @AppStorage("modelPath") var modelPath = ""
  @AppStorage("tokenizerPath") var tokenizerPath = ""
  private let fileManager = FileManager.default

  var isModelValid: Bool {
    fileManager.fileExists(atPath: modelPath)
  }

  var isTokenizerValid: Bool {
    fileManager.fileExists(atPath: tokenizerPath)
  }

  var modelName: String {
    URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent
  }

  var tokenizerName: String {
    URL(fileURLWithPath: tokenizerPath).deletingPathExtension().lastPathComponent
  }

  func createDirectoriesIfNeeded() throws {
    guard let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
    try fileManager.createDirectory(at: documentsDirectory.appendingPathComponent("models"), withIntermediateDirectories: true, attributes: nil)
    try fileManager.createDirectory(at: documentsDirectory.appendingPathComponent("tokenizers"), withIntermediateDirectories: true, attributes: nil)
  }
}
