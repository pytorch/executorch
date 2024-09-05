/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI

final class ResourceManager: ObservableObject {
  @AppStorage("modelPath") var modelPath = ""
  @AppStorage("tokenizerPath") var tokenizerPath = ""
  @AppStorage("temperature") var temperature = 0.0 // 0 - 2.0
  @AppStorage("topK") var topK = 2 // 1 - 100
  @AppStorage("topP") var topP = 0.2 // 0 - 1.0
  @AppStorage("maxOutputTokens") var maxOutputTokens = 512 // 1 - 8192

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
