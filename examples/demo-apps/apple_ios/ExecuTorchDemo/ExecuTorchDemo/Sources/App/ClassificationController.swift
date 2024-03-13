/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ImageClassification
import MobileNetClassifier
import SwiftUI

enum Mode: String, CaseIterable {
  case xnnpack = "XNNPACK"
  case coreML = "Core ML"
  case mps = "MPS"
}

class ClassificationController: ObservableObject {
  @AppStorage("mode") var mode: Mode = .xnnpack
  @Published var classifications: [Classification] = []
  @Published var elapsedTime: TimeInterval = 0.0
  @Published var isRunning = false

  private let queue = DispatchQueue(label: "org.pytorch.executorch.demo", qos: .userInitiated)
  private var classifier: ImageClassification?
  private var currentMode: Mode = .xnnpack

  func classify(_ image: UIImage) {
    guard !isRunning else {
      print("Dropping frame")
      return
    }
    isRunning = true

    if currentMode != mode {
      currentMode = mode
      classifier = nil
    }
    queue.async {
      var classifications: [Classification] = []
      var elapsedTime: TimeInterval = -1
      do {
        if self.classifier == nil {
          self.classifier = try self.createClassifier(for: self.currentMode)
        }
        let startTime = CFAbsoluteTimeGetCurrent()
        classifications = try self.classifier?.classify(image: image) ?? []
        elapsedTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
      } catch {
        print("Error classifying image: \(error)")
      }
      DispatchQueue.main.async {
        self.classifications = classifications
        self.elapsedTime = elapsedTime
        self.isRunning = false
      }
    }
  }

  private func createClassifier(for mode: Mode) throws -> ImageClassification? {
    let modelFileName: String
    switch mode {
    case .coreML:
      modelFileName = "mv3_coreml_all"
    case .mps:
      modelFileName = "mv3_mps"
    case .xnnpack:
      modelFileName = "mv3_xnnpack_fp32"
    }
    guard let modelFilePath = Bundle.main.path(forResource: modelFileName, ofType: "pte"),
          let labelsFilePath = Bundle.main.path(forResource: "imagenet_classes", ofType: "txt")
    else { return nil }
    return try MobileNetClassifier(modelFilePath: modelFilePath, labelsFilePath: labelsFilePath)
  }
}
