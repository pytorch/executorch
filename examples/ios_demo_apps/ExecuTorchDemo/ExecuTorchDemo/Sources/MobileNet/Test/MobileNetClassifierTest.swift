/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ImageClassification
import XCTest

@testable import MobileNetClassifier

final class MobileNetClassifierTest: XCTestCase {

  func testBenchmark() throws {
    try run(
      model: "mv3",
      [
        Classification(label: "Arctic fox", confidence: 0.93),
        Classification(label: "Samoyed", confidence: 0.75),
        Classification(label: "hot pot", confidence: 0.82),
      ])
  }

  func testV3WithXnnPackBackend() throws {
    try run(
      model: "mv3_xnnpack_fp32",
      [
        Classification(label: "Arctic fox", confidence: 0.93),
        Classification(label: "Samoyed", confidence: 0.75),
        Classification(label: "hot pot", confidence: 0.82),
      ])
  }

  func testV3WithCoreMLBackend() throws {
    try run(
      model: "mv3_coreml",
      [
        Classification(label: "Arctic fox", confidence: 0.93),
        Classification(label: "Samoyed", confidence: 0.75),
        Classification(label: "hot pot", confidence: 0.82),
      ])
  }

  private func run(model modelName: String, _ expectedClassifications: [Classification]) throws {
    guard
      let modelFilePath = Bundle(for: type(of: self))
        .path(forResource: modelName, ofType: "pte"),
      let labelsFilePath = Bundle(for: type(of: self))
        .path(forResource: "imagenet_classes", ofType: "txt")
    else {
      XCTFail("Failed to get model or labels path")
      return
    }
    let classifier = try MobileNetClassifier(
      modelFilePath: modelFilePath,
      labelsFilePath: labelsFilePath)

    for expectedClassification in expectedClassifications {
      guard
        let imagePath = Bundle(for: type(of: self))
          .path(forResource: expectedClassification.label, ofType: "jpg"),
        let image = UIImage(contentsOfFile: imagePath)
      else {
        XCTFail("Failed to get image path or image")
        return
      }
      guard let classification = try classifier?.classify(image: image).first
      else {
        XCTFail("Failed to run the model")
        return
      }
      XCTAssertEqual(classification.label, expectedClassification.label)
      XCTAssertGreaterThan(classification.confidence, expectedClassification.confidence)
    }
  }
}
