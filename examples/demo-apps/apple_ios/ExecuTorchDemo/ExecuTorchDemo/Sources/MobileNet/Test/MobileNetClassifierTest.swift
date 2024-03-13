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

  func testV3WithPortableBackend() throws {
    try run(model: "mv3")
  }

  func testV3WithCoreMLBackend() throws {
    try run(model: "mv3_coreml_all")
  }

  func testV3WithMPSBackend() throws {
    try run(model: "mv3_mps")
  }

  func testV3WithXNNPACKBackend() throws {
    try run(model: "mv3_xnnpack_fp32")
  }

  private func run(model modelName: String) throws {
    guard
      let modelFilePath = Bundle(for: type(of: self))
        .path(forResource: modelName, ofType: "pte")
    else {
      XCTFail("Failed to get model path")
      return
    }
    guard
      let labelsFilePath = Bundle(for: type(of: self))
        .path(forResource: "imagenet_classes", ofType: "txt")
    else {
      XCTFail("Failed to get labels path")
      return
    }
    let classifier = try MobileNetClassifier(
      modelFilePath: modelFilePath,
      labelsFilePath: labelsFilePath)
    for expectedClassification in [
      Classification(label: "Arctic fox", confidence: 0.92),
      Classification(label: "Samoyed", confidence: 0.74),
      Classification(label: "hot pot", confidence: 0.82),
    ] {
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
