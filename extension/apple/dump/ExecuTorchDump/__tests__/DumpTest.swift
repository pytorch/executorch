/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import ExecuTorch
import ExecuTorchDump
import XCTest

class DumpTest: XCTestCase {
  var resourceBundle: Bundle {
#if SWIFT_PACKAGE
    return Bundle.module
#else
    return Bundle(for: type(of: self))
#endif
  }

  private func modelPath() throws -> String {
    if let path = resourceBundle.path(forResource: "add", ofType: "pte") {
      return path
    }
    throw XCTSkip("add.pte not bundled.")
  }

  func testAvailable() {
    // The frameworks published for Apple platforms are built with tracing on, so
    // in that build a recorder can be created. A source build without tracing
    // reports the opposite, and the initializer fails rather than recording
    // nothing, which the next test covers.
    XCTAssertTrue(Dump.isAvailable)
  }

  func testRecordsAfterRun() throws {
    let path = try modelPath()
    guard Dump.isAvailable else {
      throw XCTSkip("This runtime was built without event tracing.")
    }
    let dump = try Dump(filePath: path)
    let inputs: [Tensor<Float>] = [Tensor([1]), Tensor([1])]
    XCTAssertNoThrow(try dump.module.forward(inputs))

    let data = try dump.takeData()
    XCTAssertFalse(data.isEmpty)
  }

  func testTakeWithoutRunReportsNoData() throws {
    let path = try modelPath()
    guard Dump.isAvailable else {
      throw XCTSkip("This runtime was built without event tracing.")
    }
    let dump = try Dump(filePath: path)
    // Nothing has run yet, so there is no trace to take.
    XCTAssertThrowsError(try dump.takeData()) { error in
      XCTAssertEqual((error as NSError).domain, DumpErrorDomain)
      XCTAssertEqual(
        (error as NSError).code, DumpError.noData.rawValue)
    }
  }

  func testSecondTakeReportsNoDataWithoutCrashing() throws {
    let path = try modelPath()
    guard Dump.isAvailable else {
      throw XCTSkip("This runtime was built without event tracing.")
    }
    let dump = try Dump(filePath: path)
    let inputs: [Tensor<Float>] = [Tensor([2]), Tensor([3])]
    XCTAssertNoThrow(try dump.module.forward(inputs))

    XCTAssertNoThrow(try dump.takeData())
    // Taking the trace completes it. A second take must report that there is
    // nothing new rather than abort the process on a finalized generator.
    XCTAssertThrowsError(try dump.takeData()) { error in
      XCTAssertEqual(
        (error as NSError).code, DumpError.noData.rawValue)
    }
  }

  func testWriteToFile() throws {
    let path = try modelPath()
    guard Dump.isAvailable else {
      throw XCTSkip("This runtime was built without event tracing.")
    }
    let dump = try Dump(filePath: path)
    let inputs: [Tensor<Float>] = [Tensor([13.25]), Tensor([29.25])]
    XCTAssertNoThrow(try dump.module.forward(inputs))

    let outputPath = NSTemporaryDirectory() + "/dump_test.etdump"
    try dump.takeData(toFile: outputPath)
    XCTAssertTrue(FileManager.default.fileExists(atPath: outputPath))
    XCTAssertGreaterThan(
      try FileManager.default.attributesOfItem(atPath: outputPath)[.size]
        as? Int ?? 0, 0)
    try? FileManager.default.removeItem(atPath: outputPath)
  }
}
