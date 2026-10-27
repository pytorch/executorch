// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import CoreAI
import Foundation

private struct Fixture: Decodable {
  let function: String
  let inputs: [String]
  let outputs: [String]
  let expected: [[Float]]
}

private enum Failure: Error {
  case invalidArguments
  case missingFunction
  case incorrectOutput(String)
  case unexpectedSource
}

private func checkInference(_ model: AIModel, fixture: Fixture) async throws {
  guard let function = try model.loadFunction(named: fixture.function) else {
    throw Failure.missingFunction
  }
  guard fixture.inputs.count == 2, fixture.outputs.count == fixture.expected.count else {
    throw Failure.invalidArguments
  }
  var inputs: [String: NDArray] = [:]
  for (index, values) in [[Float(5), 9], [Float(1), 2]].enumerated() {
    var array = NDArray(shape: [2], scalarType: .float32)
    do {
      var view = array.mutableView(as: Float.self)
      view.withUnsafeMutablePointer { pointer, _, _ in
        pointer[0] = values[0]
        pointer[1] = values[1]
      }
    }
    inputs[fixture.inputs[index]] = array
  }
  var outputs = try await function.run(inputs: inputs)
  guard outputs.count == fixture.outputs.count else {
    throw Failure.incorrectOutput("count")
  }
  for (index, name) in fixture.outputs.enumerated() {
    guard let value = outputs.remove(name), let array = value.ndArray,
      array.shape == [2], array.scalarType == .float32,
      fixture.expected[index].count == 2
    else { throw Failure.incorrectOutput(name) }
    let values = array.view(as: Float.self).withUnsafePointer { pointer, _, strides in
      [pointer[0], pointer[strides[0]]]
    }
    guard zip(values, fixture.expected[index]).allSatisfy({ abs($0 - $1) < 0.001 }) else {
      throw Failure.incorrectOutput("\(name): \(values), expected \(fixture.expected[index])")
    }
  }
}

@main
private struct SDKSourceLifetimeTest {
  static func main() async {
    do {
      let args = CommandLine.arguments
      if args.count == 2, args[1] == "architecture" {
        print(AIModel.deviceArchitectureName)
        return
      }
      guard args.count == 5 else { throw Failure.invalidArguments }
      let mode = args[1]
      let source = URL(fileURLWithPath: args[2]).standardizedFileURL
      let fixtureURL = URL(fileURLWithPath: args[3]).standardizedFileURL
      let bookmarkURL = URL(fileURLWithPath: args[4])
      let directory = source.deletingLastPathComponent()
      guard ["model.aimodel", "model.aimodelc"].contains(source.lastPathComponent),
        directory == fixtureURL.deletingLastPathComponent(),
        directory.deletingLastPathComponent().lastPathComponent.hasPrefix("coreai-lifetime-")
      else { throw Failure.invalidArguments }
      if mode == "evict" {
        try AIModelCache.deleteEntry(referencedBy: Data(contentsOf: bookmarkURL))
        return
      }
      let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
      if mode == "specialize" {
        let model = try await AIModel.specialize(
          contentsOf: source, options: .default, cache: .default, cachePolicy: .persistent)
        try model.bookmarkData.write(to: bookmarkURL, options: .atomic)
        try await checkInference(model, fixture: fixture)
        try FileManager.default.removeItem(at: source)
        guard !FileManager.default.fileExists(atPath: source.path) else {
          throw Failure.unexpectedSource
        }
        try await checkInference(model, fixture: fixture)
        print("Live inference passed after source removal")
      } else if mode == "restore" {
        guard !FileManager.default.fileExists(atPath: source.path) else {
          throw Failure.unexpectedSource
        }
        guard let model = try AIModel(resolvingBookmark: Data(contentsOf: bookmarkURL)) else {
          exit(3)
        }
        try await checkInference(model, fixture: fixture)
        print("Bookmark restoration and inference passed without source")
      } else {
        throw Failure.invalidArguments
      }
    } catch {
      print("SDK source-lifetime test failed: \(error)")
      exit(1)
    }
  }
}
