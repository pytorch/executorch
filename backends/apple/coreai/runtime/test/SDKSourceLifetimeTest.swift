// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import Foundation
import Synchronization

private struct Fixture: Decodable {
  let function: String
  let inputs: [String]
  let outputs: [String]
  let expected: [[Float]]
}

private enum Failure: Error {
  case invalidArguments
  case invalidResult
  case incorrectOutput(String)
  case unexpectedSource
}

// Bridge results are immutable and consumed only after completion.
private struct BridgeResult<Value>: @unchecked Sendable {
  let value: Result<Value, Error>
}

// The executable waits on an ordinary thread, not Swift's cooperative pool.
private func waitForResult<Value>(
  _ operation: (@escaping @Sendable (Result<Value, Error>) -> Void) -> Void
) throws -> Value {
  let result = Mutex<BridgeResult<Value>?>(nil)
  let ready = DispatchSemaphore(value: 0)
  operation { value in
    result.withLock { $0 = BridgeResult(value: value) }
    ready.signal()
  }
  ready.wait()
  let value = result.withLock { value in
    let snapshot = value
    value = nil
    return snapshot
  }
  guard let value else { throw Failure.invalidResult }
  return try value.value.get()
}

private func acquire(
  _ operation: (@escaping ETCoreAIAcquisitionCompletion) -> Void
) throws -> (any ETCoreAIPreparedModel)? {
  try waitForResult { completed in
    operation { status, model, error in
      if let error {
        completed(.failure(error))
      } else if (status == .hit && model != nil) || (status == .miss && model == nil) {
        completed(.success(model))
      } else {
        completed(.failure(Failure.invalidResult))
      }
    }
  }
}

private func checkInference(_ model: any ETCoreAIPreparedModel, fixture: Fixture) throws {
  guard fixture.inputs.count == 2, fixture.outputs.count == fixture.expected.count else {
    throw Failure.invalidArguments
  }
  let session: any ETCoreAISession = try waitForResult { completed in
    model.loadFunctionNamed(
      fixture.function, inputNames: fixture.inputs, outputNames: fixture.outputs
    ) { session, error in
      if let error {
        completed(.failure(error))
      } else if let session {
        completed(.success(session))
      } else {
        completed(.failure(Failure.invalidResult))
      }
    }
  }
  let storage = [[Float(5), 9], [Float(1), 2]].map { values in
    values.withUnsafeBytes { NSData(bytes: $0.baseAddress, length: $0.count) }
  }
  let inputs = storage.map {
    ETCoreAIInputTensor(
      bytes: $0.bytes, byteCount: UInt($0.length), shape: [2], scalarType: .float32)
  }
  let outputs: [ETCoreAITensor] = try withExtendedLifetime(storage) {
    try waitForResult { completed in
      session.executeInputs(inputs) { outputs, error in
        if let error {
          completed(.failure(error))
        } else if let outputs {
          completed(.success(outputs))
        } else {
          completed(.failure(Failure.invalidResult))
        }
      }
    }
  }
  guard outputs.count == fixture.outputs.count else {
    throw Failure.incorrectOutput("count")
  }
  for (index, output) in outputs.enumerated() {
    let name = fixture.outputs[index]
    guard output.shape == [2], output.scalarType == .float32,
      output.data.count == 8, fixture.expected[index].count == 2
    else { throw Failure.incorrectOutput(name) }
    let values = output.data.withUnsafeBytes {
      [$0.loadUnaligned(as: Float.self), $0.loadUnaligned(fromByteOffset: 4, as: Float.self)]
    }
    guard zip(values, fixture.expected[index]).allSatisfy({ abs($0 - $1) < 0.001 }) else {
      throw Failure.incorrectOutput("\(name): \(values), expected \(fixture.expected[index])")
    }
  }
}

@main
private struct SDKSourceLifetimeTest {
  static func main() {
    do {
      let args = CommandLine.arguments
      if args.count == 2, args[1] == "architecture" {
        print(ETCoreAISwiftModelLoader.deviceArchitectureName)
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
      let loader = ETCoreAISwiftModelLoader()
      if mode == "evict" {
        let bookmark = try Data(contentsOf: bookmarkURL)
        let _: Void = try waitForResult { completed in
          loader.evictModel(withBookmark: bookmark) { error in
            if let error { completed(.failure(error)) } else { completed(.success(())) }
          }
        }
        return
      }
      let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
      if mode == "specialize" {
        guard let model = try acquire({ loader.specializeModel(at: source, completion: $0) }) else {
          throw Failure.invalidResult
        }
        try model.copyBookmarkData().write(to: bookmarkURL, options: .atomic)
        try checkInference(model, fixture: fixture)
        try FileManager.default.removeItem(at: source)
        guard !FileManager.default.fileExists(atPath: source.path) else {
          throw Failure.unexpectedSource
        }
        try checkInference(model, fixture: fixture)
        print("Bridge inference passed after source removal")
      } else if mode == "restore" {
        guard !FileManager.default.fileExists(atPath: source.path) else {
          throw Failure.unexpectedSource
        }
        let bookmark = try Data(contentsOf: bookmarkURL)
        guard
          let model = try acquire({ loader.restoreModel(fromBookmark: bookmark, completion: $0) })
        else {
          exit(3)
        }
        try checkInference(model, fixture: fixture)
        print("Bridge restoration and inference passed without source")
      } else {
        throw Failure.invalidArguments
      }
    } catch {
      print("SDK source-lifetime test failed: \(error)")
      exit(1)
    }
  }
}
