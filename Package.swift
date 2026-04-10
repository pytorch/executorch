// swift-tools-version:5.9
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import PackageDescription

let version = "1.3.0.20260410"
let url = "https://ossci-ios.s3.amazonaws.com/executorch/"
let debug_suffix = "_debug"
let dependencies_suffix = "_with_dependencies"

func deliverables(_ dict: [String: [String: Any]]) -> [String: [String: Any]] {
  dict
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      result[key] = value
      result[key + debug_suffix] = value
    }
    .reduce(into: [String: [String: Any]]()) { result, pair in
      let (key, value) = pair
      var newValue = value
      if key.hasSuffix(debug_suffix) {
        for (k, v) in value where k.hasSuffix(debug_suffix) {
          let trimmed = String(k.dropLast(debug_suffix.count))
          newValue[trimmed] = v
        }
      }
      result[key] = newValue.filter { !$0.key.hasSuffix(debug_suffix) }
    }
}

let products = deliverables([
  "backend_coreml": [
    "sha256": "1115e6810dd5351cfe4d1025f5cb938919da74a6e21adafee70a5a342c16a41f",
    "sha256" + debug_suffix: "f267369f080ab8238102897c3e33b2cb50b4d0133b5d04154f9d774946a68526",
    "frameworks": [
      "Accelerate",
      "CoreML",
    ],
    "libraries": [
      "sqlite3",
    ],
  ],
  "backend_mps": [
    "sha256": "8254338ac5e8906c38f1e14bacbbfb8b6bf116d093b9f811023498ac285b3c70",
    "sha256" + debug_suffix: "410f93c80121c5520bad8c0af93642b2d1759bba45f315ff355509ad820154c0",
    "frameworks": [
      "Metal",
      "MetalPerformanceShaders",
      "MetalPerformanceShadersGraph",
    ],
  ],
  "backend_xnnpack": [
    "sha256": "337b1f323960ced81ab61e8aa5d4eea5633475eeed55849619893f915f81f22a",
    "sha256" + debug_suffix: "eca0664c1579966c8b484a2cd68cd3446c736f9e1383016a2b50ca958225126d",
    "targets": [
      "threadpool",
    ],
  ],
  "executorch": [
    "sha256": "5c04baaf3db9dce299fb247900bd5ed41443539c6f97688503d72a43abe54fbf",
    "sha256" + debug_suffix: "bc515156efcf4de50c7134b593aeb52c67e9dabf430b6e47c84cb1fc687845e4",
    "libraries": [
      "c++",
    ],
  ],
  "executorch_llm": [
    "sha256": "9acd6f67820dbb39d05c91b8b9591e0fe10e7eaf063ef8c285b1fb1989ced54b",
    "sha256" + debug_suffix: "ca080a2b39cea750760912225561f6960f4754f596bd4a96dd12bdd44bb1d4c6",
    "targets": [
      "executorch",
    ],
  ],
  "kernels_llm": [
    "sha256": "e8e4fca485a02879f453f8446ed4266b5ec14cff5fc6b1fa8c929a31bc05ce48",
    "sha256" + debug_suffix: "25ada3f46b2029ad1d463a84c0a78892b90e0a63be02380aee0d607e1716635c",
  ],
  "kernels_optimized": [
    "sha256": "2ec7735d7be57c427bfb21f22a29500e297829a6f47937885c83e104d34ba2a4",
    "sha256" + debug_suffix: "8a04efbf6e34abdde6df2d69a0df6ac35b74cc7d94680d55029738c2c224d5b2",
    "frameworks": [
      "Accelerate",
    ],
    "targets": [
      "threadpool",
    ],
  ],
  "kernels_quantized": [
    "sha256": "b64df55ef0b109968deef229e0490624aad202cfd75dfba4b88b36cee6ca17d8",
    "sha256" + debug_suffix: "235bf56bf547b101a9d1b4b7aaad86842b22c9aea86b53eafd1fb8d54bd3ac48",
  ],
  "kernels_torchao": [
    "sha256": "059e7960c809f1868df7abb2545895364b48aaa992dadb2617080468b47cb557",
    "sha256" + debug_suffix: "0041e9e89bc9dde1f4af07ac9d5d3f6f89a2294523f33147df0635177e12bb1a",
    "targets": [
      "threadpool",
    ],
  ],
])

let targets = deliverables([
  "threadpool": [
    "sha256": "8af6f9a97139ff4e47de081fffbf69c9cbb182a035cadbca4c9b23758ef62eae",
    "sha256" + debug_suffix: "03ec6f020bb63fcb0b1b5036f31f2085143f120519899dfc8bc840b1703e1a34",
  ],
])

let packageProducts: [Product] = products.keys.map { key -> Product in
  .library(name: key, targets: ["\(key)\(dependencies_suffix)"])
}.sorted { $0.name < $1.name }

var packageTargets: [Target] = []

for (key, value) in targets {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
}

for (key, value) in products {
  packageTargets.append(.binaryTarget(
    name: key,
    url: "\(url)\(key)-\(version).zip",
    checksum: value["sha256"] as? String ?? ""
  ))
  let target: Target = .target(
    name: "\(key)\(dependencies_suffix)",
    dependencies: ([key] + (value["targets"] as? [String] ?? []).map {
      key.hasSuffix(debug_suffix) ? $0 + debug_suffix : $0
    }).map { .target(name: $0) },
    path: ".Package.swift/\(key)",
    linkerSettings:
      (value["frameworks"] as? [String] ?? []).map { .linkedFramework($0) } +
      (value["libraries"] as? [String] ?? []).map { .linkedLibrary($0) }
  )
  packageTargets.append(target)
}

let package = Package(
  name: "executorch",
  platforms: [
    .iOS(.v17),
    .macOS(.v12),
  ],
  products: packageProducts,
  targets: packageTargets
)
